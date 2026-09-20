"""
Feature-engineering experiment for the DKV Mobility Azure ML interview case.

Purpose:
Compare the SAME HistGradientBoosting model with and without deterministic
domain-engineered features, using ONLY the 80% training partition.

The 20% holdout test set is deliberately never loaded here.

Both candidates use:
- exactly the same 5 stratified CV folds,
- the same feature transformer,
- the same classifier hyperparameters,
- the same scoring metrics.

Therefore the only experimental treatment is the additional engineered
feature set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import (  # noqa: E402
    ENGINEERED_FEATURES,
    add_domain_features,
)


TARGET_COLUMN = "default"

CATEGORICAL_FEATURES = [
    "sex",
    "education",
    "marriage",
]

ORIGINAL_NUMERIC_FEATURES = [
    "limit_bal",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
]


def build_hgb_pipeline(
    numeric_features: list[str],
    random_state: int,
) -> Pipeline:
    """
    Build the same HGB pipeline used for both experiment arms.

    StandardScaler is intentionally retained here because the purpose of this
    experiment is to isolate the effect of feature engineering relative to the
    current baseline. Scaling is not required by HGB, but changing it here
    would introduce a second experimental difference.
    """

    feature_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "feature_transformer",
                feature_transformer,
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_pipeline(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
) -> dict:
    """Evaluate one candidate on the fixed shared CV folds."""

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "neg_brier": make_scorer(
            brier_score_loss,
            response_method="predict_proba",
            greater_is_better=False,
        ),
    }

    raw = cross_validate(
        model_pipeline,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    result = {
        "folds": {},
        "summary": {},
    }

    for metric_name in scoring:
        values = raw[f"test_{metric_name}"]

        if metric_name == "neg_brier":
            values = -values
            output_name = "brier"
        else:
            output_name = metric_name

        result["folds"][output_name] = [
            float(v) for v in values
        ]
        result["summary"][f"{output_name}_mean"] = float(
            np.mean(values)
        )
        result["summary"][f"{output_name}_std"] = float(
            np.std(values)
        )

    return result


def create_comparison_table(
    baseline_result: dict,
    engineered_result: dict,
) -> pd.DataFrame:
    """Create a fold-level paired comparison."""

    rows = []

    metrics = [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "brier",
    ]

    n_folds = len(
        baseline_result["folds"]["roc_auc"]
    )

    for fold_index in range(n_folds):
        row = {
            "fold": fold_index + 1,
        }

        for metric in metrics:
            baseline_value = (
                baseline_result["folds"][metric][fold_index]
            )
            engineered_value = (
                engineered_result["folds"][metric][fold_index]
            )

            row[f"baseline_{metric}"] = baseline_value
            row[f"engineered_{metric}"] = engineered_value

            # For Brier lower is better, so improvement has reversed sign.
            if metric == "brier":
                delta = baseline_value - engineered_value
            else:
                delta = engineered_value - baseline_value

            row[f"improvement_{metric}"] = delta

        rows.append(row)

    return pd.DataFrame(rows)


def plot_roc_auc_by_fold(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot paired fold-level ROC-AUC results."""

    x = np.arange(len(comparison_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        x - width / 2,
        comparison_df["baseline_roc_auc"],
        width,
        label="Original features",
    )
    ax.bar(
        x + width / 2,
        comparison_df["engineered_roc_auc"],
        width,
        label="Original + engineered",
    )

    ax.set_xticks(
        x,
        [f"Fold {i}" for i in comparison_df["fold"]],
    )
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        "Feature Engineering – Paired 5-Fold CV ROC-AUC"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train/train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/feature_experiment",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_data)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Training data does not contain '{TARGET_COLUMN}'."
        )

    X_original = train_df.drop(
        columns=[TARGET_COLUMN]
    )
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_engineered = add_domain_features(
        X_original
    )

    engineered_numeric_features = (
        ORIGINAL_NUMERIC_FEATURES
        + ENGINEERED_FEATURES
    )

    print(
        f"Training observations: {len(train_df):,}"
    )
    print(
        f"Default rate: {y_train.mean():.2%}"
    )
    print(
        f"Original features: {X_original.shape[1]}"
    )
    print(
        f"Engineered features added: "
        f"{len(ENGINEERED_FEATURES)}"
    )
    print(
        f"Total engineered candidate features: "
        f"{X_engineered.shape[1]}"
    )

    print("\nEngineered feature list:")
    for feature in ENGINEERED_FEATURES:
        print(f"  - {feature}")

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    # Materialize the fold indices once so both candidates use EXACTLY
    # the same observations in every train/validation fold.
    cv_splits = list(
        cv.split(
            X_original,
            y_train,
        )
    )

    baseline_pipeline = build_hgb_pipeline(
        ORIGINAL_NUMERIC_FEATURES,
        args.random_state,
    )

    engineered_pipeline = build_hgb_pipeline(
        engineered_numeric_features,
        args.random_state,
    )

    print(
        "\nEvaluating baseline HGB "
        "(original features only)..."
    )
    baseline_result = evaluate_pipeline(
        baseline_pipeline,
        X_original,
        y_train,
        cv_splits,
    )

    print(
        "Evaluating HGB "
        "(original + domain-engineered features)..."
    )
    engineered_result = evaluate_pipeline(
        engineered_pipeline,
        X_engineered,
        y_train,
        cv_splits,
    )

    comparison_df = create_comparison_table(
        baseline_result,
        engineered_result,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        output_dir / "fold_comparison.csv"
    )
    results_path = (
        output_dir / "feature_experiment_results.json"
    )
    plot_path = (
        output_dir / "roc_auc_by_fold.png"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    payload = {
        "engineered_features": ENGINEERED_FEATURES,
        "baseline": baseline_result,
        "engineered": engineered_result,
        "mean_improvement": {
            metric: float(
                comparison_df[
                    f"improvement_{metric}"
                ].mean()
            )
            for metric in [
                "accuracy",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
                "brier",
            ]
        },
        "fold_wins_roc_auc": int(
            (
                comparison_df[
                    "improvement_roc_auc"
                ] > 0
            ).sum()
        ),
    }

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    plot_roc_auc_by_fold(
        comparison_df,
        plot_path,
    )

    b = baseline_result["summary"]
    e = engineered_result["summary"]

    print("\n5-fold CV summary")
    print("-----------------")
    print(
        "Metric       Baseline HGB   Engineered HGB   Improvement"
    )

    for metric in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "recall",
        "precision",
        "f1",
    ]:
        baseline_mean = b[f"{metric}_mean"]
        engineered_mean = e[f"{metric}_mean"]
        improvement = (
            engineered_mean - baseline_mean
        )

        print(
            f"{metric:<12} "
            f"{baseline_mean:>12.4f} "
            f"{engineered_mean:>16.4f} "
            f"{improvement:>13.4f}"
        )

    brier_improvement = (
        b["brier_mean"] - e["brier_mean"]
    )
    print(
        f"{'brier':<12} "
        f"{b['brier_mean']:>12.4f} "
        f"{e['brier_mean']:>16.4f} "
        f"{brier_improvement:>13.4f}"
    )

    print(
        "\nROC-AUC fold wins for engineered model: "
        f"{payload['fold_wins_roc_auc']}/{args.cv_folds}"
    )

    print(
        "\nArtifacts written to: "
        f"{output_dir}"
    )
    print(
        "\nDecision rule:"
        "\nDo not use the holdout set here."
        "\nRetain engineered features only if the CV improvement is "
        "consistent across folds and practically meaningful, without "
        "materially worsening probability quality."
    )


if __name__ == "__main__":
    main()
