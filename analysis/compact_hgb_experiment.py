"""
Compact-HGB experiment for the DKV Mobility Azure ML interview case.

Goal
----
Compare the existing full 23-feature HistGradientBoosting model against a
compact HistGradientBoosting model using the stable feature subset selected
by sparsity_path_experiment.py.

Only the 80% training partition is used.
The 20% holdout test set is never loaded.

Both HGB candidates use:
- identical 5-fold stratified CV splits,
- identical model hyperparameters,
- identical preprocessing rules,
- the only difference is the input feature subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "default"

CATEGORICAL_FEATURES = [
    "sex",
    "education",
    "marriage",
]

NUMERIC_FEATURES = [
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

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def build_hgb_pipeline(
    selected_features: list[str],
    random_state: int,
) -> Pipeline:
    """Build HGB using exactly the requested original input features."""

    selected_set = set(
        selected_features
    )

    selected_numeric = [
        feature
        for feature in (
            NUMERIC_FEATURES
        )
        if feature in selected_set
    ]

    selected_categorical = [
        feature
        for feature in (
            CATEGORICAL_FEATURES
        )
        if feature in selected_set
    ]

    transformers = []

    if selected_numeric:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                selected_numeric,
            )
        )

    if selected_categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                selected_categorical,
            )
        )

    if not transformers:
        raise ValueError(
            "Feature subset is empty."
        )

    transformer = (
        ColumnTransformer(
            transformers=(
                transformers
            ),
            remainder="drop",
        )
    )

    return Pipeline(
        steps=[
            (
                "feature_transformer",
                transformer,
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


def calculate_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Calculate validation metrics."""

    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "brier": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),
    }


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
) -> dict:
    """Evaluate one HGB model on fixed shared CV folds."""

    fold_records = []

    for fold_number, (
        train_index,
        valid_index,
    ) in enumerate(
        cv_splits,
        start=1,
    ):
        X_train = X.iloc[
            train_index
        ]
        y_train = y.iloc[
            train_index
        ]
        X_valid = X.iloc[
            valid_index
        ]
        y_valid = y.iloc[
            valid_index
        ]

        pipeline.fit(
            X_train,
            y_train,
        )

        probability = (
            pipeline.predict_proba(
                X_valid
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_valid,
            probability,
        )

        fold_records.append(
            {
                "fold": fold_number,
                **metrics,
            }
        )

        print(
            f"{model_name} fold "
            f"{fold_number}: "
            f"ROC-AUC="
            f"{metrics['roc_auc']:.4f}, "
            f"PR-AUC="
            f"{metrics['pr_auc']:.4f}, "
            f"Brier="
            f"{metrics['brier']:.4f}"
        )

    summary = {}

    for metric in [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "brier",
    ]:
        values = np.array(
            [
                record[metric]
                for record in (
                    fold_records
                )
            ],
            dtype=float,
        )

        summary[
            f"{metric}_mean"
        ] = float(
            values.mean()
        )
        summary[
            f"{metric}_std"
        ] = float(
            values.std()
        )

    return {
        "folds": fold_records,
        "summary": summary,
    }


def build_fold_comparison(
    full_result: dict,
    compact_result: dict,
) -> pd.DataFrame:
    """Create paired fold-level comparison."""

    rows = []

    for full_fold, compact_fold in zip(
        full_result["folds"],
        compact_result["folds"],
    ):
        row = {
            "fold": (
                full_fold["fold"]
            )
        }

        for metric in [
            "accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "brier",
        ]:
            row[
                f"full_{metric}"
            ] = full_fold[
                metric
            ]
            row[
                f"compact_{metric}"
            ] = compact_fold[
                metric
            ]

            if metric == "brier":
                improvement = (
                    full_fold[metric]
                    - compact_fold[
                        metric
                    ]
                )
            else:
                improvement = (
                    compact_fold[
                        metric
                    ]
                    - full_fold[
                        metric
                    ]
                )

            row[
                f"compact_improvement_{metric}"
            ] = improvement

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def plot_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot paired full-vs-compact ROC-AUC."""

    x = np.arange(
        len(comparison_df)
    )
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        comparison_df[
            "full_roc_auc"
        ],
        width,
        label="Full HGB",
    )
    ax.bar(
        x + width / 2,
        comparison_df[
            "compact_roc_auc"
        ],
        width,
        label="Compact HGB",
    )

    ax.set_xticks(
        x,
        [
            f"Fold {value}"
            for value in (
                comparison_df[
                    "fold"
                ]
            )
        ],
    )
    ax.set_ylabel(
        "ROC-AUC"
    )
    ax.set_title(
        "Full vs Compact HGB – Paired 5-Fold CV"
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
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=str,
        default=(
            "data/processed/train/train.csv"
        ),
    )

    parser.add_argument(
        "--feature-set-json",
        type=str,
        default=(
            "outputs/sparsity_path/"
            "selected_feature_set.json"
        ),
    )

    parser.add_argument(
        "--feature-source",
        choices=[
            "stable",
            "refit",
        ],
        default="stable",
        help=(
            "Use stable CV features or "
            "features selected after full-train refit."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/compact_hgb"
        ),
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

    train_df = pd.read_csv(
        args.train_data
    )

    if (
        TARGET_COLUMN
        not in train_df.columns
    ):
        raise ValueError(
            f"Training data does not "
            f"contain '{TARGET_COLUMN}'."
        )

    X = train_df.drop(
        columns=[
            TARGET_COLUMN
        ]
    )
    y = train_df[
        TARGET_COLUMN
    ].astype(int)

    if (
        set(X.columns)
        != set(ALL_FEATURES)
    ):
        raise ValueError(
            "Training feature schema "
            "does not match expected schema."
        )

    feature_payload = json.loads(
        Path(
            args.feature_set_json
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        args.feature_source
        == "stable"
    ):
        compact_features = (
            feature_payload[
                "stable_features_80pct"
            ]
        )
    else:
        compact_features = (
            feature_payload[
                "refit_full_train_features"
            ]
        )

    if not compact_features:
        raise ValueError(
            "Selected compact feature "
            "set is empty."
        )

    unknown = sorted(
        set(compact_features)
        - set(ALL_FEATURES)
    )

    if unknown:
        raise ValueError(
            "Unknown features in subset: "
            f"{unknown}"
        )

    print(
        f"Training observations: "
        f"{len(train_df):,}"
    )
    print(
        f"Default rate: "
        f"{y.mean():.2%}"
    )
    print(
        f"Full model features: "
        f"{len(ALL_FEATURES)}"
    )
    print(
        f"Compact model features: "
        f"{len(compact_features)}"
    )
    print(
        "\nCompact feature set:"
    )
    for feature in (
        compact_features
    ):
        print(
            f"  - {feature}"
        )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=(
            args.random_state
        ),
    )

    cv_splits = list(
        cv.split(
            X,
            y,
        )
    )

    full_pipeline = (
        build_hgb_pipeline(
            ALL_FEATURES,
            args.random_state,
        )
    )

    compact_pipeline = (
        build_hgb_pipeline(
            compact_features,
            args.random_state,
        )
    )

    print(
        "\nEvaluating full HGB..."
    )
    full_result = (
        evaluate_model(
            "full_hgb",
            full_pipeline,
            X,
            y,
            cv_splits,
        )
    )

    print(
        "\nEvaluating compact HGB..."
    )
    compact_result = (
        evaluate_model(
            "compact_hgb",
            compact_pipeline,
            X,
            y,
            cv_splits,
        )
    )

    comparison_df = (
        build_fold_comparison(
            full_result,
            compact_result,
        )
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        output_dir
        / "fold_comparison.csv",
        index=False,
    )

    payload = {
        "feature_source": (
            args.feature_source
        ),
        "compact_features": (
            compact_features
        ),
        "full_feature_count": (
            len(ALL_FEATURES)
        ),
        "compact_feature_count": (
            len(compact_features)
        ),
        "full_hgb": (
            full_result
        ),
        "compact_hgb": (
            compact_result
        ),
    }

    with (
        output_dir
        / "compact_hgb_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    plot_comparison(
        comparison_df,
        output_dir
        / "full_vs_compact_roc_auc.png",
    )

    full_summary = (
        full_result[
            "summary"
        ]
    )
    compact_summary = (
        compact_result[
            "summary"
        ]
    )

    print(
        "\n\nFull vs Compact HGB"
    )
    print(
        "==================="
    )
    print(
        "Metric       Full HGB   Compact HGB   Compact Δ"
    )

    for metric in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "recall",
        "precision",
        "f1",
    ]:
        full_value = (
            full_summary[
                f"{metric}_mean"
            ]
        )
        compact_value = (
            compact_summary[
                f"{metric}_mean"
            ]
        )

        print(
            f"{metric:<12} "
            f"{full_value:>8.4f} "
            f"{compact_value:>13.4f} "
            f"{compact_value - full_value:>11.4f}"
        )

    full_brier = (
        full_summary[
            "brier_mean"
        ]
    )
    compact_brier = (
        compact_summary[
            "brier_mean"
        ]
    )

    print(
        f"{'brier':<12} "
        f"{full_brier:>8.4f} "
        f"{compact_brier:>13.4f} "
        f"{full_brier - compact_brier:>11.4f}"
    )

    roc_loss = (
        full_summary[
            "roc_auc_mean"
        ]
        - compact_summary[
            "roc_auc_mean"
        ]
    )

    print(
        f"\nFeature reduction: "
        f"{len(ALL_FEATURES)} -> "
        f"{len(compact_features)}"
    )
    print(
        f"ROC-AUC loss vs full HGB: "
        f"{roc_loss:.4f}"
    )
    print(
        f"\nArtifacts written to: "
        f"{output_dir}"
    )
    print(
        "\nInterpretation rule:"
        "\nA compact model is attractive only if the feature reduction is "
        "substantial and the CV loss is practically small."
        "\nDo not use the holdout to tune the subset."
    )


if __name__ == "__main__":
    main()
