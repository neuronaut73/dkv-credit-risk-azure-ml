"""
Compare three HGB variants on the same 5-fold CV splits:

1. Full raw HGB:       23 original features
2. Expanded HGB:       23 original + 13 engineered = 36 features
3. Compact HGB:        stable subset selected from the 36-feature space

Only the 80% training partition is used.
The 20% holdout test set is never loaded.

The feature engineering is deterministic and row-wise.
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

RAW_FEATURES = (
    ORIGINAL_NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

EXPANDED_FEATURES = (
    ORIGINAL_NUMERIC_FEATURES
    + ENGINEERED_FEATURES
    + CATEGORICAL_FEATURES
)


def build_hgb_pipeline(
    selected_features: list[str],
    random_state: int,
) -> Pipeline:
    """Build HGB using exactly the requested source features."""

    selected_set = set(
        selected_features
    )

    numeric_pool = (
        ORIGINAL_NUMERIC_FEATURES
        + ENGINEERED_FEATURES
    )

    selected_numeric = [
        feature
        for feature in numeric_pool
        if feature in selected_set
    ]

    selected_categorical = [
        feature
        for feature in CATEGORICAL_FEATURES
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
            transformers=transformers,
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
    """Evaluate one HGB candidate on the shared CV folds."""

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
                for record in fold_records
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


def build_comparison_table(
    results: dict,
) -> pd.DataFrame:
    """Create a fold-level table for all three HGB variants."""

    rows = []

    n_folds = len(
        results[
            "full_raw_hgb"
        ][
            "folds"
        ]
    )

    for fold_index in range(
        n_folds
    ):
        row = {
            "fold": fold_index + 1
        }

        for model_name, result in (
            results.items()
        ):
            fold = result[
                "folds"
            ][fold_index]

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
                    f"{model_name}_{metric}"
                ] = fold[
                    metric
                ]

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def plot_roc_auc_comparison(
    results: dict,
    output_path: Path,
) -> None:
    """Plot mean ROC-AUC with fold variability."""

    labels = [
        "23 raw",
        "36 expanded",
        "compact selected",
    ]

    model_keys = [
        "full_raw_hgb",
        "expanded_hgb",
        "compact_hgb",
    ]

    means = [
        results[key][
            "summary"
        ][
            "roc_auc_mean"
        ]
        for key in model_keys
    ]

    stds = [
        results[key][
            "summary"
        ][
            "roc_auc_std"
        ]
        for key in model_keys
    ]

    x = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
    )

    ax.set_xticks(
        x,
        labels,
    )
    ax.set_ylabel(
        "5-fold CV ROC-AUC"
    )
    ax.set_title(
        "HGB Comparison – Raw vs Expanded vs Compact"
    )

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
            "outputs/sparsity_path_36/"
            "selected_feature_set_36.json"
        ),
    )

    parser.add_argument(
        "--feature-source",
        choices=[
            "stable",
            "refit",
        ],
        default="stable",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/compact_hgb_36"
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

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Training data does not "
            f"contain '{TARGET_COLUMN}'."
        )

    X_raw = train_df.drop(
        columns=[TARGET_COLUMN]
    )
    y = train_df[
        TARGET_COLUMN
    ].astype(int)

    if set(X_raw.columns) != set(
        RAW_FEATURES
    ):
        raise ValueError(
            "Raw training feature schema "
            "does not match expected schema."
        )

    X_expanded = (
        add_domain_features(
            X_raw
        )
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
        - set(EXPANDED_FEATURES)
    )

    if unknown:
        raise ValueError(
            "Unknown features in compact set: "
            f"{unknown}"
        )

    compact_engineered = [
        feature
        for feature in compact_features
        if feature in ENGINEERED_FEATURES
    ]

    compact_original = [
        feature
        for feature in compact_features
        if feature not in ENGINEERED_FEATURES
    ]

    print(
        f"Training observations: "
        f"{len(train_df):,}"
    )
    print(
        f"Default rate: "
        f"{y.mean():.2%}"
    )
    print(
        f"Raw features: "
        f"{len(RAW_FEATURES)}"
    )
    print(
        f"Expanded features: "
        f"{len(EXPANDED_FEATURES)}"
    )
    print(
        f"Compact selected features: "
        f"{len(compact_features)}"
    )

    print(
        f"\nCompact original features "
        f"({len(compact_original)}):"
    )
    for feature in compact_original:
        print(
            f"  - {feature}"
        )

    print(
        f"\nCompact engineered features "
        f"({len(compact_engineered)}):"
    )
    for feature in compact_engineered:
        print(
            f"  - {feature}"
        )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    cv_splits = list(
        cv.split(
            X_raw,
            y,
        )
    )

    full_raw_pipeline = (
        build_hgb_pipeline(
            RAW_FEATURES,
            args.random_state,
        )
    )

    expanded_pipeline = (
        build_hgb_pipeline(
            EXPANDED_FEATURES,
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
        "\nEvaluating full raw HGB "
        "(23 features)..."
    )
    full_raw_result = (
        evaluate_model(
            "full_raw_hgb",
            full_raw_pipeline,
            X_expanded,
            y,
            cv_splits,
        )
    )

    print(
        "\nEvaluating expanded HGB "
        "(36 features)..."
    )
    expanded_result = (
        evaluate_model(
            "expanded_hgb",
            expanded_pipeline,
            X_expanded,
            y,
            cv_splits,
        )
    )

    print(
        "\nEvaluating compact HGB "
        "(selected from 36)..."
    )
    compact_result = (
        evaluate_model(
            "compact_hgb",
            compact_pipeline,
            X_expanded,
            y,
            cv_splits,
        )
    )

    results = {
        "full_raw_hgb": (
            full_raw_result
        ),
        "expanded_hgb": (
            expanded_result
        ),
        "compact_hgb": (
            compact_result
        ),
    }

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df = (
        build_comparison_table(
            results
        )
    )

    comparison_df.to_csv(
        output_dir
        / "fold_comparison_3models.csv",
        index=False,
    )

    payload = {
        "feature_source": (
            args.feature_source
        ),
        "raw_feature_count": (
            len(RAW_FEATURES)
        ),
        "expanded_feature_count": (
            len(EXPANDED_FEATURES)
        ),
        "compact_feature_count": (
            len(compact_features)
        ),
        "compact_features": (
            compact_features
        ),
        "compact_original_features": (
            compact_original
        ),
        "compact_engineered_features": (
            compact_engineered
        ),
        **results,
    }

    with (
        output_dir
        / "hgb_3model_comparison.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    plot_roc_auc_comparison(
        results,
        output_dir
        / "hgb_3model_roc_auc.png",
    )

    print(
        "\n\nHGB comparison"
    )
    print(
        "=============="
    )
    print(
        "Metric       23 raw     36 expanded   compact"
    )

    for metric in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "recall",
        "precision",
        "f1",
    ]:
        raw_value = (
            full_raw_result[
                "summary"
            ][
                f"{metric}_mean"
            ]
        )
        expanded_value = (
            expanded_result[
                "summary"
            ][
                f"{metric}_mean"
            ]
        )
        compact_value = (
            compact_result[
                "summary"
            ][
                f"{metric}_mean"
            ]
        )

        print(
            f"{metric:<12} "
            f"{raw_value:>8.4f} "
            f"{expanded_value:>13.4f} "
            f"{compact_value:>10.4f}"
        )

    raw_brier = (
        full_raw_result[
            "summary"
        ][
            "brier_mean"
        ]
    )
    expanded_brier = (
        expanded_result[
            "summary"
        ][
            "brier_mean"
        ]
    )
    compact_brier = (
        compact_result[
            "summary"
        ][
            "brier_mean"
        ]
    )

    print(
        f"{'brier':<12} "
        f"{raw_brier:>8.4f} "
        f"{expanded_brier:>13.4f} "
        f"{compact_brier:>10.4f}"
    )

    raw_roc = (
        full_raw_result[
            "summary"
        ][
            "roc_auc_mean"
        ]
    )
    compact_roc = (
        compact_result[
            "summary"
        ][
            "roc_auc_mean"
        ]
    )

    print(
        f"\nFeature reduction vs expanded: "
        f"{len(EXPANDED_FEATURES)} -> "
        f"{len(compact_features)}"
    )
    print(
        f"Feature reduction vs raw: "
        f"{len(RAW_FEATURES)} -> "
        f"{len(compact_features)}"
    )
    print(
        f"Compact ROC-AUC delta vs 23 raw: "
        f"{compact_roc - raw_roc:+.4f}"
    )

    print(
        f"\nArtifacts written to: "
        f"{output_dir}"
    )

    print(
        "\nInterpretation:"
        "\n- 23 raw HGB remains the current benchmark."
        "\n- 36 expanded HGB tests whether engineering adds signal."
        "\n- Compact HGB tests whether engineered summaries can replace "
        "multiple raw monthly variables."
        "\n- Do not use the holdout to tune the subset."
    )


if __name__ == "__main__":
    main()
