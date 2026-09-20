"""
Sparsity-path experiment on the EXPANDED 36-feature space.

DKV Mobility Azure ML interview case.

Goal
----
Start from:
- 23 original UCI features
- 13 deterministic, domain-engineered features

Then find the smallest sparse logistic model whose cross-validated
discrimination remains close to the best model.

Only the 80% training partition is used.
The 20% holdout test set is never loaded.

Method
------
1. Add deterministic domain features from src/features.py.
2. Evaluate a path of L1 and Elastic-Net logistic models.
3. Use identical stratified 5-fold CV splits for every configuration.
4. Record ROC-AUC, PR-AUC, Brier score, and number of selected SOURCE
   features for every configuration.
5. Apply the one-standard-error rule (default) or a user-defined tolerance.
6. Among near-optimal models, choose the sparsest configuration.
7. Measure feature-selection stability across folds.
8. Refit the selected sparse model on the full training partition.
9. Persist the selected feature set for the compact-HGB experiment.

Important
---------
Feature engineering is deterministic and row-wise. No statistics are learned
from the dataset before cross-validation.

The holdout test set is intentionally not used here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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

EXPANDED_NUMERIC_FEATURES = (
    ORIGINAL_NUMERIC_FEATURES
    + ENGINEERED_FEATURES
)

ALL_EXPANDED_FEATURES = (
    EXPANDED_NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

DEFAULT_C_GRID = [
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
]

DEFAULT_ELASTIC_L1_RATIOS = [
    0.5,
    0.8,
]


def build_feature_transformer() -> ColumnTransformer:
    """Build preprocessing for the expanded 36-feature logistic model."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                EXPANDED_NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_sparse_pipeline(
    penalty: str,
    c_value: float,
    random_state: int,
    l1_ratio: float | None = None,
) -> Pipeline:
    """Build one fixed sparse-model configuration."""

    if penalty == "l1":
        classifier = LogisticRegression(
            penalty="l1",
            C=c_value,
            solver="saga",
            max_iter=5000,
            random_state=random_state,
        )
    elif penalty == "elasticnet":
        if l1_ratio is None:
            raise ValueError(
                "Elastic Net requires l1_ratio."
            )
        classifier = LogisticRegression(
            penalty="elasticnet",
            C=c_value,
            l1_ratio=l1_ratio,
            solver="saga",
            max_iter=5000,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unsupported penalty: {penalty}"
        )

    return Pipeline(
        steps=[
            (
                "feature_transformer",
                build_feature_transformer(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Calculate validation metrics."""

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "brier": float(
            brier_score_loss(
                y_true,
                y_probability,
            )
        ),
    }


def transformed_to_source_feature(
    transformed_name: str,
) -> str:
    """
    Map transformed sklearn columns back to one source feature.

    Examples:
      numeric__pay_0 -> pay_0
      numeric__max_delinquency -> max_delinquency
      categorical__sex_2 -> sex
    """

    if transformed_name.startswith(
        "numeric__"
    ):
        return transformed_name.replace(
            "numeric__",
            "",
            1,
        )

    if transformed_name.startswith(
        "categorical__"
    ):
        remainder = transformed_name.replace(
            "categorical__",
            "",
            1,
        )

        for feature in sorted(
            CATEGORICAL_FEATURES,
            key=len,
            reverse=True,
        ):
            if (
                remainder == feature
                or remainder.startswith(
                    f"{feature}_"
                )
            ):
                return feature

    raise ValueError(
        "Could not map transformed feature "
        f"'{transformed_name}'."
    )


def extract_selection(
    fitted_pipeline: Pipeline,
    zero_tolerance: float = 1e-10,
) -> dict:
    """Extract non-zero transformed and source features."""

    transformer = (
        fitted_pipeline.named_steps[
            "feature_transformer"
        ]
    )
    classifier = (
        fitted_pipeline.named_steps[
            "classifier"
        ]
    )

    transformed_names = (
        transformer.get_feature_names_out()
    )
    coefficients = classifier.coef_[0]

    selected_transformed = []
    selected_source = set()
    coefficient_rows = []

    for name, coefficient in zip(
        transformed_names,
        coefficients,
    ):
        source_feature = (
            transformed_to_source_feature(
                name
            )
        )

        selected = (
            abs(coefficient)
            > zero_tolerance
        )

        coefficient_rows.append(
            {
                "transformed_feature": name,
                "source_feature": source_feature,
                "coefficient": float(
                    coefficient
                ),
                "abs_coefficient": float(
                    abs(coefficient)
                ),
                "selected": bool(
                    selected
                ),
                "feature_type": (
                    "engineered"
                    if source_feature
                    in ENGINEERED_FEATURES
                    else "original"
                ),
            }
        )

        if selected:
            selected_transformed.append(
                name
            )
            selected_source.add(
                source_feature
            )

    return {
        "selected_transformed": (
            selected_transformed
        ),
        "selected_source": sorted(
            selected_source
        ),
        "n_selected_transformed": len(
            selected_transformed
        ),
        "n_selected_source": len(
            selected_source
        ),
        "coefficient_rows": (
            coefficient_rows
        ),
    }


def create_candidate_grid(
    c_grid: list[float],
    elastic_l1_ratios: list[float],
) -> list[dict]:
    """Create all L1 and Elastic-Net path configurations."""

    candidates = []

    for c_value in c_grid:
        candidates.append(
            {
                "model_family": "l1",
                "C": c_value,
                "l1_ratio": None,
            }
        )

    for l1_ratio in elastic_l1_ratios:
        for c_value in c_grid:
            candidates.append(
                {
                    "model_family": (
                        "elastic_net"
                    ),
                    "C": c_value,
                    "l1_ratio": l1_ratio,
                }
            )

    return candidates


def candidate_id(
    candidate: dict,
) -> str:
    """Create a readable candidate ID."""

    if (
        candidate["model_family"]
        == "l1"
    ):
        return (
            f"l1_C={candidate['C']:g}"
        )

    return (
        "elastic_net_"
        f"l1ratio={candidate['l1_ratio']:g}_"
        f"C={candidate['C']:g}"
    )


def evaluate_candidate(
    candidate: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
    random_state: int,
) -> dict:
    """Evaluate one fixed sparse configuration."""

    fold_records = []
    selection_counter = Counter()

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

        pipeline = build_sparse_pipeline(
            penalty=(
                "l1"
                if candidate[
                    "model_family"
                ] == "l1"
                else "elasticnet"
            ),
            c_value=candidate["C"],
            l1_ratio=(
                candidate[
                    "l1_ratio"
                ]
            ),
            random_state=(
                random_state
                + fold_number
            ),
        )

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

        selection = extract_selection(
            pipeline
        )

        selection_counter.update(
            selection[
                "selected_source"
            ]
        )

        fold_records.append(
            {
                "fold": fold_number,
                "metrics": metrics,
                "n_selected_source": (
                    selection[
                        "n_selected_source"
                    ]
                ),
                "n_selected_transformed": (
                    selection[
                        "n_selected_transformed"
                    ]
                ),
                "selected_source": (
                    selection[
                        "selected_source"
                    ]
                ),
            }
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
                fold["metrics"][metric]
                for fold in fold_records
            ],
            dtype=float,
        )

        summary[
            f"{metric}_mean"
        ] = float(values.mean())
        summary[
            f"{metric}_std"
        ] = float(values.std())

    selected_counts = np.array(
        [
            fold[
                "n_selected_source"
            ]
            for fold in fold_records
        ],
        dtype=float,
    )

    summary[
        "selected_source_mean"
    ] = float(selected_counts.mean())
    summary[
        "selected_source_std"
    ] = float(selected_counts.std())
    summary[
        "selected_source_min"
    ] = int(selected_counts.min())
    summary[
        "selected_source_max"
    ] = int(selected_counts.max())

    stability = {
        feature: (
            selection_counter[
                feature
            ]
            / len(cv_splits)
        )
        for feature in (
            ALL_EXPANDED_FEATURES
        )
    }

    return {
        "candidate": candidate,
        "folds": fold_records,
        "summary": summary,
        "selection_stability": stability,
    }


def select_parsimonious_candidate(
    path_df: pd.DataFrame,
    selection_rule: str,
    roc_tolerance: float,
    n_folds: int,
) -> tuple[pd.Series, float, str]:
    """Choose the sparsest configuration inside the near-optimal region."""

    best_index = (
        path_df["roc_auc_mean"]
        .idxmax()
    )
    best_row = path_df.loc[
        best_index
    ]

    if selection_rule == "one_se":
        best_se = (
            best_row["roc_auc_std"]
            / math.sqrt(n_folds)
        )
        threshold = (
            best_row["roc_auc_mean"]
            - best_se
        )
        description = (
            "one-standard-error rule"
        )
    elif selection_rule == "tolerance":
        threshold = (
            best_row["roc_auc_mean"]
            - roc_tolerance
        )
        description = (
            f"ROC-AUC tolerance "
            f"{roc_tolerance:.4f}"
        )
    else:
        raise ValueError(
            "selection_rule must be "
            "'one_se' or 'tolerance'."
        )

    near_optimal = path_df[
        path_df[
            "roc_auc_mean"
        ] >= threshold
    ].copy()

    near_optimal = (
        near_optimal.sort_values(
            by=[
                "selected_source_mean",
                "roc_auc_mean",
                "pr_auc_mean",
                "brier_mean",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
    )

    return (
        near_optimal.iloc[0],
        float(threshold),
        description,
    )


def plot_sparsity_frontier(
    path_df: pd.DataFrame,
    selected_id: str,
    output_path: Path,
) -> None:
    """Plot CV ROC-AUC against mean number of selected source features."""

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for family, group in (
        path_df.groupby(
            "model_label"
        )
    ):
        group = (
            group.sort_values(
                "selected_source_mean"
            )
        )

        ax.plot(
            group[
                "selected_source_mean"
            ],
            group[
                "roc_auc_mean"
            ],
            marker="o",
            label=family,
        )

    selected_row = (
        path_df[
            path_df[
                "candidate_id"
            ] == selected_id
        ]
        .iloc[0]
    )

    ax.scatter(
        [
            selected_row[
                "selected_source_mean"
            ]
        ],
        [
            selected_row[
                "roc_auc_mean"
            ]
        ],
        s=130,
        marker="*",
        label="Selected parsimonious model",
    )

    ax.set_xlabel(
        "Mean selected source features"
    )
    ax.set_ylabel(
        "5-fold CV ROC-AUC"
    )
    ax.set_title(
        "Expanded Feature Space – Sparsity/Performance Frontier"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metric_vs_c(
    path_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot ROC-AUC along the regularization path."""

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for label, group in (
        path_df.groupby(
            "model_label"
        )
    ):
        group = group.sort_values(
            "C"
        )

        ax.plot(
            group["C"],
            group[
                "roc_auc_mean"
            ],
            marker="o",
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "C (larger = weaker regularization)"
    )
    ax.set_ylabel(
        "5-fold CV ROC-AUC"
    )
    ax.set_title(
        "Expanded Feature Space – Regularization Path"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_float_list(
    value: str,
) -> list[float]:
    """Parse comma-separated floats."""

    return [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


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
        "--output-dir",
        type=str,
        default=(
            "outputs/sparsity_path_36"
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

    parser.add_argument(
        "--c-grid",
        type=str,
        default=",".join(
            str(value)
            for value in DEFAULT_C_GRID
        ),
    )

    parser.add_argument(
        "--elastic-l1-ratios",
        type=str,
        default=",".join(
            str(value)
            for value in (
                DEFAULT_ELASTIC_L1_RATIOS
            )
        ),
    )

    parser.add_argument(
        "--selection-rule",
        choices=[
            "one_se",
            "tolerance",
        ],
        default="one_se",
    )

    parser.add_argument(
        "--roc-tolerance",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.80,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    c_grid = parse_float_list(
        args.c_grid
    )
    elastic_ratios = (
        parse_float_list(
            args.elastic_l1_ratios
        )
    )

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

    expected_raw_features = set(
        ORIGINAL_NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    if set(X_raw.columns) != expected_raw_features:
        raise ValueError(
            "Raw training feature schema "
            "does not match expected schema."
        )

    X = add_domain_features(
        X_raw
    )

    if set(X.columns) != set(
        ALL_EXPANDED_FEATURES
    ):
        raise ValueError(
            "Expanded feature schema "
            "does not match expected 36 features."
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
        f"Original features: "
        f"{X_raw.shape[1]}"
    )
    print(
        f"Engineered features: "
        f"{len(ENGINEERED_FEATURES)}"
    )
    print(
        f"Expanded feature space: "
        f"{X.shape[1]}"
    )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    cv_splits = list(
        cv.split(
            X,
            y,
        )
    )

    candidates = create_candidate_grid(
        c_grid,
        elastic_ratios,
    )

    print(
        f"Evaluating "
        f"{len(candidates)} "
        f"sparse configurations..."
    )

    results = []
    path_rows = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        cid = candidate_id(
            candidate
        )

        print(
            f"\n[{index}/{len(candidates)}] "
            f"{cid}"
        )

        result = evaluate_candidate(
            candidate,
            X,
            y,
            cv_splits,
            args.random_state,
        )

        results.append(
            result
        )

        summary = result[
            "summary"
        ]

        model_label = (
            "L1"
            if candidate[
                "model_family"
            ] == "l1"
            else (
                "Elastic Net "
                f"(l1_ratio="
                f"{candidate['l1_ratio']:g})"
            )
        )

        path_rows.append(
            {
                "candidate_id": cid,
                "model_family": (
                    candidate[
                        "model_family"
                    ]
                ),
                "model_label": model_label,
                "C": candidate["C"],
                "l1_ratio": (
                    candidate[
                        "l1_ratio"
                    ]
                ),
                **summary,
            }
        )

        print(
            f"ROC-AUC="
            f"{summary['roc_auc_mean']:.4f} "
            f"(+/- "
            f"{summary['roc_auc_std']:.4f}), "
            f"PR-AUC="
            f"{summary['pr_auc_mean']:.4f}, "
            f"Brier="
            f"{summary['brier_mean']:.4f}, "
            f"selected source features="
            f"{summary['selected_source_mean']:.1f}"
        )

    path_df = pd.DataFrame(
        path_rows
    )

    selected_row, threshold, (
        selection_description
    ) = (
        select_parsimonious_candidate(
            path_df,
            args.selection_rule,
            args.roc_tolerance,
            args.cv_folds,
        )
    )

    selected_id = str(
        selected_row[
            "candidate_id"
        ]
    )

    selected_result = next(
        result
        for result in results
        if candidate_id(
            result[
                "candidate"
            ]
        ) == selected_id
    )

    stability = (
        selected_result[
            "selection_stability"
        ]
    )

    stable_features = sorted(
        [
            feature
            for feature, rate
            in stability.items()
            if rate
            >= args.stability_threshold
        ]
    )

    selected_candidate = (
        selected_result[
            "candidate"
        ]
    )

    final_sparse_pipeline = (
        build_sparse_pipeline(
            penalty=(
                "l1"
                if selected_candidate[
                    "model_family"
                ] == "l1"
                else "elasticnet"
            ),
            c_value=(
                selected_candidate[
                    "C"
                ]
            ),
            l1_ratio=(
                selected_candidate[
                    "l1_ratio"
                ]
            ),
            random_state=args.random_state,
        )
    )

    final_sparse_pipeline.fit(
        X,
        y,
    )

    full_train_selection = (
        extract_selection(
            final_sparse_pipeline
        )
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_df.to_csv(
        output_dir
        / "sparsity_path_36.csv",
        index=False,
    )

    stability_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "feature_type": (
                    "engineered"
                    if feature
                    in ENGINEERED_FEATURES
                    else "original"
                ),
                "selection_rate": (
                    stability[
                        feature
                    ]
                ),
                "selected_folds": int(
                    round(
                        stability[
                            feature
                        ]
                        * args.cv_folds
                    )
                ),
                "total_folds": (
                    args.cv_folds
                ),
                "stable": (
                    stability[
                        feature
                    ]
                    >= args.stability_threshold
                ),
            }
            for feature in (
                ALL_EXPANDED_FEATURES
            )
        ]
    ).sort_values(
        by=[
            "selection_rate",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    )

    stability_df.to_csv(
        output_dir
        / "selected_model_feature_stability_36.csv",
        index=False,
    )

    pd.DataFrame(
        full_train_selection[
            "coefficient_rows"
        ]
    ).to_csv(
        output_dir
        / "selected_sparse_model_coefficients_36.csv",
        index=False,
    )

    feature_payload = {
        "feature_space": (
            "23 original + 13 engineered"
        ),
        "selection_rule": (
            args.selection_rule
        ),
        "selection_description": (
            selection_description
        ),
        "roc_auc_acceptance_threshold": (
            threshold
        ),
        "selected_candidate_id": (
            selected_id
        ),
        "selected_candidate": (
            selected_candidate
        ),
        "selected_candidate_cv_summary": (
            selected_result[
                "summary"
            ]
        ),
        "stability_threshold": (
            args.stability_threshold
        ),
        "stable_features_80pct": (
            stable_features
        ),
        "refit_full_train_features": (
            full_train_selection[
                "selected_source"
            ]
        ),
        "engineered_features_available": (
            ENGINEERED_FEATURES
        ),
    }

    with (
        output_dir
        / "selected_feature_set_36.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_payload,
            file,
            indent=2,
        )

    with (
        output_dir
        / "all_path_results_36.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    joblib.dump(
        final_sparse_pipeline,
        output_dir
        / "selected_sparse_model_36.joblib",
    )

    plot_sparsity_frontier(
        path_df,
        selected_id,
        output_dir
        / "sparsity_performance_frontier_36.png",
    )

    plot_metric_vs_c(
        path_df,
        output_dir
        / "regularization_path_roc_auc_36.png",
    )

    best_row = path_df.loc[
        path_df[
            "roc_auc_mean"
        ].idxmax()
    ]

    stable_engineered = [
        feature
        for feature in stable_features
        if feature in ENGINEERED_FEATURES
    ]

    stable_original = [
        feature
        for feature in stable_features
        if feature not in ENGINEERED_FEATURES
    ]

    print(
        "\n\nExpanded-space sparsity result"
    )
    print(
        "================================"
    )
    print(
        f"Best observed CV ROC-AUC: "
        f"{best_row['roc_auc_mean']:.4f} "
        f"({best_row['candidate_id']})"
    )
    print(
        f"Near-optimal threshold "
        f"({selection_description}): "
        f"{threshold:.4f}"
    )

    print(
        "\nSelected parsimonious model:"
    )
    print(
        f"  {selected_id}"
    )
    print(
        f"  ROC-AUC: "
        f"{selected_row['roc_auc_mean']:.4f}"
    )
    print(
        f"  PR-AUC:  "
        f"{selected_row['pr_auc_mean']:.4f}"
    )
    print(
        f"  Brier:   "
        f"{selected_row['brier_mean']:.4f}"
    )
    print(
        f"  Mean selected source features: "
        f"{selected_row['selected_source_mean']:.1f}"
    )

    print(
        f"\nStable features "
        f"(>={args.stability_threshold:.0%} folds): "
        f"{len(stable_features)}"
    )

    print(
        f"\nOriginal stable features "
        f"({len(stable_original)}):"
    )
    for feature in stable_original:
        print(
            f"  - {feature}: "
            f"{stability[feature]:.0%}"
        )

    print(
        f"\nEngineered stable features "
        f"({len(stable_engineered)}):"
    )
    for feature in stable_engineered:
        print(
            f"  - {feature}: "
            f"{stability[feature]:.0%}"
        )

    print(
        "\nFeatures selected after refit "
        f"on all training data: "
        f"{len(full_train_selection['selected_source'])}"
    )
    for feature in (
        full_train_selection[
            "selected_source"
        ]
    ):
        feature_type = (
            "engineered"
            if feature
            in ENGINEERED_FEATURES
            else "original"
        )
        print(
            f"  - {feature} "
            f"[{feature_type}]"
        )

    print(
        f"\nArtifacts written to: "
        f"{output_dir}"
    )

    print(
        "\nNext step:"
        "\nRun compact_hgb_experiment.py using selected_feature_set_36.json."
        "\nDo NOT use the holdout to choose the feature subset."
    )


if __name__ == "__main__":
    main()
