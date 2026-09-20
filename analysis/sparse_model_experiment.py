"""
Sparse linear model experiment for the DKV Mobility Azure ML interview case.

Purpose
-------
Compare regularized logistic-regression variants under leakage-safe nested
cross-validation using ONLY the 80% training partition:

- L2 logistic regression (ridge-like reference)
- L1 logistic regression (lasso-like sparse model)
- Elastic Net logistic regression (sparse + more stable under correlation)

The outer 5-fold CV estimates generalization performance.
Within every outer-training fold, an inner CV selects hyperparameters.

This script also measures feature-selection stability for L1 and Elastic Net:
how often each ORIGINAL input feature is retained across outer folds.

Important
---------
- The untouched 20% holdout test set is NEVER loaded here.
- Scaling and one-hot encoding are fitted inside the CV pipeline.
- Hyperparameters are tuned only inside each outer-training fold.
- Feature-selection stability is descriptive and based only on training CV.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


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

ALL_ORIGINAL_FEATURES = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)


def build_feature_transformer() -> ColumnTransformer:
    """
    Build leakage-safe transformations for logistic regression.

    Numeric variables are standardized because regularization depends on
    coefficient scale.

    Categorical variables use reference coding (drop="first") to avoid
    redundant dummy columns and to make coefficients easier to interpret.
    """

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
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


def build_model_specifications(
    random_state: int,
) -> dict:
    """
    Define model pipelines and inner-CV hyperparameter grids.
    """

    l2_pipeline = Pipeline(
        steps=[
            (
                "feature_transformer",
                build_feature_transformer(),
            ),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    l1_pipeline = Pipeline(
        steps=[
            (
                "feature_transformer",
                build_feature_transformer(),
            ),
            (
                "classifier",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    elastic_net_pipeline = Pipeline(
        steps=[
            (
                "feature_transformer",
                build_feature_transformer(),
            ),
            (
                "classifier",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Moderate grids: enough to compare sparsity/performance without turning
    # an interview case into a large hyperparameter-search exercise.
    return {
        "logistic_l2": {
            "pipeline": l2_pipeline,
            "param_grid": {
                "classifier__C": [
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1.0,
                    5.0,
                    10.0,
                ],
            },
        },
        "logistic_l1": {
            "pipeline": l1_pipeline,
            "param_grid": {
                "classifier__C": [
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.5,
                    1.0,
                ],
            },
        },
        "logistic_elastic_net": {
            "pipeline": elastic_net_pipeline,
            "param_grid": {
                "classifier__C": [
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.5,
                    1.0,
                ],
                "classifier__l1_ratio": [
                    0.2,
                    0.5,
                    0.8,
                ],
            },
        },
    }


def calculate_metrics(
    y_true: pd.Series,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Calculate outer-fold validation metrics."""

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


def transformed_to_original_feature(
    transformed_name: str,
) -> str:
    """
    Map a transformed feature name back to its original input feature.

    Examples:
      numeric__pay_0       -> pay_0
      categorical__sex_2   -> sex
      categorical__education_3 -> education
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

        # Match the longest categorical feature name first.
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
        f"'{transformed_name}' to an original feature."
    )


def extract_selected_features(
    fitted_pipeline: Pipeline,
    zero_tolerance: float = 1e-10,
) -> dict:
    """
    Extract selected transformed/original features from a fitted model.

    For L1/Elastic Net, zero coefficients represent excluded transformed
    features. For L2, practically all coefficients remain non-zero.
    """

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

    if len(
        transformed_names
    ) != len(coefficients):
        raise ValueError(
            "Transformed feature names and "
            "coefficient vector differ in length."
        )

    selected_transformed = []
    selected_original = set()

    coefficient_rows = []

    for name, coefficient in zip(
        transformed_names,
        coefficients,
    ):
        original = (
            transformed_to_original_feature(
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
                "original_feature": original,
                "coefficient": float(
                    coefficient
                ),
                "abs_coefficient": float(
                    abs(coefficient)
                ),
                "selected": bool(
                    selected
                ),
            }
        )

        if selected:
            selected_transformed.append(
                name
            )
            selected_original.add(
                original
            )

    return {
        "selected_transformed": (
            selected_transformed
        ),
        "selected_original": sorted(
            selected_original
        ),
        "coefficient_rows": (
            coefficient_rows
        ),
        "n_selected_transformed": len(
            selected_transformed
        ),
        "n_selected_original": len(
            selected_original
        ),
    }


def summarize_model_results(
    fold_records: list[dict],
) -> dict:
    """Summarize outer-fold performance."""

    metrics = [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "brier",
    ]

    summary = {}

    for metric in metrics:
        values = np.array(
            [
                record["metrics"][metric]
                for record in fold_records
            ],
            dtype=float,
        )

        summary[f"{metric}_mean"] = float(
            values.mean()
        )
        summary[f"{metric}_std"] = float(
            values.std()
        )

    selected_counts = np.array(
        [
            record[
                "selection"
            ][
                "n_selected_original"
            ]
            for record in fold_records
        ],
        dtype=float,
    )

    summary[
        "selected_original_features_mean"
    ] = float(
        selected_counts.mean()
    )
    summary[
        "selected_original_features_std"
    ] = float(
        selected_counts.std()
    )

    return summary


def calculate_selection_stability(
    fold_records: list[dict],
    n_outer_folds: int,
) -> pd.DataFrame:
    """
    Count how often each original feature is selected across outer folds.
    """

    counter = Counter()

    for record in fold_records:
        counter.update(
            record[
                "selection"
            ][
                "selected_original"
            ]
        )

    rows = []

    for feature in ALL_ORIGINAL_FEATURES:
        selected_folds = counter[
            feature
        ]

        rows.append(
            {
                "feature": feature,
                "selected_folds": (
                    selected_folds
                ),
                "total_folds": (
                    n_outer_folds
                ),
                "selection_rate": (
                    selected_folds
                    / n_outer_folds
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "selected_folds",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def plot_model_performance(
    summaries: dict,
    output_path: Path,
) -> None:
    """Plot outer-CV ROC-AUC for the three regularization strategies."""

    labels = list(
        summaries.keys()
    )
    means = [
        summaries[label][
            "roc_auc_mean"
        ]
        for label in labels
    ]
    errors = [
        summaries[label][
            "roc_auc_std"
        ]
        for label in labels
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    x = np.arange(
        len(labels)
    )

    ax.bar(
        x,
        means,
        yerr=errors,
        capsize=5,
    )

    ax.set_xticks(
        x,
        [
            "L2",
            "L1",
            "Elastic Net",
        ],
    )
    ax.set_ylabel(
        "Outer-CV ROC-AUC"
    )
    ax.set_title(
        "Regularized Logistic Models – Nested 5-Fold CV"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_selection_stability(
    stability_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot original-feature selection frequency across outer folds."""

    plot_df = (
        stability_df
        .sort_values(
            [
                "selected_folds",
                "feature",
            ],
            ascending=[
                True,
                True,
            ],
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.barh(
        plot_df["feature"],
        plot_df[
            "selection_rate"
        ],
    )

    ax.set_xlim(
        0,
        1,
    )
    ax.set_xlabel(
        "Fraction of outer folds in which feature was retained"
    )
    ax.set_ylabel(
        "Original input feature"
    )
    ax.set_title(
        f"Selection Stability – {model_name}"
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
        "--output-dir",
        type=str,
        default=(
            "outputs/sparse_model_experiment"
        ),
    )

    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
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
            f"Training data does not contain "
            f"'{TARGET_COLUMN}'."
        )

    X = train_df.drop(
        columns=[
            TARGET_COLUMN
        ]
    )
    y = train_df[
        TARGET_COLUMN
    ].astype(int)

    expected_features = set(
        ALL_ORIGINAL_FEATURES
    )

    if (
        set(X.columns)
        != expected_features
    ):
        raise ValueError(
            "Training feature schema "
            "does not match expected schema."
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
        f"Original input features: "
        f"{X.shape[1]}"
    )

    outer_cv = (
        StratifiedKFold(
            n_splits=args.outer_folds,
            shuffle=True,
            random_state=(
                args.random_state
            ),
        )
    )

    # Materialize once so every model sees exactly the same outer folds.
    outer_splits = list(
        outer_cv.split(
            X,
            y,
        )
    )

    model_specs = (
        build_model_specifications(
            args.random_state
        )
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}
    summaries = {}
    stability_tables = {}

    for (
        model_name,
        specification,
    ) in model_specs.items():

        print(
            "\n================================"
        )
        print(
            f"Evaluating: {model_name}"
        )
        print(
            "================================"
        )

        fold_records = []

        for (
            fold_number,
            (
                train_index,
                valid_index,
            ),
        ) in enumerate(
            outer_splits,
            start=1,
        ):

            X_outer_train = (
                X.iloc[
                    train_index
                ]
            )
            y_outer_train = (
                y.iloc[
                    train_index
                ]
            )
            X_outer_valid = (
                X.iloc[
                    valid_index
                ]
            )
            y_outer_valid = (
                y.iloc[
                    valid_index
                ]
            )

            inner_cv = (
                StratifiedKFold(
                    n_splits=(
                        args.inner_folds
                    ),
                    shuffle=True,
                    random_state=(
                        args.random_state
                        + fold_number
                    ),
                )
            )

            search = GridSearchCV(
                estimator=(
                    specification[
                        "pipeline"
                    ]
                ),
                param_grid=(
                    specification[
                        "param_grid"
                    ]
                ),
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )

            search.fit(
                X_outer_train,
                y_outer_train,
            )

            best_pipeline = (
                search.best_estimator_
            )

            y_probability = (
                best_pipeline.predict_proba(
                    X_outer_valid
                )[:, 1]
            )

            metrics = (
                calculate_metrics(
                    y_outer_valid,
                    y_probability,
                )
            )

            selection = (
                extract_selected_features(
                    best_pipeline
                )
            )

            record = {
                "fold": (
                    fold_number
                ),
                "best_params": (
                    search.best_params_
                ),
                "inner_best_roc_auc": (
                    float(
                        search.best_score_
                    )
                ),
                "metrics": metrics,
                "selection": {
                    "selected_original": (
                        selection[
                            "selected_original"
                        ]
                    ),
                    "selected_transformed": (
                        selection[
                            "selected_transformed"
                        ]
                    ),
                    "n_selected_original": (
                        selection[
                            "n_selected_original"
                        ]
                    ),
                    "n_selected_transformed": (
                        selection[
                            "n_selected_transformed"
                        ]
                    ),
                },
            }

            fold_records.append(
                record
            )

            coefficient_path = (
                output_dir
                / (
                    f"{model_name}_"
                    f"fold_{fold_number}_"
                    "coefficients.csv"
                )
            )

            pd.DataFrame(
                selection[
                    "coefficient_rows"
                ]
            ).to_csv(
                coefficient_path,
                index=False,
            )

            print(
                f"Fold {fold_number}: "
                f"ROC-AUC="
                f"{metrics['roc_auc']:.4f}, "
                f"PR-AUC="
                f"{metrics['pr_auc']:.4f}, "
                f"Brier="
                f"{metrics['brier']:.4f}, "
                f"selected original="
                f"{selection['n_selected_original']}, "
                f"best params="
                f"{search.best_params_}"
            )

        summary = (
            summarize_model_results(
                fold_records
            )
        )

        stability_df = (
            calculate_selection_stability(
                fold_records,
                args.outer_folds,
            )
        )

        all_results[
            model_name
        ] = {
            "folds": (
                fold_records
            ),
            "summary": (
                summary
            ),
        }

        summaries[
            model_name
        ] = summary

        stability_tables[
            model_name
        ] = stability_df

        stability_path = (
            output_dir
            / (
                f"{model_name}_"
                "selection_stability.csv"
            )
        )

        stability_df.to_csv(
            stability_path,
            index=False,
        )

        if model_name in {
            "logistic_l1",
            "logistic_elastic_net",
        }:
            plot_selection_stability(
                stability_df,
                model_name,
                output_dir
                / (
                    f"{model_name}_"
                    "selection_stability.png"
                ),
            )

    results_path = (
        output_dir
        / "nested_cv_results.json"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=2,
        )

    summary_rows = []

    for (
        model_name,
        summary,
    ) in summaries.items():
        summary_rows.append(
            {
                "model": (
                    model_name
                ),
                **summary,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        output_dir
        / "model_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    plot_model_performance(
        summaries,
        output_dir
        / "nested_cv_roc_auc.png",
    )

    print(
        "\n\nNested-CV summary"
    )
    print(
        "================="
    )

    display_columns = [
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "brier_mean",
        "selected_original_features_mean",
    ]

    print(
        summary_df[
            display_columns
        ]
        .to_string(
            index=False,
            float_format=(
                lambda value: (
                    f"{value:.4f}"
                )
            ),
        )
    )

    print(
        "\nStable features "
        "(selected in >=80% of outer folds)"
    )
    print(
        "==========================================="
    )

    for model_name in [
        "logistic_l1",
        "logistic_elastic_net",
    ]:
        stability_df = (
            stability_tables[
                model_name
            ]
        )

        stable = (
            stability_df[
                stability_df[
                    "selection_rate"
                ] >= 0.80
            ]
        )

        print(
            f"\n{model_name}: "
            f"{len(stable)} stable features"
        )

        if stable.empty:
            print(
                "  None"
            )
        else:
            for _, row in (
                stable.iterrows()
            ):
                print(
                    f"  {row['feature']}: "
                    f"{int(row['selected_folds'])}/"
                    f"{args.outer_folds}"
                )

    print(
        f"\nArtifacts written to: "
        f"{output_dir}"
    )

    print(
        "\nInterpretation:"
        "\n- L2 is the regularized dense reference."
        "\n- L1 shows the performance cost/benefit of aggressive sparsity."
        "\n- Elastic Net balances sparsity and correlated-feature stability."
        "\n- The holdout test set has not been used."
    )


if __name__ == "__main__":
    main()
