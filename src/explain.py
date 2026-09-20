"""
Explainability utilities for the DKV Mobility Azure ML interview case.

This script is intentionally separate from model training and selection.

It produces two complementary views:

1. Selected nonlinear model:
   Model-agnostic permutation importance on the already frozen holdout model.
   This measures how much ROC-AUC deteriorates when one original input feature
   is randomly permuted.

2. Transparent reference model:
   A separate logistic regression is fitted on the training partition only.
   Numeric features are standardized. Categorical features use reference
   coding (drop="first") so that category coefficients have a clear reference
   category.

Important:
- The selected production candidate is NOT refitted here.
- Holdout permutation importance is for post-hoc interpretation only.
- Do not use holdout explainability results to tune or re-select the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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


def load_partition(
    path: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a prepared train/test partition."""

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Data does not contain target column '{TARGET_COLUMN}'."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    expected_features = set(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    if set(X.columns) != expected_features:
        raise ValueError(
            "Feature schema does not match the expected training schema."
        )

    return X, y


def calculate_permutation_importance(
    model_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Calculate model-agnostic feature importance for the selected model.

    Importance is measured as the reduction in holdout ROC-AUC after
    permuting one ORIGINAL input feature at a time. Because the complete
    sklearn Pipeline is evaluated, permutation happens before the fitted
    feature transformer.
    """

    result = permutation_importance(
        model_pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df["abs_importance"] = (
        importance_df["importance_mean"].abs()
    )

    return importance_df.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> None:
    """Plot the most influential original features."""

    plot_df = (
        importance_df
        .head(top_n)
        .sort_values("importance_mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
    )

    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Decrease in ROC-AUC after permutation")
    ax.set_ylabel("Original input feature")
    ax.set_title(
        "Permutation Importance – Selected Holdout Model"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_interpretable_logistic_pipeline(
    random_state: int,
) -> Pipeline:
    """
    Build a transparent logistic reference model.

    This is intentionally separate from the selected nonlinear model.

    Numeric features are standardized, so their coefficients describe the
    change in log-odds associated with a one-standard-deviation increase.

    Categorical features use drop="first", making the omitted first category
    the reference category for coefficient interpretation.
    """

    feature_transformer = ColumnTransformer(
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

    return Pipeline(
        steps=[
            (
                "feature_transformer",
                feature_transformer,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def extract_logistic_coefficients(
    logistic_pipeline: Pipeline,
) -> tuple[pd.DataFrame, dict]:
    """
    Extract coefficients, odds ratios, and categorical reference levels.
    """

    feature_transformer = logistic_pipeline.named_steps[
        "feature_transformer"
    ]
    classifier = logistic_pipeline.named_steps["classifier"]

    raw_feature_names = (
        feature_transformer.get_feature_names_out()
    )

    feature_names = [
        name.replace("numeric__", "")
        .replace("categorical__", "")
        for name in raw_feature_names
    ]

    coefficients = classifier.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )

    coefficient_df["abs_coefficient"] = (
        coefficient_df["coefficient"].abs()
    )

    coefficient_df["direction"] = np.where(
        coefficient_df["coefficient"] >= 0,
        "higher predicted default odds",
        "lower predicted default odds",
    )

    coefficient_df = coefficient_df.sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    categorical_encoder = (
        feature_transformer
        .named_transformers_["categorical"]
    )

    reference_categories = {}
    for feature, category_values, drop_index in zip(
        CATEGORICAL_FEATURES,
        categorical_encoder.categories_,
        categorical_encoder.drop_idx_,
    ):
        value = category_values[drop_index]
        reference_categories[feature] = (
            value.item()
            if hasattr(value, "item")
            else value
        )

    return coefficient_df, reference_categories


def plot_logistic_coefficients(
    coefficient_df: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> None:
    """Plot the largest logistic coefficients by absolute magnitude."""

    plot_df = (
        coefficient_df
        .head(top_n)
        .sort_values("coefficient", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        plot_df["feature"],
        plot_df["coefficient"],
    )

    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Logistic regression coefficient")
    ax.set_ylabel("Transformed feature")
    ax.set_title(
        "Transparent Logistic Reference – Largest Coefficients"
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
        default="data/processed/train/train.csv",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test/test.csv",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/model/model.joblib",
    )

    parser.add_argument(
        "--explain-output",
        type=str,
        default="outputs/explainability",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1.")

    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")

    output_dir = Path(args.explain_output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading prepared train and holdout partitions...")

    X_train, y_train = load_partition(
        args.train_data,
    )

    X_test, y_test = load_partition(
        args.test_data,
    )

    print(
        f"Training observations: {len(X_train):,}"
    )
    print(
        f"Holdout observations:  {len(X_test):,}"
    )

    print("\nLoading selected fitted model pipeline...")

    selected_model_pipeline = joblib.load(
        args.model_path,
    )

    print(
        "Calculating holdout permutation importance "
        f"({args.n_repeats} repeats)..."
    )

    importance_df = calculate_permutation_importance(
        selected_model_pipeline,
        X_test,
        y_test,
        args.n_repeats,
        args.random_state,
    )

    importance_csv = (
        output_dir / "permutation_importance.csv"
    )
    importance_plot = (
        output_dir / "permutation_importance.png"
    )

    importance_df.to_csv(
        importance_csv,
        index=False,
    )

    plot_permutation_importance(
        importance_df,
        importance_plot,
        args.top_n,
    )

    print("\nTop permutation-importance features:")
    print(
        importance_df[
            [
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nFitting separate interpretable "
        "logistic reference model on TRAINING data only..."
    )

    logistic_pipeline = (
        build_interpretable_logistic_pipeline(
            args.random_state
        )
    )

    logistic_pipeline.fit(
        X_train,
        y_train,
    )

    (
        coefficient_df,
        reference_categories,
    ) = extract_logistic_coefficients(
        logistic_pipeline
    )

    coefficient_csv = (
        output_dir / "logistic_coefficients.csv"
    )
    coefficient_plot = (
        output_dir / "logistic_coefficients.png"
    )
    reference_path = (
        output_dir / "logistic_reference_categories.json"
    )

    coefficient_df.to_csv(
        coefficient_csv,
        index=False,
    )

    plot_logistic_coefficients(
        coefficient_df,
        coefficient_plot,
        args.top_n,
    )

    with reference_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            reference_categories,
            file,
            indent=2,
        )

    print("\nLargest logistic coefficients:")
    print(
        coefficient_df[
            [
                "feature",
                "coefficient",
                "odds_ratio",
                "direction",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nCategorical reference levels:")
    for feature, reference in (
        reference_categories.items()
    ):
        print(
            f"  {feature}: {reference}"
        )

    print(
        f"\nExplainability artifacts written to: "
        f"{output_dir}"
    )

    print(
        "\nInterpretation note:"
        "\n- Permutation importance explains the selected nonlinear model."
        "\n- Logistic coefficients explain the separate transparent reference model."
        "\n- Neither should be interpreted as causal effects."
    )


if __name__ == "__main__":
    main()
