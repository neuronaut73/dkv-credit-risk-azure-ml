"""
Exact-budget feature-selection experiment for the DKV Mobility Azure ML case.

Purpose
-------
Answer one final, tightly scoped question:

    Can HistGradientBoosting retain nearly all predictive performance
    with an explicit feature budget of 6 or 10 inputs?

We test BOTH candidate spaces:
1. RAW:      23 original UCI features
2. EXPANDED: 23 original + 13 deterministic engineered features

Method
------
Leakage-safe nested selection:

Outer 5-fold CV
    Outer training fold
        -> inner CV tunes Elastic-Net logistic selector
        -> rank source features by absolute standardized coefficient
        -> select exactly top 6 or top 10
        -> fit HGB using only those selected features
    Outer validation fold
        -> evaluate untouched outer-fold observations

The same outer folds are used for:
- full 23-feature HGB
- full 36-feature HGB
- raw top-6 HGB
- raw top-10 HGB
- expanded top-6 HGB
- expanded top-10 HGB

The 20% holdout test set is NEVER loaded.

Why Elastic Net for ranking?
----------------------------
Numeric inputs are standardized before fitting. Elastic Net combines L1
sparsity with L2 stabilization, which is useful for correlated monthly
credit-history variables. Ranking is performed only inside each outer
training fold, so outer-fold performance remains leakage-safe.

After outer-CV evaluation, the selector is fitted once on ALL training data
to provide candidate final top-6/top-10 feature lists. Those full-training
lists are NOT used to calculate the outer-CV performance estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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

EXPANDED_NUMERIC_FEATURES = (
    ORIGINAL_NUMERIC_FEATURES
    + ENGINEERED_FEATURES
)

EXPANDED_FEATURES = (
    EXPANDED_NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

BUDGETS = [6, 10]


def split_feature_types(
    features: list[str],
) -> tuple[list[str], list[str]]:
    """Split requested source features into numeric/categorical groups."""

    feature_set = set(features)

    numeric_pool = set(
        EXPANDED_NUMERIC_FEATURES
    )

    numeric = [
        feature
        for feature in features
        if feature in numeric_pool
    ]

    categorical = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in feature_set
    ]

    return numeric, categorical


def build_transformer(
    features: list[str],
    *,
    drop_first: bool,
) -> ColumnTransformer:
    """Build preprocessing for exactly the requested source features."""

    numeric, categorical = (
        split_feature_types(
            features
        )
    )

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=(
                        "first"
                        if drop_first
                        else None
                    ),
                    sparse_output=False,
                ),
                categorical,
            )
        )

    if not transformers:
        raise ValueError(
            "Feature list is empty."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_selector_pipeline(
    features: list[str],
    random_state: int,
) -> Pipeline:
    """Build Elastic-Net logistic model used only for feature ranking."""

    return Pipeline(
        steps=[
            (
                "feature_transformer",
                build_transformer(
                    features,
                    drop_first=True,
                ),
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


def build_hgb_pipeline(
    features: list[str],
    random_state: int,
) -> Pipeline:
    """
    Build HGB for the requested source features.

    Scaling is retained to stay directly comparable with the existing
    project HGB pipeline, although tree boosting does not require scaling.
    """

    return Pipeline(
        steps=[
            (
                "feature_transformer",
                build_transformer(
                    features,
                    drop_first=False,
                ),
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


def transformed_to_source_feature(
    transformed_name: str,
) -> str:
    """Map transformed sklearn column name back to source feature."""

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


def source_feature_scores(
    fitted_selector: Pipeline,
    source_features: list[str],
) -> pd.DataFrame:
    """
    Rank source features by maximum absolute standardized coefficient.

    Categorical source features can expand to multiple dummy coefficients;
    their source-level score is the maximum absolute dummy coefficient.
    """

    transformer = (
        fitted_selector.named_steps[
            "feature_transformer"
        ]
    )
    classifier = (
        fitted_selector.named_steps[
            "classifier"
        ]
    )

    transformed_names = (
        transformer.get_feature_names_out()
    )
    coefficients = np.abs(
        classifier.coef_[0]
    )

    scores = {
        feature: 0.0
        for feature in source_features
    }

    for transformed_name, coefficient in zip(
        transformed_names,
        coefficients,
    ):
        source = (
            transformed_to_source_feature(
                transformed_name
            )
        )

        scores[source] = max(
            scores[source],
            float(coefficient),
        )

    ranking = pd.DataFrame(
        [
            {
                "feature": feature,
                "score": score,
                "feature_type": (
                    "engineered"
                    if feature
                    in ENGINEERED_FEATURES
                    else "original"
                ),
            }
            for feature, score
            in scores.items()
        ]
    )

    return (
        ranking.sort_values(
            by=[
                "score",
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


def fit_selector_and_rank(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    source_features: list[str],
    inner_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict, Pipeline]:
    """
    Tune Elastic Net inside the outer-training fold and return a ranking.
    """

    selector = (
        build_selector_pipeline(
            source_features,
            random_state,
        )
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=selector,
        param_grid={
            "classifier__C": [
                0.003,
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
            ],
            "classifier__l1_ratio": [
                0.5,
                0.8,
            ],
        },
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    search.fit(
        X_train[
            source_features
        ],
        y_train,
    )

    ranking = source_feature_scores(
        search.best_estimator_,
        source_features,
    )

    return (
        ranking,
        search.best_params_,
        search.best_estimator_,
    )


def calculate_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Calculate outer-validation metrics."""

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


def fit_and_evaluate_hgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    features: list[str],
    random_state: int,
) -> dict:
    """Fit one HGB model and evaluate the outer validation fold."""

    pipeline = build_hgb_pipeline(
        features,
        random_state,
    )

    pipeline.fit(
        X_train[
            features
        ],
        y_train,
    )

    probability = (
        pipeline.predict_proba(
            X_valid[
                features
            ]
        )[:, 1]
    )

    return calculate_metrics(
        y_valid,
        probability,
    )


def summarize_records(
    records: list[dict],
) -> dict:
    """Summarize fold metrics for one model."""

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
                record[
                    "metrics"
                ][metric]
                for record in records
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

    return summary


def selection_stability(
    records: list[dict],
    candidate_features: list[str],
) -> pd.DataFrame:
    """Count feature selections across outer folds."""

    counter = Counter()

    for record in records:
        counter.update(
            record.get(
                "selected_features",
                [],
            )
        )

    n_folds = len(records)

    rows = []

    for feature in candidate_features:
        selected_folds = (
            counter[feature]
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": (
                    "engineered"
                    if feature
                    in ENGINEERED_FEATURES
                    else "original"
                ),
                "selected_folds": (
                    selected_folds
                ),
                "total_folds": n_folds,
                "selection_rate": (
                    selected_folds
                    / n_folds
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
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


def plot_roc_auc_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot outer-CV ROC-AUC for all model variants."""

    plot_df = summary_df.sort_values(
        "roc_auc_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        plot_df["model"],
        plot_df["roc_auc_mean"],
        xerr=plot_df["roc_auc_std"],
    )

    ax.set_xlabel(
        "Outer-CV ROC-AUC"
    )
    ax.set_ylabel(
        "Model"
    )
    ax.set_title(
        "Exact Feature-Budget Comparison"
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
            "outputs/budget_feature_selection"
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
        default=3,
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
        columns=[
            TARGET_COLUMN
        ]
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

    X_expanded = add_domain_features(
        X_raw
    )

    if set(X_expanded.columns) != set(
        EXPANDED_FEATURES
    ):
        raise ValueError(
            "Expanded feature schema "
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
        f"Raw feature space: "
        f"{len(RAW_FEATURES)}"
    )
    print(
        f"Expanded feature space: "
        f"{len(EXPANDED_FEATURES)}"
    )
    print(
        f"Budgets: {BUDGETS}"
    )

    outer_cv = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    outer_splits = list(
        outer_cv.split(
            X_raw,
            y,
        )
    )

    records = defaultdict(
        list
    )

    for fold_number, (
        train_index,
        valid_index,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        print(
            "\n===================================="
        )
        print(
            f"Outer fold {fold_number}/"
            f"{args.outer_folds}"
        )
        print(
            "===================================="
        )

        X_train_expanded = (
            X_expanded.iloc[
                train_index
            ]
        )
        X_valid_expanded = (
            X_expanded.iloc[
                valid_index
            ]
        )
        y_train = y.iloc[
            train_index
        ]
        y_valid = y.iloc[
            valid_index
        ]

        # Baseline: full raw HGB.
        raw_full_metrics = (
            fit_and_evaluate_hgb(
                X_train_expanded,
                y_train,
                X_valid_expanded,
                y_valid,
                RAW_FEATURES,
                args.random_state
                + fold_number,
            )
        )

        records[
            "full_raw_23"
        ].append(
            {
                "fold": fold_number,
                "metrics": (
                    raw_full_metrics
                ),
            }
        )

        # Reference only: all 36 features.
        expanded_full_metrics = (
            fit_and_evaluate_hgb(
                X_train_expanded,
                y_train,
                X_valid_expanded,
                y_valid,
                EXPANDED_FEATURES,
                args.random_state
                + fold_number,
            )
        )

        records[
            "full_expanded_36"
        ].append(
            {
                "fold": fold_number,
                "metrics": (
                    expanded_full_metrics
                ),
            }
        )

        print(
            "Full raw 23: "
            f"ROC-AUC="
            f"{raw_full_metrics['roc_auc']:.4f}"
        )
        print(
            "Full expanded 36: "
            f"ROC-AUC="
            f"{expanded_full_metrics['roc_auc']:.4f}"
        )

        for (
            space_name,
            candidate_features,
        ) in [
            (
                "raw",
                RAW_FEATURES,
            ),
            (
                "expanded",
                EXPANDED_FEATURES,
            ),
        ]:
            ranking, best_params, _ = (
                fit_selector_and_rank(
                    X_train_expanded,
                    y_train,
                    candidate_features,
                    args.inner_folds,
                    args.random_state
                    + 100
                    + fold_number,
                )
            )

            print(
                f"\n{space_name.upper()} selector "
                f"best params: "
                f"{best_params}"
            )

            for budget in BUDGETS:
                selected_features = (
                    ranking.head(
                        budget
                    )[
                        "feature"
                    ]
                    .tolist()
                )

                model_name = (
                    f"{space_name}_top_{budget}"
                )

                metrics = (
                    fit_and_evaluate_hgb(
                        X_train_expanded,
                        y_train,
                        X_valid_expanded,
                        y_valid,
                        selected_features,
                        args.random_state
                        + fold_number,
                    )
                )

                records[
                    model_name
                ].append(
                    {
                        "fold": fold_number,
                        "metrics": metrics,
                        "selected_features": (
                            selected_features
                        ),
                        "selector_best_params": (
                            best_params
                        ),
                    }
                )

                print(
                    f"{model_name}: "
                    f"ROC-AUC="
                    f"{metrics['roc_auc']:.4f}, "
                    f"PR-AUC="
                    f"{metrics['pr_auc']:.4f}, "
                    f"Brier="
                    f"{metrics['brier']:.4f}"
                )
                print(
                    "  features: "
                    + ", ".join(
                        selected_features
                    )
                )

    summaries = {}

    for model_name, model_records in (
        records.items()
    ):
        summaries[
            model_name
        ] = summarize_records(
            model_records
        )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for model_name, summary in (
        summaries.items()
    ):
        feature_count = (
            23
            if model_name
            == "full_raw_23"
            else 36
            if model_name
            == "full_expanded_36"
            else int(
                model_name.split(
                    "_"
                )[-1]
            )
        )

        summary_rows.append(
            {
                "model": model_name,
                "feature_count": (
                    feature_count
                ),
                **summary,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        "roc_auc_mean",
        ascending=False,
    )

    summary_df.to_csv(
        output_dir
        / "budget_model_summary.csv",
        index=False,
    )

    # Save fold-level records.
    with (
        output_dir
        / "budget_nested_cv_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dict(records),
            file,
            indent=2,
        )

    # Save stability for every compact model.
    stability_outputs = {}

    for space_name, candidate_features in [
        (
            "raw",
            RAW_FEATURES,
        ),
        (
            "expanded",
            EXPANDED_FEATURES,
        ),
    ]:
        for budget in BUDGETS:
            model_name = (
                f"{space_name}_top_{budget}"
            )

            stability_df = (
                selection_stability(
                    records[
                        model_name
                    ],
                    candidate_features,
                )
            )

            stability_outputs[
                model_name
            ] = stability_df

            stability_df.to_csv(
                output_dir
                / (
                    f"{model_name}_"
                    "selection_stability.csv"
                ),
                index=False,
            )

    # Fit selectors once on all TRAINING data to create candidate final lists.
    final_feature_sets = {}

    for (
        space_name,
        candidate_features,
    ) in [
        (
            "raw",
            RAW_FEATURES,
        ),
        (
            "expanded",
            EXPANDED_FEATURES,
        ),
    ]:
        ranking, best_params, (
            fitted_selector
        ) = (
            fit_selector_and_rank(
                X_expanded,
                y,
                candidate_features,
                args.inner_folds,
                args.random_state
                + 999,
            )
        )

        ranking.to_csv(
            output_dir
            / (
                f"{space_name}_"
                "full_train_feature_ranking.csv"
            ),
            index=False,
        )

        joblib.dump(
            fitted_selector,
            output_dir
            / (
                f"{space_name}_"
                "full_train_selector.joblib"
            ),
        )

        final_feature_sets[
            space_name
        ] = {
            "selector_best_params": (
                best_params
            ),
        }

        for budget in BUDGETS:
            final_feature_sets[
                space_name
            ][
                f"top_{budget}"
            ] = (
                ranking.head(
                    budget
                )[
                    "feature"
                ]
                .tolist()
            )

    with (
        output_dir
        / "candidate_final_feature_sets.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_feature_sets,
            file,
            indent=2,
        )

    plot_roc_auc_summary(
        summary_df,
        output_dir
        / "budget_roc_auc_comparison.png",
    )

    print(
        "\n\nExact-budget nested-CV summary"
    )
    print(
        "=============================="
    )

    print(
        summary_df[
            [
                "model",
                "feature_count",
                "roc_auc_mean",
                "roc_auc_std",
                "pr_auc_mean",
                "brier_mean",
                "recall_mean",
                "f1_mean",
            ]
        ]
        .to_string(
            index=False,
            float_format=(
                lambda x: (
                    f"{x:.4f}"
                )
            ),
        )
    )

    print(
        "\nCandidate final feature sets "
        "(fit on all TRAIN data; not used for CV estimates)"
    )
    print(
        "============================================================"
    )

    for space_name in [
        "raw",
        "expanded",
    ]:
        print(
            f"\n{space_name.upper()}"
        )

        for budget in BUDGETS:
            features = (
                final_feature_sets[
                    space_name
                ][
                    f"top_{budget}"
                ]
            )

            print(
                f"  top {budget}: "
                + ", ".join(
                    features
                )
            )

    # Lightweight MLflow tracking. In Azure ML these metrics appear in the
    # workspace run; locally they are logged to the local MLflow store.
    mlflow.log_param(
        "experiment_type",
        "exact_budget_feature_selection",
    )
    mlflow.log_param(
        "outer_folds",
        args.outer_folds,
    )
    mlflow.log_param(
        "inner_folds",
        args.inner_folds,
    )
    mlflow.log_param(
        "budgets",
        ",".join(
            str(value)
            for value in BUDGETS
        ),
    )

    for model_name, summary in (
        summaries.items()
    ):
        mlflow.log_metric(
            f"{model_name}_roc_auc_mean",
            summary[
                "roc_auc_mean"
            ],
        )
        mlflow.log_metric(
            f"{model_name}_pr_auc_mean",
            summary[
                "pr_auc_mean"
            ],
        )
        mlflow.log_metric(
            f"{model_name}_brier_mean",
            summary[
                "brier_mean"
            ],
        )

    mlflow.log_artifacts(
        str(output_dir),
        artifact_path=(
            "budget_feature_selection"
        ),
    )

    print(
        f"\nArtifacts written to: "
        f"{output_dir}"
    )

    print(
        "\nInterpretation:"
        "\n- RAW top-6/top-10 tests whether a small subset of the original "
        "variables is sufficient."
        "\n- EXPANDED top-6/top-10 tests whether engineered summaries can "
        "replace several monthly raw variables."
        "\n- Selection happens inside each outer-training fold."
        "\n- The holdout test set has not been touched."
    )


if __name__ == "__main__":
    main()
