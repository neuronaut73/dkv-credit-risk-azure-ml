"""
Explainability for the frozen 10-feature compact HGB challenger.

DKV Mobility Azure ML interview case.

This script:
1. Reads the final 10-feature set selected from the expanded 36-feature space.
2. Recreates the deterministic engineered features from src/features.py.
3. Fits the compact HGB once on the full 80% training partition.
4. Saves the fitted compact model.
5. Evaluates it on the untouched 20% holdout.
6. Produces model-level explainability:
   - permutation importance on the holdout (ROC-AUC decrease),
   - one-way partial-dependence plots for the most important numeric features,
   - optional SHAP global importance if shap is installed and --with-shap is used.

Important:
- Feature selection is NOT repeated here.
- The selected 10-feature set is read from the prior nested-CV experiment.
- The holdout is NOT used for feature selection or tuning.
- Explainability outputs describe model behaviour, not causality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

EXPANDED_FEATURES = (
    EXPANDED_NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def load_compact_feature_set(
    feature_set_json: str,
) -> list[str]:
    """Load the frozen expanded top-10 feature set."""

    payload = json.loads(
        Path(feature_set_json).read_text(
            encoding="utf-8"
        )
    )

    try:
        features = payload[
            "expanded"
        ][
            "top_10"
        ]
    except KeyError as exc:
        raise ValueError(
            "Expected JSON structure "
            "payload['expanded']['top_10']."
        ) from exc

    if len(features) != 10:
        raise ValueError(
            f"Expected exactly 10 features, found {len(features)}."
        )

    unknown = sorted(
        set(features)
        - set(EXPANDED_FEATURES)
    )

    if unknown:
        raise ValueError(
            f"Unknown features in compact set: {unknown}"
        )

    return list(features)


def build_compact_pipeline(
    selected_features: list[str],
    random_state: int,
) -> Pipeline:
    """Build the same compact HGB architecture used in CV."""

    selected_set = set(
        selected_features
    )

    numeric_features = [
        feature
        for feature in EXPANDED_NUMERIC_FEATURES
        if feature in selected_set
    ]

    categorical_features = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in selected_set
    ]

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "Compact feature set is empty."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            (
                "feature_transformer",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def calculate_holdout_metrics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Calculate final holdout metrics for the compact challenger."""

    probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                prediction,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                probability,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "brier": float(
            brier_score_loss(
                y_test,
                probability,
            )
        ),
        "log_loss": float(
            log_loss(
                y_test,
                probability,
            )
        ),
        "threshold": float(
            threshold
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_permutation_plot(
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot source-level permutation importance."""

    plot_df = importance_df.sort_values(
        "importance_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
    )

    ax.set_xlabel(
        "Decrease in holdout ROC-AUC after permutation"
    )
    ax.set_ylabel(
        "Compact-model feature"
    )
    ax.set_title(
        "Permutation Importance – Compact 10-Feature HGB"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def calculate_manual_pdp(
    model: Pipeline,
    X: pd.DataFrame,
    feature: str,
    grid_points: int = 20,
) -> pd.DataFrame:
    """
    Model-agnostic one-way partial-dependence approximation.

    For every grid value, set the feature to that value for all observations
    and record the average predicted default probability.
    """

    series = X[
        feature
    ]

    if feature in CATEGORICAL_FEATURES:
        grid = np.array(
            sorted(
                series.dropna()
                .unique()
                .tolist()
            )
        )
    else:
        lower = float(
            series.quantile(0.05)
        )
        upper = float(
            series.quantile(0.95)
        )

        if lower == upper:
            grid = np.array(
                [lower]
            )
        else:
            grid = np.linspace(
                lower,
                upper,
                grid_points,
            )

    rows = []

    for value in grid:
        modified = X.copy()
        modified[
            feature
        ] = value

        mean_probability = float(
            model.predict_proba(
                modified
            )[:, 1].mean()
        )

        rows.append(
            {
                "feature": feature,
                "feature_value": (
                    float(value)
                    if np.issubdtype(
                        np.asarray(value).dtype,
                        np.number,
                    )
                    else str(value)
                ),
                "mean_predicted_default_probability": (
                    mean_probability
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_pdp_plot(
    pdp_df: pd.DataFrame,
    feature: str,
    output_path: Path,
) -> None:
    """Plot one-way partial dependence."""

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        pdp_df[
            "feature_value"
        ],
        pdp_df[
            "mean_predicted_default_probability"
        ],
        marker="o",
    )

    ax.set_xlabel(
        feature
    )
    ax.set_ylabel(
        "Mean predicted default probability"
    )
    ax.set_title(
        f"Partial Dependence – {feature}"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def transformed_to_source_feature(
    transformed_name: str,
) -> str:
    """Map transformed feature names back to source-level features."""

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


def create_optional_shap_outputs(
    model: Pipeline,
    X_test: pd.DataFrame,
    output_dir: Path,
    max_rows: int,
    random_state: int,
) -> None:
    """
    Optional SHAP global importance for the HGB classifier.

    SHAP values are calculated on transformed HGB inputs and then aggregated
    back to the 10 source features.
    """

    try:
        import shap
    except ImportError:
        print(
            "\nSHAP requested, but package 'shap' is not installed."
        )
        print(
            "Install it with: pip install shap"
        )
        return

    sample_size = min(
        max_rows,
        len(X_test),
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=random_state,
    )

    transformer = model.named_steps[
        "feature_transformer"
    ]
    classifier = model.named_steps[
        "classifier"
    ]

    transformed = transformer.transform(
        X_sample
    )

    transformed_names = (
        transformer.get_feature_names_out()
    )

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        transformed
    )

    if isinstance(
        shap_values,
        list,
    ):
        shap_values = shap_values[
            -1
        ]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[
            :,
            :,
            -1,
        ]

    if shap_values.shape[1] != len(
        transformed_names
    ):
        raise ValueError(
            "Unexpected SHAP output shape."
        )

    source_features = []

    for name in transformed_names:
        source_features.append(
            transformed_to_source_feature(
                name
            )
        )

    source_order = list(
        X_test.columns
    )

    source_shap = np.zeros(
        (
            shap_values.shape[0],
            len(source_order),
        )
    )

    for transformed_index, source in enumerate(
        source_features
    ):
        source_index = (
            source_order.index(
                source
            )
        )

        source_shap[
            :,
            source_index
        ] += shap_values[
            :,
            transformed_index
        ]

    mean_abs = np.mean(
        np.abs(
            source_shap
        ),
        axis=0,
    )

    shap_df = pd.DataFrame(
        {
            "feature": source_order,
            "mean_abs_shap": (
                mean_abs
            ),
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    shap_df.to_csv(
        output_dir
        / "shap_global_importance.csv",
        index=False,
    )

    plot_df = shap_df.sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_df["feature"],
        plot_df[
            "mean_abs_shap"
        ],
    )

    ax.set_xlabel(
        "Mean absolute SHAP value"
    )
    ax.set_ylabel(
        "Compact-model feature"
    )
    ax.set_title(
        "SHAP Global Importance – Compact 10-Feature HGB"
    )

    fig.tight_layout()
    fig.savefig(
        output_dir
        / "shap_global_importance.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        "\nSHAP global importance:"
    )
    print(
        shap_df.to_string(
            index=False,
            float_format=(
                lambda value: (
                    f"{value:.5f}"
                )
            ),
        )
    )


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
        "--test-data",
        type=str,
        default=(
            "data/processed/test/test.csv"
        ),
    )

    parser.add_argument(
        "--feature-set-json",
        type=str,
        default=(
            "outputs/budget_feature_selection/"
            "candidate_final_feature_sets.json"
        ),
    )

    parser.add_argument(
        "--model-output",
        type=str,
        default=(
            "outputs/model_compact_10"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/explainability_compact_10"
        ),
    )

    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--pdp-features",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--with-shap",
        action="store_true",
        help=(
            "Also calculate SHAP global importance "
            "if the shap package is installed."
        ),
    )

    parser.add_argument(
        "--shap-max-rows",
        type=int,
        default=1500,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_features = (
        load_compact_feature_set(
            args.feature_set_json
        )
    )

    print(
        "Frozen compact feature set:"
    )
    for feature in selected_features:
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

    train_df = pd.read_csv(
        args.train_data
    )
    test_df = pd.read_csv(
        args.test_data
    )

    for name, frame in [
        (
            "train",
            train_df,
        ),
        (
            "test",
            test_df,
        ),
    ]:
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(
                f"{name} data does not "
                f"contain '{TARGET_COLUMN}'."
            )

    X_train_raw = train_df.drop(
        columns=[
            TARGET_COLUMN
        ]
    )
    y_train = train_df[
        TARGET_COLUMN
    ].astype(int)

    X_test_raw = test_df.drop(
        columns=[
            TARGET_COLUMN
        ]
    )
    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    X_train_expanded = (
        add_domain_features(
            X_train_raw
        )
    )
    X_test_expanded = (
        add_domain_features(
            X_test_raw
        )
    )

    X_train = X_train_expanded[
        selected_features
    ].copy()

    X_test = X_test_expanded[
        selected_features
    ].copy()

    model = build_compact_pipeline(
        selected_features,
        args.random_state,
    )

    print(
        "\nFitting frozen compact HGB "
        "on all training data..."
    )

    model.fit(
        X_train,
        y_train,
    )

    model_output = Path(
        args.model_output
    )
    model_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_output
        / "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = (
        calculate_holdout_metrics(
            model,
            X_test,
            y_test,
        )
    )

    with (
        output_dir
        / "holdout_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        "\nCompact 10-feature holdout metrics"
    )
    print(
        "=================================="
    )

    for key in [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "brier",
        "log_loss",
    ]:
        print(
            f"{key:<10}: "
            f"{metrics[key]:.4f}"
        )

    print(
        f"confusion : "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']}"
    )

    print(
        "\nCalculating permutation importance..."
    )

    permutation = (
        permutation_importance(
            estimator=model,
            X=X_test,
            y=y_test,
            scoring="roc_auc",
            n_repeats=(
                args.permutation_repeats
            ),
            random_state=(
                args.random_state
            ),
            n_jobs=-1,
        )
    )

    importance_df = pd.DataFrame(
        {
            "feature": (
                X_test.columns
            ),
            "importance_mean": (
                permutation.importances_mean
            ),
            "importance_std": (
                permutation.importances_std
            ),
            "feature_type": [
                (
                    "engineered"
                    if feature
                    in ENGINEERED_FEATURES
                    else "original"
                )
                for feature in (
                    X_test.columns
                )
            ],
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(
        drop=True
    )

    importance_df.to_csv(
        output_dir
        / "permutation_importance.csv",
        index=False,
    )

    make_permutation_plot(
        importance_df,
        output_dir
        / "permutation_importance.png",
    )

    print(
        "\nPermutation importance:"
    )
    print(
        importance_df.to_string(
            index=False,
            float_format=(
                lambda value: (
                    f"{value:.5f}"
                )
            ),
        )
    )

    # Prefer numeric features for PDPs; category effects can be added later.
    top_pdp_features = [
        feature
        for feature in (
            importance_df[
                "feature"
            ]
        )
        if feature
        not in CATEGORICAL_FEATURES
    ][
        : args.pdp_features
    ]

    print(
        "\nCreating PDPs for:"
    )

    for feature in (
        top_pdp_features
    ):
        print(
            f"  - {feature}"
        )

        pdp_df = (
            calculate_manual_pdp(
                model,
                X_test,
                feature,
            )
        )

        pdp_df.to_csv(
            output_dir
            / f"pdp_{feature}.csv",
            index=False,
        )

        make_pdp_plot(
            pdp_df,
            feature,
            output_dir
            / f"pdp_{feature}.png",
        )

    if args.with_shap:
        create_optional_shap_outputs(
            model=model,
            X_test=X_test,
            output_dir=output_dir,
            max_rows=(
                args.shap_max_rows
            ),
            random_state=(
                args.random_state
            ),
        )

    metadata = {
        "model_role": (
            "interpretable_compact_challenger"
        ),
        "feature_set_source": (
            args.feature_set_json
        ),
        "selected_features": (
            selected_features
        ),
        "model_path": str(
            model_path
        ),
        "permutation_scoring": (
            "roc_auc"
        ),
        "permutation_repeats": (
            args.permutation_repeats
        ),
        "pdp_features": (
            top_pdp_features
        ),
        "notes": [
            (
                "The compact 10-feature set was selected "
                "before this explainability step."
            ),
            (
                "The holdout is used for final evaluation "
                "and post-hoc explanation, not tuning."
            ),
            (
                "Permutation importance and SHAP describe "
                "model behaviour, not causal effects."
            ),
            (
                "Correlated features can share importance."
            ),
        ],
    }

    with (
        output_dir
        / "explainability_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nCompact model written to: "
        f"{model_path}"
    )
    print(
        f"Explainability artifacts written to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
