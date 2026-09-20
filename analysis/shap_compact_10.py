"""
SHAP explainability for the frozen compact 10-feature HGB challenger.

Outputs
-------
1. Global SHAP importance (mean absolute SHAP value)
2. SHAP beeswarm summary across a holdout sample
3. Local SHAP waterfall for a representative high-risk holdout customer
4. CSV/JSON artifacts for presentation and auditability

Method
------
- The compact feature set and fitted HGB are frozen before this step.
- Deterministic engineered features are recreated from src/features.py.
- A training-data background sample defines the SHAP reference distribution.
- Explanations are calculated on holdout observations.
- SHAP is applied to the classifier's predicted default probability.
- One-hot-expanded SHAP values are aggregated back to the 10 source features.

The local example is selected WITHOUT using the true target:
the holdout customer whose predicted probability is closest to the
90th percentile of all holdout predictions.

Important
---------
SHAP explains model behaviour, not causality.
Correlated variables can share or redistribute attribution.
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


def parse_args() -> argparse.Namespace:
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
        "--feature-set-json",
        type=str,
        default=(
            "outputs/budget_feature_selection/"
            "candidate_final_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/model_compact_10/model.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/explainability_compact_10/shap"
        ),
    )
    parser.add_argument(
        "--background-rows",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--global-rows",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--local-percentile",
        type=float,
        default=0.90,
        help=(
            "Predicted-risk percentile used to select the "
            "representative local explanation."
        ),
    )
    parser.add_argument(
        "--permutation-rounds",
        type=int,
        default=5,
        help=(
            "SHAP permutation cycles. Higher values are slower "
            "but reduce Monte Carlo noise."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_selected_features(
    feature_set_json: str,
) -> list[str]:
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
            f"Expected exactly 10 compact features, "
            f"found {len(features)}."
        )

    return list(features)


def transformed_to_source_feature(
    transformed_name: str,
) -> str:
    """Map a transformed ColumnTransformer output back to its source feature."""

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
        f"'{transformed_name}' to a source feature."
    )


def aggregate_shap_to_source(
    shap_values: np.ndarray,
    transformed_names: np.ndarray,
    source_features: list[str],
) -> np.ndarray:
    """
    Aggregate transformed-column SHAP values back to source features.

    For example, all one-hot columns belonging to education are summed.
    """

    source_matrix = np.zeros(
        (
            shap_values.shape[0],
            len(source_features),
        ),
        dtype=float,
    )

    source_index = {
        feature: index
        for index, feature
        in enumerate(
            source_features
        )
    }

    for column_index, transformed_name in enumerate(
        transformed_names
    ):
        source = transformed_to_source_feature(
            transformed_name
        )

        if source not in source_index:
            raise ValueError(
                f"Transformed feature '{transformed_name}' "
                f"maps to unexpected source '{source}'."
            )

        source_matrix[
            :,
            source_index[source],
        ] += shap_values[
            :,
            column_index,
        ]

    return source_matrix


def make_global_bar_plot(
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = importance_df.sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["mean_abs_shap"],
    )

    ax.set_xlabel(
        "Mean absolute SHAP contribution to predicted default probability"
    )
    ax.set_ylabel(
        "Compact-model feature"
    )
    ax.set_title(
        "Global SHAP Importance – Compact 10-Feature HGB"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not 0 < args.local_percentile < 1:
        raise ValueError(
            "--local-percentile must be between 0 and 1."
        )

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "The 'shap' package is required. "
            "Install it with: pip install shap"
        ) from exc

    selected_features = (
        load_selected_features(
            args.feature_set_json
        )
    )

    print(
        "Frozen compact feature set:"
    )
    for feature in selected_features:
        feature_type = (
            "engineered"
            if feature in ENGINEERED_FEATURES
            else "original"
        )
        print(
            f"  - {feature} [{feature_type}]"
        )

    train_df = pd.read_csv(
        args.train_data
    )
    test_df = pd.read_csv(
        args.test_data
    )

    for name, frame in [
        ("train", train_df),
        ("test", test_df),
    ]:
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(
                f"{name} data does not contain "
                f"'{TARGET_COLUMN}'."
            )

    X_train_raw = train_df.drop(
        columns=[TARGET_COLUMN]
    )
    X_test_raw = test_df.drop(
        columns=[TARGET_COLUMN]
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

    model = joblib.load(
        args.model_path
    )

    transformer = model.named_steps[
        "feature_transformer"
    ]
    classifier = model.named_steps[
        "classifier"
    ]

    transformed_names = np.asarray(
        transformer.get_feature_names_out()
    )

    # Training background defines the SHAP reference distribution.
    background_n = min(
        args.background_rows,
        len(X_train),
    )

    background_source = X_train.sample(
        n=background_n,
        random_state=args.random_state,
    )

    background_transformed = (
        transformer.transform(
            background_source
        )
    )

    # Global SHAP explanations are calculated only on a holdout sample.
    global_n = min(
        args.global_rows,
        len(X_test),
    )

    global_source = X_test.sample(
        n=global_n,
        random_state=args.random_state,
    )

    global_transformed = (
        transformer.transform(
            global_source
        )
    )

    # Select local case based only on MODEL PREDICTION, not actual target.
    holdout_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    target_probability = float(
        np.quantile(
            holdout_probability,
            args.local_percentile,
        )
    )

    local_position = int(
        np.argmin(
            np.abs(
                holdout_probability
                - target_probability
            )
        )
    )

    local_source = X_test.iloc[
        [
            local_position
        ]
    ].copy()

    local_transformed = (
        transformer.transform(
            local_source
        )
    )

    local_predicted_probability = float(
        holdout_probability[
            local_position
        ]
    )

    local_actual_default = int(
        y_test.iloc[
            local_position
        ]
    )

    print(
        f"\nGlobal SHAP sample: "
        f"{global_n} holdout rows"
    )
    print(
        f"Background sample: "
        f"{background_n} training rows"
    )
    print(
        f"Local explanation: holdout row "
        f"{local_position}, predicted PD="
        f"{local_predicted_probability:.4f}, "
        f"actual default="
        f"{local_actual_default}"
    )

    def predict_probability(
        transformed_array: np.ndarray,
    ) -> np.ndarray:
        return classifier.predict_proba(
            transformed_array
        )[:, 1]

    n_transformed_features = (
        background_transformed.shape[
            1
        ]
    )

    max_evals = (
        args.permutation_rounds
        * (
            2
            * n_transformed_features
            + 1
        )
    )

    print(
        f"\nCreating probability-scale SHAP "
        f"PermutationExplainer "
        f"(max_evals={max_evals})..."
    )

    explainer = shap.Explainer(
        predict_probability,
        background_transformed,
        algorithm="permutation",
        feature_names=(
            transformed_names.tolist()
        ),
    )

    global_explanation = explainer(
        global_transformed,
        max_evals=max_evals,
    )

    local_explanation = explainer(
        local_transformed,
        max_evals=max_evals,
    )

    global_values = np.asarray(
        global_explanation.values
    )

    local_values = np.asarray(
        local_explanation.values
    )

    if global_values.ndim != 2:
        raise ValueError(
            "Expected two-dimensional global SHAP values, "
            f"received shape {global_values.shape}."
        )

    if local_values.ndim != 2:
        raise ValueError(
            "Expected two-dimensional local SHAP values, "
            f"received shape {local_values.shape}."
        )

    global_source_shap = (
        aggregate_shap_to_source(
            global_values,
            transformed_names,
            selected_features,
        )
    )

    local_source_shap = (
        aggregate_shap_to_source(
            local_values,
            transformed_names,
            selected_features,
        )
    )

    mean_abs_shap = np.mean(
        np.abs(
            global_source_shap
        ),
        axis=0,
    )

    importance_df = pd.DataFrame(
        {
            "feature": selected_features,
            "mean_abs_shap": (
                mean_abs_shap
            ),
            "feature_type": [
                (
                    "engineered"
                    if feature
                    in ENGINEERED_FEATURES
                    else "original"
                )
                for feature in (
                    selected_features
                )
            ],
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        output_dir
        / "shap_global_importance.csv",
        index=False,
    )

    make_global_bar_plot(
        importance_df,
        output_dir
        / "shap_global_importance.png",
    )

    # Create source-level SHAP Explanation for a proper beeswarm.
    source_global_explanation = (
        shap.Explanation(
            values=global_source_shap,
            base_values=np.asarray(
                global_explanation.base_values
            ),
            data=global_source[
                selected_features
            ].to_numpy(),
            feature_names=(
                selected_features
            ),
        )
    )

    shap.plots.beeswarm(
        source_global_explanation,
        max_display=10,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir
        / "shap_beeswarm.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    local_base_value = float(
        np.asarray(
            local_explanation.base_values
        ).reshape(-1)[0]
    )

    local_values_source = (
        local_source_shap[
            0
        ]
    )

    additive_probability = float(
        local_base_value
        + local_values_source.sum()
    )

    local_df = pd.DataFrame(
        {
            "feature": selected_features,
            "feature_value": [
                local_source.iloc[
                    0
                ][feature]
                for feature in (
                    selected_features
                )
            ],
            "shap_contribution": (
                local_values_source
            ),
            "abs_shap_contribution": (
                np.abs(
                    local_values_source
                )
            ),
            "feature_type": [
                (
                    "engineered"
                    if feature
                    in ENGINEERED_FEATURES
                    else "original"
                )
                for feature in (
                    selected_features
                )
            ],
        }
    ).sort_values(
        "abs_shap_contribution",
        ascending=False,
    ).reset_index(
        drop=True
    )

    local_df.to_csv(
        output_dir
        / "shap_local_high_risk_customer.csv",
        index=False,
    )

    local_source_explanation = (
        shap.Explanation(
            values=(
                local_values_source
            ),
            base_values=(
                local_base_value
            ),
            data=local_source[
                selected_features
            ].iloc[
                0
            ].to_numpy(),
            feature_names=(
                selected_features
            ),
        )
    )

    shap.plots.waterfall(
        local_source_explanation,
        max_display=10,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir
        / "shap_local_waterfall.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    metadata = {
        "model_role": (
            "interpretable_compact_challenger"
        ),
        "model_path": (
            args.model_path
        ),
        "feature_set_source": (
            args.feature_set_json
        ),
        "selected_features": (
            selected_features
        ),
        "background_rows": (
            background_n
        ),
        "global_explanation_rows": (
            global_n
        ),
        "shap_algorithm": (
            "permutation"
        ),
        "model_output": (
            "predicted default probability"
        ),
        "permutation_rounds": (
            args.permutation_rounds
        ),
        "max_evals_per_explanation": (
            max_evals
        ),
        "local_case_selection": {
            "rule": (
                "closest holdout prediction to "
                f"{args.local_percentile:.0%} "
                "predicted-risk percentile"
            ),
            "holdout_row_position": (
                local_position
            ),
            "predicted_default_probability": (
                local_predicted_probability
            ),
            "actual_default": (
                local_actual_default
            ),
            "base_probability": (
                local_base_value
            ),
            "base_plus_shap_sum": (
                additive_probability
            ),
            "additivity_difference": float(
                additive_probability
                - local_predicted_probability
            ),
        },
        "notes": [
            (
                "The true target was not used to select "
                "the local example."
            ),
            (
                "One-hot SHAP contributions are aggregated "
                "back to source features."
            ),
            (
                "SHAP describes model behaviour, not causality."
            ),
            (
                "Correlated features can redistribute "
                "SHAP attribution."
            ),
        ],
    }

    with (
        output_dir
        / "shap_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=str,
        )

    print(
        "\nGlobal SHAP importance"
    )
    print(
        "======================"
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

    print(
        "\nLocal high-risk explanation"
    )
    print(
        "==========================="
    )
    print(
        f"Base probability:      "
        f"{local_base_value:.4f}"
    )
    print(
        f"Predicted probability: "
        f"{local_predicted_probability:.4f}"
    )
    print(
        f"Base + SHAP sum:        "
        f"{additive_probability:.4f}"
    )
    print(
        f"Actual default label:   "
        f"{local_actual_default}"
    )
    print()

    print(
        local_df[
            [
                "feature",
                "feature_value",
                "shap_contribution",
            ]
        ].to_string(
            index=False,
            float_format=(
                lambda value: (
                    f"{value:.5f}"
                )
            ),
        )
    )

    print(
        f"\nArtifacts written to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
