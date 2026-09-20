"""
Grouped permutation importance for the frozen compact 10-feature HGB.

DKV Mobility Azure ML interview case.

Purpose
-------
Measure the joint predictive contribution of four business-oriented
feature blocks instead of interpreting correlated features only one-by-one.

The 10 compact-model features are grouped as:

1. Delinquency behaviour
   - max_delinquency
   - delinquent_months
   - severe_delinquent_months
   - pay_0
   - pay_3

2. Payment behaviour
   - mean_payment_amount
   - payment_amount_std

3. Credit capacity / balance
   - limit_bal
   - bill_amt2

4. Socio-demographic
   - education

Method
------
- Load the already fitted compact model.
- Recreate deterministic engineered features from src/features.py.
- Compute baseline holdout ROC-AUC.
- For each group, apply the SAME random row permutation to every feature
  in that group. This preserves the within-group joint structure while
  breaking its association with the target and the other feature blocks.
- Repeat the permutation many times.
- Report the mean and standard deviation of the ROC-AUC decrease.

Important
---------
This is post-hoc model explainability.
It does not retrain or tune the model.
Grouped permutation importance measures model reliance, not causality.
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

from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import add_domain_features  # noqa: E402


TARGET_COLUMN = "default"

FEATURE_GROUPS = {
    "Delinquency behaviour": [
        "max_delinquency",
        "delinquent_months",
        "severe_delinquent_months",
        "pay_0",
        "pay_3",
    ],
    "Payment behaviour": [
        "mean_payment_amount",
        "payment_amount_std",
    ],
    "Credit capacity / balance": [
        "limit_bal",
        "bill_amt2",
    ],
    "Socio-demographic": [
        "education",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        default=(
            "outputs/model_compact_10/"
            "model.joblib"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/explainability_compact_10/"
            "grouped_permutation"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
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

    return list(features)


def validate_groups(
    selected_features: list[str],
) -> None:
    grouped_features = [
        feature
        for group_features in (
            FEATURE_GROUPS.values()
        )
        for feature in group_features
    ]

    duplicates = sorted(
        {
            feature
            for feature in grouped_features
            if grouped_features.count(
                feature
            ) > 1
        }
    )

    if duplicates:
        raise ValueError(
            "Features occur in multiple groups: "
            f"{duplicates}"
        )

    missing = sorted(
        set(selected_features)
        - set(grouped_features)
    )

    unexpected = sorted(
        set(grouped_features)
        - set(selected_features)
    )

    if missing or unexpected:
        raise ValueError(
            "Group definition does not exactly "
            "cover the frozen compact feature set. "
            f"Missing from groups: {missing}; "
            f"unexpected in groups: {unexpected}"
        )


def calculate_grouped_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    repeats: int,
    random_state: int,
) -> tuple[float, pd.DataFrame, dict]:
    """
    Permute each group jointly using one shared row permutation per repeat.

    Using the same row permutation for all columns in a block preserves
    their internal row-wise relationships.
    """

    baseline_probability = (
        model.predict_proba(
            X
        )[:, 1]
    )

    baseline_roc_auc = float(
        roc_auc_score(
            y,
            baseline_probability,
        )
    )

    rng = np.random.default_rng(
        random_state
    )

    result_rows = []
    raw_results = {}

    n_rows = len(X)

    for group_name, group_features in (
        FEATURE_GROUPS.items()
    ):
        decreases = []

        for repeat in range(
            repeats
        ):
            permutation = (
                rng.permutation(
                    n_rows
                )
            )

            X_permuted = X.copy()

            # Crucial: use the SAME row permutation for every feature
            # in the group to preserve within-group correlation patterns.
            X_permuted.loc[
                :,
                group_features,
            ] = (
                X[
                    group_features
                ]
                .iloc[
                    permutation
                ]
                .to_numpy()
            )

            probability = (
                model.predict_proba(
                    X_permuted
                )[:, 1]
            )

            permuted_roc_auc = float(
                roc_auc_score(
                    y,
                    probability,
                )
            )

            decrease = (
                baseline_roc_auc
                - permuted_roc_auc
            )

            decreases.append(
                decrease
            )

        decreases_array = np.array(
            decreases,
            dtype=float,
        )

        mean_decrease = float(
            decreases_array.mean()
        )
        std_decrease = float(
            decreases_array.std()
        )

        result_rows.append(
            {
                "group": group_name,
                "feature_count": (
                    len(
                        group_features
                    )
                ),
                "features": ", ".join(
                    group_features
                ),
                "importance_mean": (
                    mean_decrease
                ),
                "importance_std": (
                    std_decrease
                ),
                "importance_min": float(
                    decreases_array.min()
                ),
                "importance_max": float(
                    decreases_array.max()
                ),
            }
        )

        raw_results[
            group_name
        ] = {
            "features": (
                group_features
            ),
            "roc_auc_decreases": (
                decreases
            ),
        }

    results_df = pd.DataFrame(
        result_rows
    ).sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return (
        baseline_roc_auc,
        results_df,
        raw_results,
    )


def make_plot(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = (
        results_df.sort_values(
            "importance_mean",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_df["group"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
    )

    ax.set_xlabel(
        "Decrease in holdout ROC-AUC after joint group permutation"
    )
    ax.set_ylabel(
        "Business feature block"
    )
    ax.set_title(
        "Grouped Permutation Importance – Compact 10-Feature HGB"
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

    selected_features = (
        load_selected_features(
            args.feature_set_json
        )
    )

    validate_groups(
        selected_features
    )

    test_df = pd.read_csv(
        args.test_data
    )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Test data does not contain "
            f"'{TARGET_COLUMN}'."
        )

    X_raw = test_df.drop(
        columns=[
            TARGET_COLUMN
        ]
    )
    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    X_expanded = add_domain_features(
        X_raw
    )

    X_test = X_expanded[
        selected_features
    ].copy()

    model = joblib.load(
        args.model_path
    )

    print(
        "Frozen compact model feature groups:"
    )

    for group_name, features in (
        FEATURE_GROUPS.items()
    ):
        print(
            f"\n{group_name}:"
        )
        for feature in features:
            print(
                f"  - {feature}"
            )

    baseline_roc_auc, (
        results_df
    ), raw_results = (
        calculate_grouped_permutation_importance(
            model=model,
            X=X_test,
            y=y_test,
            repeats=args.repeats,
            random_state=args.random_state,
        )
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_dir
        / "grouped_permutation_importance.csv",
        index=False,
    )

    make_plot(
        results_df,
        output_dir
        / "grouped_permutation_importance.png",
    )

    payload = {
        "baseline_holdout_roc_auc": (
            baseline_roc_auc
        ),
        "repeats": (
            args.repeats
        ),
        "random_state": (
            args.random_state
        ),
        "groups": (
            raw_results
        ),
        "method": (
            "Joint row permutation within each feature group "
            "using the same row permutation for every feature "
            "in the group."
        ),
        "notes": [
            (
                "The same row permutation is applied to all "
                "features in a group to preserve within-group "
                "relationships."
            ),
            (
                "Importance is the decrease in holdout ROC-AUC."
            ),
            (
                "Grouped permutation importance measures model "
                "reliance and is not causal."
            ),
            (
                "Groups with different feature counts are not "
                "directly normalized for group size."
            ),
        ],
    }

    with (
        output_dir
        / "grouped_permutation_details.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print(
        f"\nBaseline holdout ROC-AUC: "
        f"{baseline_roc_auc:.4f}"
    )

    print(
        "\nGrouped permutation importance"
    )
    print(
        "=============================="
    )

    print(
        results_df[
            [
                "group",
                "feature_count",
                "importance_mean",
                "importance_std",
            ]
        ]
        .to_string(
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

    print(
        "\nInterpretation:"
        "\n- A larger ROC-AUC drop means the model relies more "
        "on that business feature block."
        "\n- The delinquency block should be interpreted jointly "
        "because its features are intentionally related."
        "\n- This is post-hoc explanation, not causal inference."
    )


if __name__ == "__main__":
    main()
