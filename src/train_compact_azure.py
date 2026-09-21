"""
Azure ML training/evaluation component for the frozen compact 10-feature HGB.

This is deliberately separate from model-selection research:
- feature selection has already been completed by nested CV,
- the selected top-10 feature set is read from the upstream research output,
- the model is fit once on all development/training data,
- the untouched holdout is used once for final evaluation,
- parameters, metrics and artifacts are logged to MLflow.

The output model is a custom sklearn/joblib artifact that can be registered
in the Azure ML model registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--feature-selection-output",
        type=str,
        required=True,
        help=(
            "Directory produced by budget_feature_selection_experiment.py."
        ),
    )
    parser.add_argument(
        "--model-output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--evaluation-output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def resolve_csv(path_value: str, filename: str) -> Path:
    """
    Accept either a direct CSV path or a mounted directory containing it.
    """

    path = Path(path_value)

    if path.is_file():
        return path

    candidate = path / filename

    if candidate.exists():
        return candidate

    matches = list(
        path.rglob(filename)
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve '{filename}' from '{path_value}'."
    )


def load_selected_features(
    feature_selection_output: str,
) -> list[str]:
    output_dir = Path(
        feature_selection_output
    )

    candidate = (
        output_dir
        / "candidate_final_feature_sets.json"
    )

    if not candidate.exists():
        matches = list(
            output_dir.rglob(
                "candidate_final_feature_sets.json"
            )
        )

        if len(matches) != 1:
            raise FileNotFoundError(
                "Could not uniquely resolve "
                "candidate_final_feature_sets.json "
                f"under {output_dir}."
            )

        candidate = matches[0]

    payload = json.loads(
        candidate.read_text(
            encoding="utf-8"
        )
    )

    try:
        selected = payload[
            "expanded"
        ][
            "top_10"
        ]
    except KeyError as exc:
        raise ValueError(
            "Expected "
            "payload['expanded']['top_10'] "
            "in candidate feature-set JSON."
        ) from exc

    if len(selected) != 10:
        raise ValueError(
            f"Expected exactly 10 selected features, "
            f"found {len(selected)}."
        )

    return list(selected)


def build_pipeline(
    selected_features: list[str],
    random_state: int,
) -> Pipeline:
    selected_set = set(
        selected_features
    )

    numeric_features = [
        feature
        for feature in (
            EXPANDED_NUMERIC_FEATURES
        )
        if feature in selected_set
    ]

    categorical_features = [
        feature
        for feature in (
            CATEGORICAL_FEATURES
        )
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

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = (
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            random_state=random_state,
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
                classifier,
            ),
        ]
    )


def calculate_metrics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

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


def main() -> None:
    args = parse_args()

    train_path = resolve_csv(
        args.train_data,
        "train.csv",
    )
    test_path = resolve_csv(
        args.test_data,
        "test.csv",
    )

    selected_features = (
        load_selected_features(
            args.feature_selection_output
        )
    )

    train_df = pd.read_csv(
        train_path
    )
    test_df = pd.read_csv(
        test_path
    )

    X_train_raw = train_df.drop(
        columns=[TARGET_COLUMN]
    )
    y_train = train_df[
        TARGET_COLUMN
    ].astype(int)

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

    print(
        "Frozen compact feature set:"
    )
    for feature in selected_features:
        print(
            f"  - {feature}"
        )

    model = build_pipeline(
        selected_features,
        args.random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    metrics = calculate_metrics(
        model,
        X_test,
        y_test,
        args.threshold,
    )

    model_output = Path(
        args.model_output
    )
    evaluation_output = Path(
        args.evaluation_output
    )

    model_output.mkdir(
        parents=True,
        exist_ok=True,
    )
    evaluation_output.mkdir(
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

    model_metadata = {
        "model_role": (
            "interpretable_compact_challenger"
        ),
        "algorithm": (
            "HistGradientBoostingClassifier"
        ),
        "feature_count": (
            len(selected_features)
        ),
        "selected_features": (
            selected_features
        ),
        "feature_space": (
            "23 original + 13 deterministic engineered candidates"
        ),
        "selection_method": (
            "Nested-CV Elastic-Net ranking with exact feature budget=10"
        ),
        "threshold": (
            args.threshold
        ),
        "random_state": (
            args.random_state
        ),
    }

    (
        model_output
        / "model_metadata.json"
    ).write_text(
        json.dumps(
            model_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        evaluation_output
        / "holdout_metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        evaluation_output
        / "selected_features.json"
    ).write_text(
        json.dumps(
            {
                "selected_features": (
                    selected_features
                )
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Azure ML automatically connects MLflow to the workspace for the job.
    mlflow.set_tag(
        "model_role",
        "interpretable_compact_challenger",
    )
    mlflow.set_tag(
        "model_family",
        "hist_gradient_boosting",
    )
    mlflow.set_tag(
        "feature_selection",
        "nested_cv_elastic_net_budget_10",
    )

    mlflow.log_param(
        "feature_count",
        len(selected_features),
    )
    mlflow.log_param(
        "selected_features",
        ",".join(
            selected_features
        ),
    )
    mlflow.log_param(
        "threshold",
        args.threshold,
    )
    mlflow.log_param(
        "random_state",
        args.random_state,
    )

    for metric_name in [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "brier",
        "log_loss",
    ]:
        mlflow.log_metric(
            f"holdout_{metric_name}",
            metrics[
                metric_name
            ],
        )

    mlflow.log_metric(
        "holdout_tn",
        metrics["tn"],
    )
    mlflow.log_metric(
        "holdout_fp",
        metrics["fp"],
    )
    mlflow.log_metric(
        "holdout_fn",
        metrics["fn"],
    )
    mlflow.log_metric(
        "holdout_tp",
        metrics["tp"],
    )

    mlflow.log_artifact(
        str(
            model_output
            / "model_metadata.json"
        ),
        artifact_path="model_metadata",
    )
    mlflow.log_artifact(
        str(
            evaluation_output
            / "holdout_metrics.json"
        ),
        artifact_path="evaluation",
    )
    mlflow.log_artifact(
        str(
            evaluation_output
            / "selected_features.json"
        ),
        artifact_path="evaluation",
    )

    print(
        "\nCompact holdout metrics"
    )
    print(
        "======================="
    )
    for name, value in (
        metrics.items()
    ):
        print(
            f"{name}: {value}"
        )

    print(
        f"\nModel written to: "
        f"{model_path}"
    )
    print(
        f"Evaluation written to: "
        f"{evaluation_output}"
    )


if __name__ == "__main__":
    main()
