"""
Holdout evaluation component for the DKV Mobility Azure ML interview case.

Responsibilities:
- Load the untouched holdout test partition.
- Load the complete fitted scikit-learn pipeline produced by train.py.
- Generate default probabilities and class predictions.
- Evaluate discrimination, classification performance, and calibration.
- Persist metrics and diagnostic plots.
- Log evaluation results to MLflow.

Important:
The holdout test set is used only after model selection and final fitting.
No model or preprocessing parameters are fitted in this component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
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


TARGET_COLUMN = "default"


def load_test_data(
    test_data_path: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the untouched holdout test set."""

    test_df = pd.read_csv(test_data_path)

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Test data does not contain '{TARGET_COLUMN}'."
        )

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    target_values = set(y_test.unique())

    if target_values != {0, 1}:
        raise ValueError(
            f"Expected binary target values {{0, 1}}, "
            f"found {target_values}."
        )

    return X_test, y_test


def calculate_metrics(
    y_true: pd.Series,
    y_probability,
    threshold: float,
) -> tuple[dict, object]:
    """
    Calculate holdout metrics.

    Probability-based metrics evaluate ranking and calibration.
    Classification metrics depend on the selected decision threshold.
    """

    y_pred = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_probability)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_probability)
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
            brier_score_loss(y_true, y_probability)
        ),
        "log_loss": float(
            log_loss(y_true, y_probability)
        ),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics, y_pred


def save_evaluation_plots(
    y_true: pd.Series,
    y_probability,
    y_pred,
    output_dir: Path,
) -> list[Path]:
    """Create and persist diagnostic evaluation plots."""

    plot_paths = []

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_true,
        y_probability,
        ax=ax,
    )
    ax.set_title("ROC Curve – Holdout Test Set")
    fig.tight_layout()

    roc_path = output_dir / "roc_curve.png"
    fig.savefig(
        roc_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    plot_paths.append(roc_path)

    # Precision-recall curve
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_probability,
        ax=ax,
    )
    ax.set_title(
        "Precision-Recall Curve – Holdout Test Set"
    )
    fig.tight_layout()

    pr_path = output_dir / "precision_recall_curve.png"
    fig.savefig(
        pr_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    plot_paths.append(pr_path)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["No default", "Default"],
        ax=ax,
        values_format="d",
    )
    ax.set_title(
        "Confusion Matrix – Holdout Test Set"
    )
    fig.tight_layout()

    confusion_path = output_dir / "confusion_matrix.png"
    fig.savefig(
        confusion_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    plot_paths.append(confusion_path)

    # Calibration curve
    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(
        y_true,
        y_probability,
        n_bins=10,
        strategy="quantile",
        ax=ax,
    )
    ax.set_title(
        "Calibration Curve – Holdout Test Set"
    )
    fig.tight_layout()

    calibration_path = output_dir / "calibration_curve.png"
    fig.savefig(
        calibration_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    plot_paths.append(calibration_path)

    return plot_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local or Azure ML execution."""

    parser = argparse.ArgumentParser()

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
        "--evaluation-output",
        type=str,
        default="outputs/evaluation",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help=(
            "Probability threshold used only for binary "
            "classification metrics."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 < args.threshold < 1:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    output_dir = Path(args.evaluation_output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading untouched holdout test data...")

    X_test, y_test = load_test_data(
        args.test_data,
    )

    print(f"Test observations: {len(X_test):,}")
    print(f"Observed default rate: {y_test.mean():.2%}")

    print("Loading fitted model pipeline...")

    model_pipeline = joblib.load(
        args.model_path,
    )

    # The fitted feature transformer contained inside the pipeline is
    # applied here using transform only. Nothing is refitted on test data.
    y_probability = model_pipeline.predict_proba(
        X_test
    )[:, 1]

    metrics, y_pred = calculate_metrics(
        y_test,
        y_probability,
        args.threshold,
    )

    metrics_path = output_dir / "test_metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    # Persist row-level predictions for auditability and later analysis.
    predictions = pd.DataFrame(
        {
            "actual_default": y_test.to_numpy(),
            "predicted_probability": y_probability,
            "predicted_class": y_pred,
        }
    )

    predictions_path = output_dir / "predictions.csv"
    predictions.to_csv(
        predictions_path,
        index=False,
    )

    plot_paths = save_evaluation_plots(
        y_test,
        y_probability,
        y_pred,
        output_dir,
    )

    # Log scalar metrics.
    for metric_name, value in metrics.items():
        if metric_name not in {
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        }:
            mlflow.log_metric(
                f"test_{metric_name}",
                value,
            )

    # Confusion-matrix counts are also useful as MLflow metrics.
    mlflow.log_metric(
        "test_true_negative",
        metrics["true_negative"],
    )
    mlflow.log_metric(
        "test_false_positive",
        metrics["false_positive"],
    )
    mlflow.log_metric(
        "test_false_negative",
        metrics["false_negative"],
    )
    mlflow.log_metric(
        "test_true_positive",
        metrics["true_positive"],
    )

    mlflow.log_artifact(
        str(metrics_path),
        artifact_path="evaluation",
    )

    mlflow.log_artifact(
        str(predictions_path),
        artifact_path="evaluation",
    )

    for plot_path in plot_paths:
        mlflow.log_artifact(
            str(plot_path),
            artifact_path="evaluation/plots",
        )

    print("\nHoldout test results")
    print("--------------------")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Brier:     {metrics['brier']:.4f}")
    print(f"Log loss:  {metrics['log_loss']:.4f}")

    print(
        f"\nClassification threshold: "
        f"{metrics['threshold']:.2f}"
    )

    print("\nConfusion matrix counts")
    print("-----------------------")
    print(
        f"True negatives:  "
        f"{metrics['true_negative']}"
    )
    print(
        f"False positives: "
        f"{metrics['false_positive']}"
    )
    print(
        f"False negatives: "
        f"{metrics['false_negative']}"
    )
    print(
        f"True positives:  "
        f"{metrics['true_positive']}"
    )

    print(f"\nMetrics written to: {metrics_path}")
    print(
        f"Predictions written to: "
        f"{predictions_path}"
    )
    print(
        f"Evaluation plots written to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()