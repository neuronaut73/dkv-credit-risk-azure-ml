"""
Model training component for the DKV Mobility Azure ML interview case.

Responsibilities:
- Load the training partition produced by preprocess.py.
- Build learned feature transformations.
- Combine feature transformation and estimation in scikit-learn Pipelines.
- Compare an interpretable baseline with a nonlinear challenger using
  leakage-safe stratified cross-validation.
- Fit the selected complete pipeline on the full training partition.
- Persist the fitted feature transformation + model as one deployable artifact.

Important:
All data-dependent feature transformations are fitted inside the
scikit-learn Pipeline. During cross-validation they are therefore refitted
independently within every training fold.

The holdout test set is never accessed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier


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


def build_feature_transformer() -> ColumnTransformer:
    """
    Build the learned feature transformation graph.

    Scaling and encoding parameters are estimated only when fit() is called
    on the surrounding scikit-learn Pipeline. During cross-validation,
    this therefore happens independently inside every training fold.
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
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_candidates(
    random_state: int,
) -> dict[str, Pipeline]:
    """
    Define an interpretable baseline and a nonlinear challenger.

    Each candidate contains both learned feature transformation and model,
    ensuring that cross-validation evaluates the complete ML pipeline.
    """

    logistic_pipeline = Pipeline(
        steps=[
            (
                "feature_transformer",
                build_feature_transformer(),
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

    boosting_pipeline = Pipeline(
        steps=[
            (
                "feature_transformer",
                build_feature_transformer(),
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
    
    dummy_pipeline = Pipeline(
    steps=[
        (
            "feature_transformer",
            build_feature_transformer(),
        ),
        (
            "classifier",
            DummyClassifier(
                strategy="prior",
            ),
        ),
    ]
)

    return {
        "dummy_prior": dummy_pipeline,
        "logistic_regression": logistic_pipeline,
        "hist_gradient_boosting": boosting_pipeline,
    }


def evaluate_candidates(
    candidates: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict:
    """
    Evaluate candidate pipelines using identical cross-validation folds.

    The complete feature-transformer + classifier pipeline is fitted
    separately in every fold, preventing preprocessing leakage.
    """

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "neg_brier": make_scorer(
            brier_score_loss,
            response_method="predict_proba",
            greater_is_better=False,
        ),
    }

    results = {}

    for model_name, model_pipeline in candidates.items():

        print(f"\nEvaluating: {model_name}")

        cv_result = cross_validate(
            model_pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        model_result = {}

        for metric_name in scoring:
            values = cv_result[f"test_{metric_name}"]

            if metric_name == "neg_brier":
                values = -values
                metric_label = "brier"
            else:
                metric_label = metric_name

            model_result[f"{metric_label}_mean"] = float(
                np.mean(values)
            )
            model_result[f"{metric_label}_std"] = float(
                np.std(values)
            )

        results[model_name] = model_result

        print(
            f"ROC-AUC:  "
            f"{model_result['roc_auc_mean']:.4f} "
            f"(+/- {model_result['roc_auc_std']:.4f})"
        )
        print(
            f"PR-AUC:   "
            f"{model_result['pr_auc_mean']:.4f} "
            f"(+/- {model_result['pr_auc_std']:.4f})"
        )
        print(
            f"Recall:   "
            f"{model_result['recall_mean']:.4f}"
        )
        print(
            f"Precision:"
            f"{model_result['precision_mean']:.4f}"
        )
        print(
            f"F1:       "
            f"{model_result['f1_mean']:.4f}"
        )
        print(
            f"Brier:    "
            f"{model_result['brier_mean']:.4f}"
        )

    return results


def select_model(results: dict) -> str:
    """
    Select the candidate with the highest mean ROC-AUC.

    ROC-AUC is used as the primary discrimination criterion.
    PR-AUC, recall, precision, F1, and Brier score remain visible for
    interpretation of class imbalance, classification trade-offs,
    and probability calibration.
    """

    return max(
        results,
        key=lambda model_name: results[model_name]["roc_auc_mean"],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local or Azure ML execution."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train/train.csv",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="outputs/model",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )

    return parser.parse_args()



def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_data)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Training data does not contain '{TARGET_COLUMN}'."
        )

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)

    print(f"Training observations: {len(train_df):,}")
    print(f"Default rate: {y_train.mean():.2%}")

    expected_features = set(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    if set(X_train.columns) != expected_features:
        raise ValueError(
            "Training feature schema does not match expected schema."
        )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    candidates = build_candidates(args.random_state)

    results = evaluate_candidates(
        candidates,
        X_train,
        y_train,
        cv,
    )

    selected_model_name = select_model(results)
    selected_pipeline = candidates[selected_model_name]

    print(f"\nSelected model: {selected_model_name}")

    # After model selection, fit the COMPLETE feature-transformer + model
    # pipeline on all available training data.
    selected_pipeline.fit(
        X_train,
        y_train,
    )

    output_dir = Path(args.model_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "cv_metrics.json"

    # Persist the complete fitted pipeline rather than the classifier alone.
    # This ensures identical feature transformation during inference.
    joblib.dump(
        selected_pipeline,
        model_path,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "selected_model": selected_model_name,
                "cv_results": results,
            },
            file,
            indent=2,
        )

    # In Azure ML, MLflow logging is connected to the workspace.
    # During local development, these calls can be logged locally.
    mlflow.log_param(
        "selected_model",
        selected_model_name,
    )
    mlflow.log_param(
        "cv_folds",
        args.cv_folds,
    )
    
    for model_name, model_metrics in results.items():
        for metric_name, value in model_metrics.items():
            mlflow.log_metric(
                f"cv_{model_name}_{metric_name}",
                value,
            )

    print(f"Model written to: {model_path}")
    print(f"CV metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()