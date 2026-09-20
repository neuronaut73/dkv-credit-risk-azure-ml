"""
Data preparation / deterministic preprocessing component for the
DKV Mobility Azure ML interview case.

Responsibilities:
- Load the raw UCI Credit Card Default dataset.
- Validate the expected schema and target.
- Apply deterministic, domain-based preprocessing rules.
- Create a reproducible stratified train/test split.
- Persist train and test datasets.

Important:
No data-dependent transformations are fitted here.

Learned feature preprocessing such as scaling, encoding, imputation,
feature selection, or resampling is intentionally NOT performed here.
Those transformations are fitted only on training data inside the
scikit-learn Pipeline in train.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


DATASET_ID = 350
TARGET_COLUMN = "default"

# UCI names X1 ... X23 are converted to business-readable names.
FEATURE_NAME_MAP = {
    "X1": "limit_bal",
    "X2": "sex",
    "X3": "education",
    "X4": "marriage",
    "X5": "age",
    "X6": "pay_0",
    "X7": "pay_2",
    "X8": "pay_3",
    "X9": "pay_4",
    "X10": "pay_5",
    "X11": "pay_6",
    "X12": "bill_amt1",
    "X13": "bill_amt2",
    "X14": "bill_amt3",
    "X15": "bill_amt4",
    "X16": "bill_amt5",
    "X17": "bill_amt6",
    "X18": "pay_amt1",
    "X19": "pay_amt2",
    "X20": "pay_amt3",
    "X21": "pay_amt4",
    "X22": "pay_amt5",
    "X23": "pay_amt6",
}

EXPECTED_FEATURE_COUNT = 23

def load_dataset() -> pd.DataFrame:
    """Load the UCI Credit Card Default dataset."""

    dataset = fetch_ucirepo(id=DATASET_ID)

    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but received {features.shape[1]}."
        )

    if target.shape[1] != 1:
        raise ValueError(
            f"Expected exactly one target column, "
            f"but received {target.shape[1]}."
        )

    # Deterministic renaming only; no parameters are learned from the data.
    features = features.rename(columns=FEATURE_NAME_MAP)

    target = target.iloc[:, 0].rename(TARGET_COLUMN)

    return pd.concat([features, target], axis=1)


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate structural assumptions without learning parameters from the data.
    """

    expected_columns = set(FEATURE_NAME_MAP.values()) | {TARGET_COLUMN}
    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns
    unexpected_columns = actual_columns - expected_columns

    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: "
            f"{sorted(missing_columns)}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Dataset contains unexpected columns: "
            f"{sorted(unexpected_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    target_values = set(df[TARGET_COLUMN].unique())

    if target_values != {0, 1}:
        raise ValueError(
            f"Expected binary target values {{0, 1}}, "
            f"found {target_values}."
        )


def apply_deterministic_preprocessing(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply fixed, domain-based preprocessing rules.

    These transformations do not estimate parameters from the sample.
    Therefore, the same rules can be applied independently to training
    and test data without introducing data leakage.

    UCI documents:
      education: 1=graduate school, 2=university,
                 3=high school, 4=other
      marriage:  1=married, 2=single, 3=other

    Historical versions of this dataset contain additional undocumented
    category codes. They are deterministically mapped to "other".
    """

    prepared = df.copy()

    # Undocumented EDUCATION codes (0, 5, 6) -> "other" (4).
    prepared["education"] = prepared["education"].replace(
        {0: 4, 5: 4, 6: 4}
    )

    # Undocumented MARRIAGE code 0 -> "other" (3).
    prepared["marriage"] = prepared["marriage"].replace({0: 3})

    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype(int)

    return prepared


def save_dataset(
    df: pd.DataFrame,
    output_dir: str,
    filename: str,
) -> Path:
    """Persist a dataframe to an output directory."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename
    df.to_csv(file_path, index=False)

    return file_path

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local or Azure ML execution."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-output",
        type=str,
        default="data/processed/train",
        help="Directory in which train.csv will be written.",
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="data/processed/test",
        help="Directory in which test.csv will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of observations reserved for the holdout test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    print("Loading UCI Credit Card Default dataset...")
    df = load_dataset()

    # Structural validation is performed before the split because it does
    # not estimate any parameters from the data.
    validate_schema(df)

    print(f"Loaded observations: {len(df):,}")
    print(f"Features: {len(df.columns) - 1}")
    print(
        "Overall default rate: "
        f"{df[TARGET_COLUMN].mean():.2%}"
    )

    # Split BEFORE any learned feature preprocessing.
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[TARGET_COLUMN],
    )

    # Apply identical fixed rules independently to both partitions.
    train_df = apply_deterministic_preprocessing(train_df)
    test_df = apply_deterministic_preprocessing(test_df)

    train_path = save_dataset(
        train_df,
        args.train_output,
        "train.csv",
    )

    test_path = save_dataset(
        test_df,
        args.test_output,
        "test.csv",
    )

    print(f"Training observations: {len(train_df):,}")
    print(f"Test observations:     {len(test_df):,}")
    print(f"Training data written to: {train_path}")
    print(f"Test data written to:     {test_path}")


if __name__ == "__main__":
    main()