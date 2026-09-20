"""
Generate a Sweetviz EDA report for the EXPANDED 36-feature training dataset.

Important:
- Uses ONLY the 80% training partition.
- Adds the same 13 deterministic domain features used in the modeling experiments.
- Does NOT load or compare against the holdout test set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import sweetviz as sv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import add_domain_features  # noqa: E402


TARGET_COLUMN = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train/train.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eda/sweetviz_train_expanded_36.html",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_data)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Training data does not contain '{TARGET_COLUMN}'."
        )

    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN].astype(int)

    X_expanded = add_domain_features(X)

    expanded_df = X_expanded.copy()
    expanded_df[TARGET_COLUMN] = y

    print(f"Training observations: {len(expanded_df):,}")
    print(f"Predictor features: {X_expanded.shape[1]}")
    print(f"Default rate: {y.mean():.2%}")

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = sv.analyze(
        source=expanded_df,
        target_feat=TARGET_COLUMN,
        pairwise_analysis="auto",
    )

    report.show_html(
        filepath=str(output_path),
        open_browser=False,
        layout="widescreen",
    )

    print(f"Sweetviz report written to: {output_path}")


if __name__ == "__main__":
    main()
