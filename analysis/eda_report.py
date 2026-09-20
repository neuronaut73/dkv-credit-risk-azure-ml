from pathlib import Path

import pandas as pd
import sweetviz as sv


TRAIN_PATH = "data/processed/train/train.csv"
OUTPUT_DIR = Path("outputs/eda")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)

    report = sv.analyze(
        [train_df, "Training Data"],
        target_feat="default",
        pairwise_analysis="on",
    )

    report.show_html(
        filepath=str(OUTPUT_DIR / "sweetviz_train_report.html"),
        open_browser=False,
    )

    print(
        "EDA report written to: "
        f"{OUTPUT_DIR / 'sweetviz_train_report.html'}"
    )


if __name__ == "__main__":
    main()