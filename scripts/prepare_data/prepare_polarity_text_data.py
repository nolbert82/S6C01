from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dataset with only review_text and polarity_label."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prepared/training_dataset.csv"),
        help="Path to the prepared training dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared/polarity_text_dataset.csv"),
        help="Path where the reduced dataset will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. Run scripts/prepare_training_data.py first."
        )

    df = pd.read_csv(args.input)
    required_columns = ["review_text", "polarity_label"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input file: {missing}")

    reduced = df[required_columns].copy()
    reduced["review_text"] = reduced["review_text"].fillna("").astype(str).str.strip()
    reduced = reduced[reduced["review_text"] != ""]
    reduced["polarity_label"] = pd.to_numeric(
        reduced["polarity_label"], errors="coerce"
    ).astype("Int64")
    reduced = reduced.dropna(subset=["polarity_label"])
    reduced["polarity_label"] = reduced["polarity_label"].astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Rows: {len(reduced)} | Columns: {len(reduced.columns)}")


if __name__ == "__main__":
    main()
