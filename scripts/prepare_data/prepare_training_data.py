from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = [
    "review_text",
    "business_name",
    "business_categories",
    "business_city",
    "business_state",
    "user_name",
]

NUMERIC_COLUMNS = [
    "review_useful",
    "review_funny",
    "review_cool",
    "business_stars",
    "business_review_count",
    "business_is_open",
    "business_latitude",
    "business_longitude",
    "user_review_count",
    "user_useful",
    "user_funny",
    "user_cool",
    "user_fans",
    "user_average_stars",
    "user_compliment_hot",
    "user_compliment_more",
    "user_compliment_profile",
    "user_compliment_cute",
    "user_compliment_list",
    "user_compliment_note",
    "user_compliment_plain",
    "user_compliment_cool",
    "user_compliment_funny",
    "user_compliment_writer",
    "user_compliment_photos",
]

SCORE_LABEL_COLUMN = "score_label"
POLARITY_LABEL_COLUMN = "polarity_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare training/testing CSVs by left-joining Yelp reviews with business and user data."
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/csv/yelp_academic_reviews4students.csv"),
        help="Path to reviews CSV.",
    )
    parser.add_argument(
        "--business",
        type=Path,
        default=Path("data/csv/yelp_academic_dataset_business.csv"),
        help="Path to business CSV.",
    )
    parser.add_argument(
        "--users",
        type=Path,
        default=Path("data/csv/yelp_academic_dataset_user4students.csv"),
        help="Path to users CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared/training_dataset.csv"),
        help="Training output CSV path. Testing output is written beside it as testing_dataset.csv.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main() -> None:
    args = parse_args()

    reviews = pd.read_csv(args.reviews)
    business = pd.read_csv(args.business)
    users = pd.read_csv(args.users)

    reviews = reviews.rename(
        columns={
            "text": "review_text",
            "stars": SCORE_LABEL_COLUMN,
            "useful": "review_useful",
            "funny": "review_funny",
            "cool": "review_cool",
        }
    )

    business = business.rename(
        columns={
            "name": "business_name",
            "categories": "business_categories",
            "city": "business_city",
            "state": "business_state",
            "stars": "business_stars",
            "review_count": "business_review_count",
            "is_open": "business_is_open",
            "latitude": "business_latitude",
            "longitude": "business_longitude",
        }
    )

    users = users.rename(
        columns={
            "name": "user_name",
            "review_count": "user_review_count",
            "useful": "user_useful",
            "funny": "user_funny",
            "cool": "user_cool",
            "fans": "user_fans",
            "average_stars": "user_average_stars",
            "compliment_hot": "user_compliment_hot",
            "compliment_more": "user_compliment_more",
            "compliment_profile": "user_compliment_profile",
            "compliment_cute": "user_compliment_cute",
            "compliment_list": "user_compliment_list",
            "compliment_note": "user_compliment_note",
            "compliment_plain": "user_compliment_plain",
            "compliment_cool": "user_compliment_cool",
            "compliment_funny": "user_compliment_funny",
            "compliment_writer": "user_compliment_writer",
            "compliment_photos": "user_compliment_photos",
        }
    )

    business_keep = ["business_id"] + [
        c for c in business.columns if c.startswith("business_") and c != "business_id"
    ]
    user_keep = ["user_id"] + [
        c for c in users.columns if c.startswith("user_") and c != "user_id"
    ]

    merged = reviews.merge(business[business_keep], on="business_id", how="left")
    merged = merged.merge(users[user_keep], on="user_id", how="left")

    merged = ensure_columns(merged, TEXT_COLUMNS, "")
    merged = ensure_columns(merged, NUMERIC_COLUMNS, 0)
    if SCORE_LABEL_COLUMN not in merged.columns:
        raise KeyError("Could not find review stars column to create the label.")

    for col in TEXT_COLUMNS:
        merged[col] = clean_text(merged[col])
    for col in NUMERIC_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    merged[SCORE_LABEL_COLUMN] = pd.to_numeric(merged[SCORE_LABEL_COLUMN], errors="coerce")
    merged = merged.dropna(subset=[SCORE_LABEL_COLUMN])
    merged[SCORE_LABEL_COLUMN] = merged[SCORE_LABEL_COLUMN].astype(int)

    merged[POLARITY_LABEL_COLUMN] = 0
    merged.loc[merged[SCORE_LABEL_COLUMN].isin([1, 2]), POLARITY_LABEL_COLUMN] = -1
    merged.loc[merged[SCORE_LABEL_COLUMN] == 3, POLARITY_LABEL_COLUMN] = 0
    merged.loc[merged[SCORE_LABEL_COLUMN].isin([4, 5]), POLARITY_LABEL_COLUMN] = 1
    merged[POLARITY_LABEL_COLUMN] = merged[POLARITY_LABEL_COLUMN].astype(int)
    merged = merged.dropna(how="all")

    ordered_columns = TEXT_COLUMNS + NUMERIC_COLUMNS + [SCORE_LABEL_COLUMN, POLARITY_LABEL_COLUMN]
    prepared = merged[ordered_columns]

    split_idx = int(len(prepared) * 0.9)
    training = prepared.iloc[:split_idx]
    testing = prepared.iloc[split_idx:]

    training_path = args.output
    testing_path = args.output.with_name("testing_dataset.csv")
    training_path.parent.mkdir(parents=True, exist_ok=True)

    training.to_csv(training_path, index=False, quoting=csv.QUOTE_MINIMAL)
    testing.to_csv(testing_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Training dataset saved to: {training_path}")
    print(f"Testing dataset saved to: {testing_path}")
    print(
        f"Training rows: {len(training)} | Testing rows: {len(testing)} | Columns: {len(prepared.columns)}"
    )


if __name__ == "__main__":
    main()
