from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save a TF-IDF vectorizer on the first 90% of the training dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prepared/training_dataset.csv"),
        help="Path to the prepared training dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("text_representations/TFIDF/saved_model"),
        help="Directory where the trained TF-IDF vectorizer will be saved.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Number of rows between progress updates while building text documents.",
    )
    return parser.parse_args()


def build_documents_with_progress(text_block: pd.DataFrame, progress_every: int) -> pd.Series:
    total_rows = len(text_block)
    documents: list[str] = []

    print(f"Building text documents: 0/{total_rows} (0.00%)")
    for idx, row in enumerate(text_block.itertuples(index=False, name=None), start=1):
        parts = []
        for value in row:
            text = str(value).strip()
            if text:
                parts.append(text)
        documents.append(" ".join(parts))

        if idx % progress_every == 0 or idx == total_rows:
            pct = (idx / total_rows) * 100
            print(f"Building text documents: {idx}/{total_rows} ({pct:.2f}%)")

    return pd.Series(documents)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. Run scripts/prepare_training_data.py first."
        )

    df = pd.read_csv(args.input)

    if "review_text" not in df.columns or "user_name" not in df.columns:
        raise KeyError(
            "Input dataset must contain both 'review_text' and 'user_name' columns."
        )

    text_block = df.loc[:, "review_text":"user_name"].copy()
    text_block = text_block.fillna("").astype(str)
    documents = build_documents_with_progress(text_block, args.progress_every)

    split_index = max(1, math.floor(len(documents) * 0.9))
    train_documents = documents.iloc[:split_index]

    print("Fitting TF-IDF vectorizer on first 90% of rows...")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_documents)
    print("TF-IDF fit complete.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "tfidf_vectorizer.pkl"
    with model_path.open("wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved vectorizer to: {model_path}")
    print(f"Total rows: {len(documents)} | Training rows (90%): {len(train_documents)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
