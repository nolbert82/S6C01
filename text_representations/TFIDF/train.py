from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save a TF-IDF vectorizer on the training dataset."
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

    df = pd.read_csv(args.input)

    text_block = df.loc[:, "review_text":"user_name"].copy()
    text_block = text_block.fillna("").astype(str)
    documents = build_documents_with_progress(text_block, args.progress_every)

    print("Fitting TF-IDF vectorizer on all training rows...")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)
    print("TF-IDF fit complete.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "tfidf_vectorizer.pkl"
    with model_path.open("wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved vectorizer to: {model_path}")
    print(f"Training rows: {len(documents)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
