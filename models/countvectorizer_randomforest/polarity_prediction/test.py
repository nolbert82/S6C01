from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test shared CountVectorizer + RandomForest polarity model on last 10% of dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prepared/training_dataset.csv"),
        help="Path to the prepared training dataset.",
    )
    parser.add_argument(
        "--vectorizer_path",
        type=Path,
        default=Path("text_representations/CountVectorizer/saved_model/count_vectorizer.pkl"),
        help="Path to the pre-fitted shared CountVectorizer.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("models/countvectorizer_randomforest/polarity_prediction/saved_model"),
        help="Directory containing the trained RandomForest model.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("models/countvectorizer_randomforest/polarity_prediction/results.txt"),
        help="Path where test results will be written.",
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
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if not args.vectorizer_path.exists():
        raise FileNotFoundError(
            f"Shared vectorizer not found: {args.vectorizer_path}. Run text_representations/CountVectorizer/train.py first."
        )

    model_path = args.model_dir / "rf_polarity_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    df = pd.read_csv(args.input)

    required_columns = {"review_text", "user_name", "polarity_label"}
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    text_block = df.loc[:, "review_text":"user_name"].copy()
    text_block = text_block.fillna("").astype(str)
    documents = build_documents_with_progress(text_block, args.progress_every)
    labels = df["polarity_label"].astype(int)

    split_index = max(1, math.floor(len(documents) * 0.9))
    test_documents = documents.iloc[split_index:]
    test_labels = labels.iloc[split_index:]

    if test_documents.empty:
        raise ValueError("No test rows found in the last 10% split.")

    print(f"Loading shared vectorizer from: {args.vectorizer_path}")
    with args.vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)

    print(f"Loading model from: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    print(f"Vectorizing test data (rows: {len(test_documents)})...")
    x_test = vectorizer.transform(test_documents)

    print("Running predictions...")
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(test_labels, y_pred)
    report = classification_report(test_labels, y_pred, digits=4)

    result_text = (
        "CountVectorizer + RandomForest (polarity_label prediction)\n"
        f"Shared vectorizer: {args.vectorizer_path}\n"
        f"Total rows: {len(documents)}\n"
        f"Train rows (first 90%): {split_index}\n"
        f"Test rows (last 10%): {len(test_documents)}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(result_text, encoding="utf-8")

    print(f"Test complete. Results written to: {args.results_path}")


if __name__ == "__main__":
    main()
