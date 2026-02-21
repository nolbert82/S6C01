from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.svm import LinearSVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SVM model to predict polarity_label using a shared TFIDF."
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
        default=Path("text_representations/TFIDF/saved_model/tfidf_vectorizer.pkl"),
        help="Path to the pre-fitted shared TFIDF.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/TFIDF_svm/polarity_prediction/saved_model"),
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Number of rows between progress updates while building text documents.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=2000,
        help="Maximum number of iterations for LinearSVC.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Stopping tolerance for LinearSVC. Higher values usually train faster.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=100000,
        help="Train only on the first N rows of the dataset. Use 0 or negative to use all rows.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level for LinearSVC training progress.",
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
            f"Input file not found: {args.input}. Run scripts/prepare_data/prepare_training_data.py first."
        )

    if not args.vectorizer_path.exists():
        raise FileNotFoundError(
            f"Shared vectorizer not found: {args.vectorizer_path}. Run text_representations/TFIDF/train.py first."
        )

    df = pd.read_csv(args.input)
    if args.max_rows > 0:
        original_rows = len(df)
        df = df.head(args.max_rows).copy()
        print(
            f"Using first {len(df)} rows out of {original_rows} total rows "
            f"(max_rows={args.max_rows})."
        )

    required_columns = {"review_text", "user_name", "polarity_label"}
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    text_block = df.loc[:, "review_text":"user_name"].copy()
    text_block = text_block.fillna("").astype(str)
    documents = build_documents_with_progress(text_block, args.progress_every)

    labels = df["polarity_label"].astype(int)
    train_documents = documents
    train_labels = labels

    print(f"Loading shared TFIDF from: {args.vectorizer_path}")
    with args.vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)

    print(f"Vectorizing training data (rows: {len(train_documents)})...")
    x_train = vectorizer.transform(train_documents)
    print("Vectorization complete.")

    print(f"Training LinearSVC (verbose={args.verbose})...")
    model = LinearSVC(
        C=1.0,
        dual=False,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=args.verbose,
        random_state=67,
    )
    model.fit(x_train, train_labels)
    print("Training complete.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "svm_polarity_model.pkl"

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to: {model_path}")
    print(f"Using shared vectorizer: {args.vectorizer_path}")
    print(f"Training rows: {len(train_documents)}")


if __name__ == "__main__":
    main()

