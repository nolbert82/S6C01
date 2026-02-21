from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RandomForest model to predict score_label using a shared TFIDF."
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
        default=Path("models/TFIDF_randomforest/score_prediction/saved_model"),
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Number of rows between progress updates while building text documents.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=30,
        help="Maximum depth of each tree in the random forest.",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=2,
        help="Minimum number of samples required to be at a leaf node.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity level for RandomForest training progress.",
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

    required_columns = {"review_text", "user_name", "score_label"}
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    text_block = df.loc[:, "review_text":"user_name"].copy()
    text_block = text_block.fillna("").astype(str)
    documents = build_documents_with_progress(text_block, args.progress_every)

    labels = df["score_label"].astype(int)
    train_documents = documents
    train_labels = labels

    print(f"Loading shared TFIDF from: {args.vectorizer_path}")
    with args.vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)

    print(f"Vectorizing training data (rows: {len(train_documents)})...")
    x_train = vectorizer.transform(train_documents)
    print("Vectorization complete.")

    print(f"Training RandomForestClassifier (verbose={args.verbose})...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1,
        verbose=args.verbose,
        random_state=67,
    )
    model.fit(x_train, train_labels)
    print("Training complete.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "rf_score_model.pkl"

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to: {model_path}")
    print(f"Using shared vectorizer: {args.vectorizer_path}")
    print(f"Training rows: {len(train_documents)}")


if __name__ == "__main__":
    main()

