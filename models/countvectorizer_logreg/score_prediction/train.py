import pickle
from pathlib import Path

import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from scripts.data_pipeline import load_joined_dataframe, split_train_test


def artifacts_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def log(message: str) -> None:
    print(f"[train] {message}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Torch device detected: {device}")

    output_dir = artifacts_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Artifacts directory: {output_dir}")

    log("Loading and joining CSV files (reviews + business + users)...")
    df = load_joined_dataframe()
    log(f"Joined dataset size: {len(df)} rows")

    log("Creating train/test split (90/10, stratified)...")
    train_df, test_df = split_train_test(df)
    log(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    log("Fitting CountVectorizer...")
    vectorizer = CountVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2,
    )
    x_train = vectorizer.fit_transform(train_df["combined_text"])
    y_train = train_df["label"]
    log(f"CountVectorizer vocabulary size: {len(vectorizer.vocabulary_)}")

    log("Training Logistic Regression on CountVectorizer features...")
    model = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    log("Model training complete")

    log("Evaluating on held-out test split...")
    x_test = vectorizer.transform(test_df["combined_text"])
    y_test = test_df["label"]
    y_pred = model.predict(x_test)

    log(f"Validation accuracy (10% split): {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4), flush=True)

    log("Saving vectorizers and model with pickle...")
    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    log("Saving held-out test split for later test.py runs...")
    test_df[["review_id", "label", "combined_text"]].to_csv(
        output_dir / "test_split.csv",
        index=False,
    )

    log("Done")
    log(f"Saved artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
