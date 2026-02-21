from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn


class BowEmbeddingMLPClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            mode="sum",
            sparse=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        offsets: torch.Tensor,
        token_weights: torch.Tensor,
    ) -> torch.Tensor:
        features = self.embedding(token_ids, offsets, per_sample_weights=token_weights)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test shared TFIDF + PyTorch MLP polarity model on the testing dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prepared/testing_dataset.csv"),
        help="Path to the prepared testing dataset.",
    )
    parser.add_argument(
        "--vectorizer_path",
        type=Path,
        default=Path("text_representations/TFIDF/saved_model/tfidf_vectorizer.pkl"),
        help="Path to the pre-fitted shared TFIDF.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("models/TFIDF_mlp/polarity_prediction/saved_model"),
        help="Directory containing the trained MLP model.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("models/TFIDF_mlp/polarity_prediction/results.txt"),
        help="Path where test results will be written.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Number of rows between progress updates while building text documents.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for inference.",
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


def csr_batch_to_embedding_inputs(
    x_csr_batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indptr = x_csr_batch.indptr
    indices = x_csr_batch.indices
    data = x_csr_batch.data

    token_ids = torch.tensor(indices, dtype=torch.long, device=device)
    offsets = torch.tensor(indptr[:-1], dtype=torch.long, device=device)
    token_weights = torch.tensor(data, dtype=torch.float32, device=device)
    return token_ids, offsets, token_weights


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if not args.vectorizer_path.exists():
        raise FileNotFoundError(
            f"Shared vectorizer not found: {args.vectorizer_path}. Run text_representations/TFIDF/train.py first."
        )

    model_path = args.model_dir / "mlp_polarity_model.pt"
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
    y_true = df["polarity_label"].astype(int).to_numpy()

    print(f"Loading shared vectorizer from: {args.vectorizer_path}")
    with args.vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)

    print(f"Vectorizing test data (rows: {len(documents)})...")
    x_test_sparse = vectorizer.transform(documents)

    print(f"Loading model from: {model_path}")
    bundle = torch.load(model_path, map_location="cpu")
    label_values = bundle["label_values"]

    model = BowEmbeddingMLPClassifier(
        vocab_size=bundle["vocab_size"],
        embedding_dim=bundle["embedding_dim"],
        hidden_dim=bundle["hidden_dim"],
        dropout=bundle["dropout"],
        num_classes=bundle["num_classes"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    preds_all: list[int] = []
    total = x_test_sparse.shape[0]
    num_batches = (total + args.batch_size - 1) // args.batch_size

    print("Running predictions...")
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, total)
            x_batch_csr = x_test_sparse[start:end]
            token_ids, offsets, token_weights = csr_batch_to_embedding_inputs(x_batch_csr, device)
            logits = model(token_ids, offsets, token_weights)
            pred_idx = logits.argmax(dim=1).cpu().tolist()
            preds_all.extend([label_values[idx] for idx in pred_idx])

            print(f"Prediction progress: batch {batch_idx + 1}/{num_batches} ({end}/{total} rows)")

    accuracy = accuracy_score(y_true, preds_all)
    report = classification_report(y_true, preds_all, digits=4)

    result_text = (
        "TFIDF + PyTorch MLP (polarity_label prediction)\n"
        f"Shared vectorizer: {args.vectorizer_path}\n"
        f"Test rows: {len(documents)}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(result_text, encoding="utf-8")

    print(f"Test complete. Results written to: {args.results_path}")


if __name__ == "__main__":
    main()

