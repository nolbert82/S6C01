from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
        description="Train a PyTorch MLP model to predict polarity_label using a shared CountVectorizer."
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
        "--output_dir",
        type=Path,
        default=Path("models/countvectorizer_mlp/polarity_prediction/saved_model"),
        help="Directory where the trained model bundle will be saved.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Number of rows between progress updates while building text documents.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden layer size for MLP.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension used by EmbeddingBag.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability in MLP layers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Train only on the first N rows of the dataset. Use 0 or negative to use all rows.",
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    set_seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. Run scripts/prepare_data/prepare_training_data.py first."
        )

    if not args.vectorizer_path.exists():
        raise FileNotFoundError(
            f"Shared vectorizer not found: {args.vectorizer_path}. Run text_representations/CountVectorizer/train.py first."
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
    label_values = sorted(labels.unique().tolist())

    print(f"Loading shared CountVectorizer from: {args.vectorizer_path}")
    with args.vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)

    print(f"Vectorizing training data (rows: {len(documents)})...")
    x_train_sparse = vectorizer.transform(documents)
    print("Vectorization complete.")

    label_to_index = {label: idx for idx, label in enumerate(label_values)}
    y_train_np = np.array([label_to_index[v] for v in labels.to_numpy()], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_size = x_train_sparse.shape[1]
    num_classes = len(label_values)
    model = BowEmbeddingMLPClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"Training MLP for {args.epochs} epoch(s), batch_size={args.batch_size}, "
        f"steps_per_epoch={math.ceil(len(y_train_np) / args.batch_size)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        n_rows = len(y_train_np)
        shuffled_indices = np.random.permutation(n_rows)
        total_steps = math.ceil(n_rows / args.batch_size)

        print(f"Epoch {epoch}/{args.epochs}")
        for step in range(1, total_steps + 1):
            start = (step - 1) * args.batch_size
            end = min(start + args.batch_size, n_rows)
            batch_idx = shuffled_indices[start:end]

            x_batch_csr = x_train_sparse[batch_idx]
            yb = torch.tensor(y_train_np[batch_idx], dtype=torch.long, device=device)
            token_ids, offsets, token_weights = csr_batch_to_embedding_inputs(x_batch_csr, device)

            optimizer.zero_grad()
            logits = model(token_ids, offsets, token_weights)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            seen += yb.size(0)

            if step % 10 == 0 or step == total_steps:
                avg_loss = running_loss / seen
                acc = correct / seen
                print(
                    f"  Step {step}/{total_steps} - "
                    f"loss: {avg_loss:.4f} - acc: {acc:.4f} - seen: {seen}/{n_rows}"
                )

        epoch_loss = running_loss / n_rows
        epoch_acc = correct / n_rows
        print(f"Epoch {epoch} done - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "mlp_polarity_model.pt"

    bundle = {
        "state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_classes": num_classes,
        "label_values": label_values,
    }
    torch.save(bundle, model_path)

    print(f"Saved model to: {model_path}")
    print(f"Using shared vectorizer: {args.vectorizer_path}")
    print(f"Training rows: {len(documents)}")


if __name__ == "__main__":
    main()
