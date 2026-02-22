from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class FrozenBertCnnClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_filters: int,
        kernel_sizes: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden_size, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        x = sequence_output.transpose(1, 2)
        pooled_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max(conv_out, dim=2).values
            pooled_outputs.append(pooled)
        features = torch.cat(pooled_outputs, dim=1)
        return self.classifier(self.dropout(features))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen DistilBERT + CNN model for polarity prediction."
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
        default=Path("models/bert_cnn/polarity_prediction/saved_model"),
        help="Directory where the trained model bundle will be saved.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Number of rows between progress updates while building text documents.",
    )
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Fast BERT-family model name from Hugging Face.",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.1,
        help="Fraction of dataset to keep from the top (default: first 10%%).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        "--num_filters",
        type=int,
        default=128,
        help="Number of CNN filters per kernel size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout in CNN head.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed.",
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


def select_top_fraction(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if fraction <= 0 or fraction > 1:
        raise ValueError("--data_fraction must be in (0, 1].")
    rows = max(1, int(len(df) * fraction))
    return df.head(rows).copy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. Run scripts/prepare_data/prepare_training_data.py first."
        )

    df = pd.read_csv(args.input)
    original_rows = len(df)
    df = select_top_fraction(df, args.data_fraction)
    print(
        f"Using first {len(df)} rows out of {original_rows} total rows "
        f"(data_fraction={args.data_fraction})."
    )

    required_columns = {"review_text", "user_name", "polarity_label"}
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    text_block = df.loc[:, "review_text":"user_name"].copy()
    text_block = text_block.fillna("").astype(str)
    documents = build_documents_with_progress(text_block, args.progress_every).tolist()
    labels = df["polarity_label"].astype(int).to_numpy()
    label_values = sorted(np.unique(labels).tolist())
    label_to_index = {label: idx for idx, label in enumerate(label_values)}
    y_train = np.array([label_to_index[v] for v in labels], dtype=np.int64)

    print(f"Loading tokenizer and frozen encoder: {args.bert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, use_fast=True)
    bert = AutoModel.from_pretrained(args.bert_model_name)
    bert.eval()
    for param in bert.parameters():
        param.requires_grad = False

    hidden_size = int(bert.config.hidden_size)
    kernel_sizes = (3, 4, 5)
    cnn_head = FrozenBertCnnClassifier(
        hidden_size=hidden_size,
        num_classes=len(label_values),
        num_filters=args.num_filters,
        kernel_sizes=kernel_sizes,
        dropout=args.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    bert = bert.to(device)
    cnn_head = cnn_head.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_rows = len(documents)
    steps_per_epoch = math.ceil(n_rows / args.batch_size)
    print(
        f"Training frozen BERT + CNN for {args.epochs} epoch(s), batch_size={args.batch_size}, "
        f"steps_per_epoch={steps_per_epoch}"
    )

    for epoch in range(1, args.epochs + 1):
        cnn_head.train()
        shuffled_indices = np.random.permutation(n_rows)
        running_loss = 0.0
        correct = 0
        seen = 0

        print(f"Epoch {epoch}/{args.epochs}")
        for step in range(1, steps_per_epoch + 1):
            start = (step - 1) * args.batch_size
            end = min(start + args.batch_size, n_rows)
            batch_idx = shuffled_indices[start:end]
            batch_docs = [documents[i] for i in batch_idx]
            yb = torch.tensor(y_train[batch_idx], dtype=torch.long, device=device)

            encoded = tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                sequence_output = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            optimizer.zero_grad()
            logits = cnn_head(sequence_output)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            seen += yb.size(0)

            if step % 10 == 0 or step == steps_per_epoch:
                avg_loss = running_loss / seen
                acc = correct / seen
                print(
                    f"  Step {step}/{steps_per_epoch} - "
                    f"loss: {avg_loss:.4f} - acc: {acc:.4f} - seen: {seen}/{n_rows}"
                )

        epoch_loss = running_loss / n_rows
        epoch_acc = correct / n_rows
        print(f"Epoch {epoch} done - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "bert_cnn_polarity_model.pt"
    bundle = {
        "cnn_state_dict": cnn_head.state_dict(),
        "bert_model_name": args.bert_model_name,
        "max_length": args.max_length,
        "hidden_size": hidden_size,
        "num_filters": args.num_filters,
        "kernel_sizes": kernel_sizes,
        "dropout": args.dropout,
        "num_classes": len(label_values),
        "label_values": label_values,
        "data_fraction": args.data_fraction,
    }
    torch.save(bundle, model_path)

    print(f"Saved model to: {model_path}")
    print(f"Training rows: {len(documents)}")


if __name__ == "__main__":
    main()

