from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for polarity prediction."
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
        default=Path("models/bert_finetune/polarity_prediction/saved_model"),
        help="Directory where the trained model bundle will be saved.",
    )
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="distilbert-base-uncased",
        help="BERT-family model name from Hugging Face.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Number of rows between progress updates while building text documents.",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.02,
        help="Fraction of dataset to keep from the top.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW.",
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
    texts = build_documents_with_progress(text_block, args.progress_every).tolist()

    raw_labels = df["polarity_label"].astype(int).to_numpy()
    label_values = sorted(np.unique(raw_labels).tolist())
    label_to_index = {label: idx for idx, label in enumerate(label_values)}
    y_train = np.array([label_to_index[v] for v in raw_labels], dtype=np.int64)

    print(f"Loading tokenizer and model: {args.bert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.bert_model_name, num_labels=len(label_values)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    dataset = TextLabelDataset(texts, y_train)

    def collate_fn(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        batch_texts = [item[0] for item in batch]
        batch_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = batch_labels
        return encoded

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = math.ceil(len(dataset) / args.batch_size)
    print(
        f"Fine-tuning BERT for {args.epochs} epoch(s), batch_size={args.batch_size}, "
        f"steps_per_epoch={total_steps}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        print(f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

            bs = batch["labels"].size(0)
            running_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == batch["labels"]).sum().item()
            seen += bs

            if step % 10 == 0 or step == total_steps:
                print(
                    f"  Step {step}/{total_steps} - "
                    f"loss: {running_loss / seen:.4f} - acc: {correct / seen:.4f} - seen: {seen}/{len(dataset)}"
                )

        print(
            f"Epoch {epoch} done - loss: {running_loss / len(dataset):.4f} - "
            f"acc: {correct / len(dataset):.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "bert_finetune_polarity_model.pt"
    bundle = {
        "state_dict": model.state_dict(),
        "bert_model_name": args.bert_model_name,
        "max_length": args.max_length,
        "num_labels": len(label_values),
        "label_values": label_values,
        "data_fraction": args.data_fraction,
    }
    torch.save(bundle, model_path)

    print(f"Saved model to: {model_path}")
    print(f"Training rows: {len(dataset)}")


if __name__ == "__main__":
    main()
