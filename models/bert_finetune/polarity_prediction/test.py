from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test fine-tuned DistilBERT polarity model on the testing dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prepared/testing_dataset.csv"),
        help="Path to the prepared testing dataset.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("models/bert_finetune/polarity_prediction/saved_model"),
        help="Directory containing the trained fine-tuned BERT model.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("models/bert_finetune/polarity_prediction/results.txt"),
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
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.1,
        help="Fraction of dataset to keep from the top.",
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


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    model_path = args.model_dir / "bert_finetune_polarity_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

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
    y_true = df["polarity_label"].astype(int).to_numpy()

    print(f"Loading model from: {model_path}")
    bundle = torch.load(model_path, map_location="cpu")
    label_values = bundle["label_values"]
    label_to_index = {label: idx for idx, label in enumerate(label_values)}

    tokenizer = AutoTokenizer.from_pretrained(bundle["bert_model_name"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        bundle["bert_model_name"], num_labels=bundle["num_labels"]
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    dataset = TextLabelDataset(texts, [label_to_index[v] for v in y_true.tolist()])

    def collate_fn(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        batch_texts = [item[0] for item in batch]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=bundle["max_length"],
            return_tensors="pt",
        )
        return encoded

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    pred_indices: list[int] = []
    total = len(dataset)
    num_batches = (total + args.batch_size - 1) // args.batch_size

    print("Running predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            pred_indices.extend(pred)
            done = min(batch_idx * args.batch_size, total)
            print(f"Prediction progress: batch {batch_idx}/{num_batches} ({done}/{total} rows)")

    y_pred = np.array([label_values[idx] for idx in pred_indices], dtype=np.int64)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    result_text = (
        "Fine-tuned DistilBERT (polarity_label prediction)\n"
        f"BERT model: {bundle['bert_model_name']}\n"
        f"Test rows: {len(dataset)}\n"
        f"Data fraction: {args.data_fraction}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(result_text, encoding="utf-8")
    print(f"Test complete. Results written to: {args.results_path}")


if __name__ == "__main__":
    main()
