from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from transformers import AutoModel, AutoTokenizer


class FrozenBertGruClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        gru_hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = gru_hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence_output)
        features = hidden[-1] if self.gru.bidirectional is False else torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(features))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test frozen DistilBERT + GRU score model on the testing dataset."
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
        default=Path("models/bert_gru/score_prediction/saved_model"),
        help="Directory containing the trained frozen BERT + GRU model.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("models/bert_gru/score_prediction/results.txt"),
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
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.1,
        help="Fraction of dataset to keep from the top (default: first 10%%).",
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

    model_path = args.model_dir / "bert_gru_score_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    df = pd.read_csv(args.input)
    original_rows = len(df)
    df = select_top_fraction(df, args.data_fraction)
    print(
        f"Using first {len(df)} rows out of {original_rows} total rows "
        f"(data_fraction={args.data_fraction})."
    )

    required_columns = {"review_text", "user_name", "score_label"}
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    text_block = df.loc[:, "review_text":"user_name"].copy()
    text_block = text_block.fillna("").astype(str)
    documents = build_documents_with_progress(text_block, args.progress_every).tolist()
    y_true = df["score_label"].astype(int).to_numpy()

    print(f"Loading model from: {model_path}")
    bundle = torch.load(model_path, map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(bundle["bert_model_name"], use_fast=True)
    bert = AutoModel.from_pretrained(bundle["bert_model_name"])
    bert.eval()
    for param in bert.parameters():
        param.requires_grad = False

    gru_head = FrozenBertGruClassifier(
        hidden_size=bundle["hidden_size"],
        num_classes=bundle["num_classes"],
        gru_hidden_size=bundle["gru_hidden_size"],
        num_layers=bundle["gru_num_layers"],
        bidirectional=bundle["bidirectional"],
        dropout=bundle["dropout"],
    )
    gru_head.load_state_dict(bundle["gru_state_dict"])
    gru_head.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    bert = bert.to(device)
    gru_head = gru_head.to(device)

    preds_all: list[int] = []
    total = len(documents)
    num_batches = (total + args.batch_size - 1) // args.batch_size

    print("Running predictions...")
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, total)
            batch_docs = documents[start:end]

            encoded = tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                max_length=bundle["max_length"],
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            sequence_output = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            logits = gru_head(sequence_output)
            pred = logits.argmax(dim=1).cpu().numpy() + 1
            preds_all.extend(pred.tolist())

            print(f"Prediction progress: batch {batch_idx + 1}/{num_batches} ({end}/{total} rows)")

    accuracy = accuracy_score(y_true, preds_all)
    report = classification_report(y_true, preds_all, digits=4)

    result_text = (
        "Frozen DistilBERT + GRU (score_label prediction)\n"
        f"BERT model: {bundle['bert_model_name']}\n"
        f"Test rows: {len(documents)}\n"
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

