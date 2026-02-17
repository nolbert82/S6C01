from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and save BERT embeddings for the first 90% of the training dataset."
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
        default=Path("text_representations/BERT/saved_model"),
        help="Directory where BERT artifacts and embeddings are saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model name to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size used to encode documents.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum token length for truncation.",
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
        raise FileNotFoundError(
            f"Input file not found: {args.input}. Run scripts/prepare_training_data.py first."
        )

    df = pd.read_csv(args.input)

    if "review_text" not in df.columns or "user_name" not in df.columns:
        raise KeyError(
            "Input dataset must contain both 'review_text' and 'user_name' columns."
        )

    text_block = df.loc[:, "review_text":"user_name"].copy()
    text_block = text_block.fillna("").astype(str)
    documents = build_documents_with_progress(text_block, args.progress_every)

    split_index = max(1, math.floor(len(documents) * 0.9))
    train_documents = documents.iloc[:split_index].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading tokenizer/model: {args.model_name}")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    total = len(train_documents)
    total_batches = math.ceil(total / args.batch_size)
    print(f"Encoding {total} documents in {total_batches} batches...")

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, total, args.batch_size), start=1):
            batch_texts = train_documents[start : start + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_embeddings)

            pct = (batch_idx / total_batches) * 100
            print(
                f"Encoding progress: batch {batch_idx}/{total_batches} ({pct:.2f}%)"
            )

    all_embeddings = np.vstack(embeddings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "train_embeddings.npy", all_embeddings)
    pd.DataFrame({"row_index": range(split_index)}).to_csv(
        args.output_dir / "train_row_indices.csv", index=False
    )

    model_dir = args.output_dir / "model"
    tokenizer_dir = args.output_dir / "tokenizer"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    print(f"Saved embeddings to: {args.output_dir / 'train_embeddings.npy'}")
    print(f"Saved row indices to: {args.output_dir / 'train_row_indices.csv'}")
    print(f"Saved model to: {model_dir}")
    print(f"Saved tokenizer to: {tokenizer_dir}")
    print(f"Total rows: {len(documents)} | Training rows (90%): {split_index}")
    print(f"Embeddings shape: {all_embeddings.shape}")


if __name__ == "__main__":
    main()
