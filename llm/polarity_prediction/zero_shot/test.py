from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


LABEL_TO_INT = {"negative": -1, "neutral": 0, "positive": 1}
VALID_INT_LABELS = {-1, 0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot polarity prediction with LM Studio (ministral3)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prepared/testing_dataset.csv"),
        help="Path to the prepared testing dataset.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:1234",
        help="LM Studio base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ministral3",
        help="Model name served by LM Studio.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of test rows to predict.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("llm/polarity_prediction/zero_shot/predictions.csv"),
        help="Where to write row-level predictions.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("llm/polarity_prediction/zero_shot/results.txt"),
        help="Where to write evaluation results.",
    )
    return parser.parse_args()


def call_chat_completion(base_url: str, model: str, prompt: str) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a sentiment classifier. Return exactly one label: "
                    "negative, neutral, or positive."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio request failed: {exc}") from exc

    data = json.loads(body)
    return data["choices"][0]["message"]["content"].strip()


def extract_label(raw_output: str) -> str:
    matches = re.findall(r"\b(negative|neutral|positive)\b", raw_output.lower())
    if matches:
        return matches[-1]
    return "unknown"


def build_prompt(review_text: str) -> str:
    return (
        "Classify the sentiment of the following Yelp review.\n"
        "Output exactly one word: negative, neutral, or positive.\n\n"
        f"Review:\n{review_text}"
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    required_columns = {"review_text", "polarity_label"}
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    test_df = df.head(args.limit).copy()
    if test_df.empty:
        raise ValueError("No rows available for testing.")

    predictions: list[int | None] = []
    raw_outputs: list[str] = []

    for idx, row in enumerate(test_df.itertuples(index=False), start=1):
        prompt = build_prompt(str(row.review_text))
        raw = call_chat_completion(args.base_url, args.model, prompt)
        pred_label = extract_label(raw)
        pred = LABEL_TO_INT.get(pred_label)
        predictions.append(pred)
        raw_outputs.append(raw)
        print(f"Processed {idx}/{len(test_df)}")

    test_df["predicted_polarity_label"] = predictions
    test_df["raw_model_output"] = raw_outputs

    valid_mask = test_df["predicted_polarity_label"].isin(VALID_INT_LABELS)
    eval_df = test_df.loc[valid_mask].copy()
    true_labels = eval_df["polarity_label"].astype(int)
    pred_labels = eval_df["predicted_polarity_label"].astype(int)

    accuracy = accuracy_score(true_labels, pred_labels) if not eval_df.empty else 0.0
    report = (
        classification_report(true_labels, pred_labels, digits=4, zero_division=0)
        if not eval_df.empty
        else "No valid predictions to evaluate."
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df = test_df.loc[:, ["predicted_polarity_label", "polarity_label", "review_text"]].copy()
    output_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    result_text = (
        "LLM Zero-Shot Polarity Prediction (ministral3 via LM Studio)\n"
        f"Input: {args.input}\n"
        f"Rows requested: {args.limit}\n"
        f"Rows predicted: {len(test_df)}\n"
        f"Rows with valid label output: {len(eval_df)}\n"
        f"Accuracy (valid outputs only): {accuracy:.4f}\n\n"
        "Classification report (valid outputs only):\n"
        f"{report}\n"
    )
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(result_text, encoding="utf-8")

    print(f"Predictions written to: {args.output_csv}")
    print(f"Results written to: {args.results_path}")


if __name__ == "__main__":
    main()
