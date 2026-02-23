from __future__ import annotations

import argparse
import ast
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-shot score prediction with LM Studio (ministral3)."
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
        default=2,
        help="Number of test rows to predict.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("llm/score_prediction/few_shot/predictions.csv"),
        help="Where to write row-level predictions.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("llm/score_prediction/few_shot/results.txt"),
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
                    "You are a strict JSON generator for Yelp review rating analysis."
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


def build_prompt(review_text: str) -> str:
    return (
        "Predict Yelp rating and aspect scores using the examples, then solve the final review.\n"
        "Return JSON only with this schema:\n"
        '{ "overall_score": <1..5>, "aspects": { "<aspect>": <1..5> } }\n\n'
        "Example 1:\n"
        "Review: [PUT YOUR EXAMPLE REVIEW 1 HERE]\n"
        'Output: { "overall_score": [PUT 1..5], "aspects": { "service": [PUT 1..5] } }\n\n'
        "Example 2:\n"
        "Review: [PUT YOUR EXAMPLE REVIEW 2 HERE]\n"
        'Output: { "overall_score": [PUT 1..5], "aspects": { "food": [PUT 1..5], "price": [PUT 1..5] } }\n\n'
        "Example 3:\n"
        "Review: [PUT YOUR EXAMPLE REVIEW 3 HERE]\n"
        'Output: { "overall_score": [PUT 1..5], "aspects": { "cleanliness": [PUT 1..5] } }\n\n'
        "Now predict for this review:\n"
        f"Review: {review_text}\n"
        "JSON:"
    )


def extract_json(raw_output: str) -> dict:
    if not raw_output:
        return {}

    text = raw_output.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    candidates: list[str] = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    cleaned_candidates: list[str] = []
    for candidate in candidates:
        cleaned_candidates.append(candidate)
        # Common LLM formatting issue: trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        if cleaned != candidate:
            cleaned_candidates.append(cleaned)

    for candidate in cleaned_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass

    return {}


def clamp_score(value: object) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 5:
        return score
    return None


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    required_columns = {"review_text", "score_label"}
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    test_df = df.head(args.limit).copy()
    if test_df.empty:
        raise ValueError("No rows available for testing.")

    predicted_scores: list[int | None] = []
    predicted_aspects_json: list[str] = []
    raw_outputs: list[str] = []

    for idx, row in enumerate(test_df.itertuples(index=False), start=1):
        prompt = build_prompt(str(row.review_text))
        try:
            raw = call_chat_completion(args.base_url, args.model, prompt)
            parsed = extract_json(raw)
        except Exception as exc:
            raw = f"__ERROR__: {exc}"
            parsed = {}
        overall = clamp_score(parsed.get("overall_score"))
        aspects = parsed.get("aspects", {})
        if not isinstance(aspects, dict):
            aspects = {}

        clean_aspects: dict[str, int] = {}
        for k, v in aspects.items():
            aspect_score = clamp_score(v)
            if aspect_score is not None:
                clean_aspects[str(k)] = aspect_score

        predicted_scores.append(overall)
        predicted_aspects_json.append(json.dumps(clean_aspects, ensure_ascii=False))
        raw_outputs.append(raw)
        print(f"Processed {idx}/{len(test_df)}")

    test_df["predicted_score_label"] = predicted_scores
    test_df["predicted_aspects_scores"] = predicted_aspects_json
    test_df["raw_model_output"] = raw_outputs

    eval_df = test_df.dropna(subset=["predicted_score_label"]).copy()
    y_true = eval_df["score_label"].astype(int)
    y_pred = eval_df["predicted_score_label"].astype(int)

    accuracy = accuracy_score(y_true, y_pred) if not eval_df.empty else 0.0
    report = (
        classification_report(y_true, y_pred, digits=4)
        if not eval_df.empty
        else "No valid predictions to evaluate."
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df = test_df.loc[:, ["predicted_score_label", "score_label", "review_text"]].copy()
    output_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    result_text = (
        "LLM Few-Shot Score Prediction + Aspect Scoring (ministral3 via LM Studio)\n"
        f"Input: {args.input}\n"
        f"Rows requested: {args.limit}\n"
        f"Rows predicted: {len(test_df)}\n"
        f"Rows with valid overall score output: {len(eval_df)}\n"
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
