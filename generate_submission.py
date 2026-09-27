"""
generate_submission.py — produce submission.jsonl for the 30 canonical
(merchant, trigger[, customer]) test pairs from challenge-brief.md §6.

Usage:
    python generate_dataset.py --seed-dir . --out ./expanded   # (in dataset/, once)
    ANTHROPIC_API_KEY=sk-... python generate_submission.py \
        --dataset-dir ../magicpin-challenge/dataset/expanded \
        --out submission.jsonl

Without ANTHROPIC_API_KEY set, runs in deterministic template-fallback mode
(still produces valid, on-brief output — just less rich than the LLM path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import composer


def load_dataset(dataset_dir: Path):
    categories = {}
    for f in (dataset_dir / "categories").glob("*.json"):
        data = json.loads(f.read_text())
        categories[data["slug"]] = data

    merchants = {}
    for f in (dataset_dir / "merchants").glob("*.json"):
        data = json.loads(f.read_text())
        merchants[data["merchant_id"]] = data

    customers = {}
    for f in (dataset_dir / "customers").glob("*.json"):
        data = json.loads(f.read_text())
        customers[data["customer_id"]] = data

    triggers = {}
    for f in (dataset_dir / "triggers").glob("*.json"):
        data = json.loads(f.read_text())
        triggers[data["id"]] = data

    test_pairs = json.loads((dataset_dir / "test_pairs.json").read_text())["pairs"]

    return categories, merchants, customers, triggers, test_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", default="submission.jsonl")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    categories, merchants, customers, triggers, test_pairs = load_dataset(dataset_dir)

    print(f"Loaded {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers, "
          f"{len(test_pairs)} test pairs")
    print(f"LLM mode: {'ON (' + composer.MODEL + ')' if composer._client else 'OFF — template fallback'}")

    lines = []
    for pair in test_pairs:
        trigger = triggers.get(pair["trigger_id"])
        merchant = merchants.get(pair["merchant_id"])
        if not trigger or not merchant:
            print(f"  SKIP {pair['test_id']} — missing trigger or merchant")
            continue
        category = categories.get(merchant.get("category_slug"))
        if not category:
            print(f"  SKIP {pair['test_id']} — missing category")
            continue
        customer = customers.get(pair.get("customer_id")) if pair.get("customer_id") else None

        composed = composer.compose(category, merchant, trigger, customer)
        lines.append({
            "test_id": pair["test_id"],
            "body": composed["body"],
            "cta": composed["cta"],
            "send_as": composed["send_as"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        })
        print(f"  {pair['test_id']} [{merchant.get('category_slug')}] -> {composed['body'][:70]}")

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
