#!/usr/bin/env python3
"""
generate_submission.py — runs bot.compose() over the 30 canonical
(merchant, trigger[, customer]) test pairs and writes submission.jsonl,
per challenge-brief.md §6-7.

Usage:
    python generate_submission.py --dataset ./dataset --out ./submission.jsonl
"""
import argparse
import json
from pathlib import Path

import bot


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./dataset")
    ap.add_argument("--out", default="./submission.jsonl")
    args = ap.parse_args()

    ds = Path(args.dataset)
    pairs = load_json(ds / "test_pairs.json")["pairs"]

    rows = []
    errors = []
    for pair in pairs:
        test_id = pair["test_id"]
        try:
            trigger = load_json(ds / "triggers" / f"{pair['trigger_id']}.json")
            merchant = load_json(ds / "merchants" / f"{pair['merchant_id']}.json")
            category = load_json(ds / "categories" / f"{merchant['category_slug']}.json")
            customer = None
            if pair.get("customer_id"):
                customer = load_json(ds / "customers" / f"{pair['customer_id']}.json")

            result = bot.compose(category, merchant, trigger, customer)
            row = {"test_id": test_id, **result}
            rows.append(row)
        except Exception as e:  # noqa: BLE001 — we want every test_id to produce a row
            errors.append((test_id, str(e)))
            rows.append({
                "test_id": test_id,
                "body": "",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": pair.get("trigger_id", ""),
                "rationale": f"ERROR: {e}",
            })

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out}")
    if errors:
        print(f"{len(errors)} test pairs errored:")
        for tid, msg in errors:
            print(f"  {tid}: {msg}")
    else:
        print("All 30 test pairs composed successfully.")


if __name__ == "__main__":
    main()
