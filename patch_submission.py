"""
patch_submission.py — Regenerate only the placeholder (failed) lines in submission.jsonl

Usage:
    python patch_submission.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from composer import compose

load_dotenv()

EXPANDED_DIR = Path(__file__).parent / "dataset" / "expanded"
SUBMISSION_PATH = Path(__file__).parent / "submission.jsonl"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    if not SUBMISSION_PATH.exists():
        print("submission.jsonl not found — run generate_submission.py first")
        sys.exit(1)

    # Load existing lines
    lines = []
    with open(SUBMISSION_PATH, encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if row:
                lines.append(json.loads(row))

    # Find placeholders (where rationale starts with "Placeholder")
    pairs_path = EXPANDED_DIR / "test_pairs.json"
    pairs = {p["test_id"]: p for p in load_json(pairs_path)["pairs"]}

    patched = 0
    for i, line in enumerate(lines):
        if "Placeholder" not in line.get("rationale", ""):
            continue

        test_id = line["test_id"]
        pair = pairs.get(test_id)
        if not pair:
            continue

        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        print(f"  Patching {test_id} ...", end=" ", flush=True)

        try:
            trigger_path = EXPANDED_DIR / "triggers" / f"{trigger_id}.json"
            trigger = load_json(trigger_path)

            merchant_path = EXPANDED_DIR / "merchants" / f"{merchant_id}.json"
            merchant = load_json(merchant_path)

            cat_slug = merchant.get("category_slug", "")
            cat_path = EXPANDED_DIR / "categories" / f"{cat_slug}.json"
            category = load_json(cat_path)

            customer = None
            if customer_id:
                cust_path = EXPANDED_DIR / "customers" / f"{customer_id}.json"
                if cust_path.exists():
                    customer = load_json(cust_path)

            result = compose(category, merchant, trigger, customer)
            result["test_id"] = test_id
            lines[i] = result
            patched += 1
            print("OK")

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(0.5)

    # Write back
    with open(SUBMISSION_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nPatched {patched} lines. submission.jsonl now has {len(lines)} entries.")


if __name__ == "__main__":
    main()
