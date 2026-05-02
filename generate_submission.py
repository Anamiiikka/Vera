"""
generate_submission.py — Generate submission.jsonl for the 30 canonical test pairs.

Usage:
    python generate_submission.py

Reads:  dataset/expanded/test_pairs.json
        dataset/expanded/{categories,merchants,customers,triggers}/
Writes: submission.jsonl  (30 lines, one JSON per line)
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


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_file(directory: Path, id_str: str) -> Path | None:
    """Find a JSON file whose stem starts with id_str or equals id_str."""
    for f in directory.glob("*.json"):
        if f.stem == id_str or f.stem.startswith(id_str):
            return f
    return None


def load_context(kind: str, context_id: str) -> dict | None:
    folder = EXPANDED_DIR / (kind + "s")  # categories, merchants, customers, triggers
    f = find_file(folder, context_id)
    if f:
        return load_json(f)
    return None


def main():
    pairs_path = EXPANDED_DIR / "test_pairs.json"
    if not pairs_path.exists():
        print(f"ERROR: {pairs_path} not found. Run: python dataset/generate_dataset.py --out dataset/expanded", file=sys.stderr)
        sys.exit(1)

    pairs = load_json(pairs_path)["pairs"]
    print(f"Found {len(pairs)} test pairs. Composing messages...")

    out_lines = []
    errors = 0

    for pair in pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        print(f"  [{test_id}] merchant={merchant_id}, trigger={trigger_id}", end=" ... ", flush=True)

        try:
            # Load trigger
            trigger = load_context("trigger", trigger_id)
            if not trigger:
                raise FileNotFoundError(f"trigger {trigger_id}")

            # Load merchant
            merchant = load_context("merchant", merchant_id)
            if not merchant:
                raise FileNotFoundError(f"merchant {merchant_id}")

            # Load category
            cat_slug = merchant.get("category_slug", "")
            # Load category directly from categories folder
            cat_path = EXPANDED_DIR / "categories" / f"{cat_slug}.json"
            if cat_path.exists():
                category = load_json(cat_path)
            else:
                raise FileNotFoundError(f"category {cat_slug}")

            # Load customer (optional)
            customer = load_context("customer", customer_id) if customer_id else None

            result = compose(category, merchant, trigger, customer)
            result["test_id"] = test_id
            out_lines.append(result)
            print("OK")

        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
            # Write a placeholder so line count stays at 30
            out_lines.append({
                "test_id": test_id,
                "body": "Namaskar! Aapke business ke baare mein kuch important share karna tha. Kya aap abhi baat kar sakte hain?",
                "cta": "yes_stop",
                "send_as": "vera",
                "suppression_key": f"placeholder_{test_id}",
                "rationale": f"Placeholder — error during generation: {e}",
            })

        # Rate limit: be nice to Groq free tier
        time.sleep(0.5)

    out_path = Path(__file__).parent / "submission.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nDone. Written {len(out_lines)} lines to {out_path}")
    if errors:
        print(f"  ({errors} errors — check placeholders above)")


if __name__ == "__main__":
    main()
