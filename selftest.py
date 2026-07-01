"""
selftest.py — Smoke-test the active LLM provider before a re-score.

Fires one real compose() and one compose_reply() through whichever provider is
selected by env vars (LLM_PROVIDER / LLM_MODEL / API keys), using the local
dataset. Prints the outputs so you can eyeball quality and confirm the model
is reachable — without deploying or spending a judging slot.

Usage (PowerShell):
  $env:LLM_PROVIDER="anthropic"; $env:ANTHROPIC_API_KEY="sk-ant-..."; python selftest.py
Usage (bash):
  LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python selftest.py

Defaults to Groq if nothing is set.
"""

from __future__ import annotations

import json
import os
import sys

from composer import compose, compose_reply
from llm import active_model_label

DATASET = os.path.join(os.path.dirname(__file__), "dataset", "expanded")


def _load(rel: str) -> dict:
    with open(os.path.join(DATASET, rel), encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    print(f"Active model: {active_model_label()}\n")

    try:
        cat = _load("categories/dentists.json")
        merchant = _load("merchants/m_001_drmeera_dentist_delhi.json")
        trigger = _load("triggers/trg_001_research_digest_dentists.json")
    except FileNotFoundError as e:
        print(f"[SKIP] dataset not found: {e}")
        return 2

    print("── compose() — research_digest ──")
    out = compose(cat, merchant, trigger)
    print("body     :", out.get("body"))
    print("cta      :", out.get("cta"), "| send_as:", out.get("send_as"))
    print("rationale:", out.get("rationale"))

    print("\n── compose_reply() — intent transition ('ok go ahead') ──")
    history = [
        {"from": "vera", "body": out.get("body", "")},
        {"from": "merchant", "message": "ok go ahead"},
    ]
    reply = compose_reply(cat, merchant, trigger, history, "ok go ahead")
    print("action:", reply.get("action"))
    print("body  :", reply.get("body"))

    # Basic sanity checks the judge cares about.
    problems = []
    if not (out.get("body") or "").strip():
        problems.append("compose returned empty body")
    if reply.get("action") not in ("send", "wait", "end"):
        problems.append(f"reply action invalid: {reply.get('action')}")
    if problems:
        print("\n[FAIL]", "; ".join(problems))
        return 1

    print("\n[OK] provider reachable, outputs well-formed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
