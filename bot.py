"""
bot.py — Vera AI Bot Server (FastAPI)

Implements all 5 endpoints required by the magicpin judge harness:
  GET  /v1/healthz       — liveness probe
  GET  /v1/metadata      — bot identity
  POST /v1/context       — receive context push (idempotent by version)
  POST /v1/tick          — periodic wake-up; bot decides what to send
  POST /v1/reply         — receive merchant/customer reply; respond

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

from composer import compose, compose_reply
from conversation_handlers import (
    ConversationState,
    ConvState,
    decide_on_reply,
    auto_reply_retry_message,
    graceful_exit_message,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Vera Bot", version="1.0.0")

# Bypass ngrok browser-warning interstitial for all API responses
class NgrokBypassMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response

app.add_middleware(NgrokBypassMiddleware)

START_TIME = time.time()

# ── In-memory state ───────────────────────────────────────────────────────────
# (scope, context_id) → {version: int, payload: dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id → ConversationState
conversations: dict[str, ConversationState] = {}

# suppression_key → True  (already sent in this session)
sent_suppression: set[str] = set()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_payload(scope: str, context_id: str) -> dict | None:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _count_contexts() -> dict[str, int]:
    counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _merchant_lang(merchant: dict) -> str:
    langs = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in langs:
        return "hi-en"
    return "en"


# ── Pydantic models ───────────────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _count_contexts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "VeraPlus",
        "team_members": ["Participant"],
        "model": "llama-3.3-70b-versatile (Groq)",
        "approach": (
            "Trigger-routed single-prompt composer with Groq LLaMA-3.3-70B. "
            "Each trigger kind gets a specialized instruction. "
            "Multi-turn: auto-reply detection + intent state machine. "
            "Anti-hallucination: only context-provided data is used."
        ),
        "contact_email": "participant@example.com",
        "version": "1.0.0",
        "submitted_at": _now_iso(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope: {body.scope}"}

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current and current["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"],
        }

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        # Skip if already suppressed this session
        trg_entry = contexts.get(("trigger", trg_id))
        if not trg_entry:
            continue
        trg = trg_entry["payload"]

        sup_key = trg.get("suppression_key", "")
        if sup_key and sup_key in sent_suppression:
            continue

        # Check expiry
        expires_at = trg.get("expires_at", "")
        if expires_at and expires_at < body.now:
            continue

        merchant_id = trg.get("merchant_id")
        if not merchant_id:
            continue

        merchant = _get_payload("merchant", merchant_id)
        if not merchant:
            continue

        cat_slug = merchant.get("category_slug", "")
        category = _get_payload("category", cat_slug)
        if not category:
            continue

        # Customer context (optional)
        customer_id = trg.get("customer_id")
        customer = _get_payload("customer", customer_id) if customer_id else None

        # Check 20-action-per-tick cap
        if len(actions) >= 20:
            break

        # Generate conversation_id
        conv_id = f"conv_{merchant_id}_{trg_id}"

        # Skip if conversation already active (use /v1/reply to continue)
        if conv_id in conversations and not conversations[conv_id].is_closed():
            continue

        try:
            result = compose(category, merchant, trg, customer)
        except Exception as e:
            # Don't crash the tick — just skip this trigger
            print(f"[COMPOSE ERROR] {trg_id}: {e}")
            continue

        # Track suppression
        if sup_key:
            sent_suppression.add(sup_key)

        # Track conversation state
        conv_state = ConversationState(
            conversation_id=conv_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            trigger_id=trg_id,
            send_as=result.get("send_as", "vera"),
        )
        conv_state.add_bot_turn(result["body"], result.get("cta", "open_ended"))
        conv_state.transition(ConvState.OPENING)
        conversations[conv_id] = conv_state

        # Determine template
        trigger_kind = trg.get("kind", "generic")
        merchant_name = merchant.get("identity", {}).get("name", "")

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger_kind}_v1",
            "template_params": [
                merchant_name,
                trigger_kind.replace("_", " ").title(),
                result["body"][:40],
            ],
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", sup_key),
            "rationale": result.get("rationale", ""),
        })

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    conv_state = conversations.get(conv_id)

    # Build minimal merchant/category if we don't have state (shouldn't happen, but safe)
    merchant_id = body.merchant_id
    merchant = _get_payload("merchant", merchant_id) if merchant_id else {}
    cat_slug = (merchant or {}).get("category_slug", "")
    category = _get_payload("category", cat_slug) or {}
    customer_id = body.customer_id
    customer = _get_payload("customer", customer_id) if customer_id else None
    lang = _merchant_lang(merchant or {})

    merchant_name = (merchant or {}).get("identity", {}).get("name", "")

    # Record the incoming message in state
    if conv_state:
        conv_state.add_human_turn(body.message, body.from_role)
    else:
        # Create minimal state
        conv_state = ConversationState(
            conversation_id=conv_id,
            merchant_id=merchant_id or "",
            customer_id=customer_id,
            trigger_id=None,
        )
        conv_state.add_human_turn(body.message, body.from_role)
        conversations[conv_id] = conv_state

    # Routing decision (no LLM needed for clear-cut cases)
    decision = decide_on_reply(conv_state, body.message)

    if decision["decision"] == "exit":
        conv_state.transition(ConvState.CLOSED)
        exit_msg = graceful_exit_message(lang)
        if conv_state.turns:
            conv_state.add_bot_turn(exit_msg.get("body", ""), "none")
        return exit_msg

    if decision["decision"] == "close_after_retry":
        conv_state.transition(ConvState.CLOSED)
        exit_msg = graceful_exit_message(lang)
        return exit_msg

    if decision["decision"] == "auto_reply_retry":
        retry_msg = auto_reply_retry_message(merchant_name or "there", lang)
        conv_state.add_bot_turn(retry_msg["body"], retry_msg.get("cta", "open_ended"))
        return retry_msg

    # Normal or intent → call LLM
    trg_payload = None
    if conv_state.trigger_id:
        trg_entry = contexts.get(("trigger", conv_state.trigger_id))
        if trg_entry:
            trg_payload = trg_entry["payload"]

    try:
        result = compose_reply(
            category=category or {},
            merchant=merchant or {},
            trigger=trg_payload or {},
            conversation_history=conv_state.get_history(),
            merchant_message=body.message,
            customer=customer,
        )
    except Exception as e:
        print(f"[REPLY ERROR] {conv_id}: {e}")
        result = {
            "action": "send",
            "body": "Samajh gaya. Aage kya karna hai batao — main ready hoon.",
            "cta": "open_ended",
            "rationale": "Fallback reply due to error.",
        }

    action = result.get("action", "send")

    if action == "end":
        conv_state.transition(ConvState.CLOSED)
    elif action == "send":
        conv_state.transition(ConvState.ENGAGED)
        conv_state.add_bot_turn(result.get("body", ""), result.get("cta", "open_ended"))

    # Build response
    response: dict[str, Any] = {
        "action": action,
        "rationale": result.get("rationale", ""),
    }
    if action == "send":
        response["body"] = result.get("body", "")
        response["cta"] = result.get("cta", "open_ended")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 900)
    elif action == "end":
        if result.get("body"):
            response["body"] = result["body"]

    return response


# ── Optional teardown ─────────────────────────────────────────────────────────

@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression.clear()
    return {"status": "wiped"}


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=8080, reload=False)
