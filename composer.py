"""
composer.py — Vera message composer using Groq (LLaMA 3.3-70B)

compose(category, merchant, trigger, customer?) → {body, cta, send_as, suppression_key, rationale}
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client: Groq | None = None

# Model cascade: try fast 70B first; if rate-limited, fall back to 8B instant
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        _client = Groq(api_key=api_key)
    return _client


# ── Trigger-type routing: short instruction appended per kind ─────────────────
TRIGGER_INSTRUCTIONS: dict[str, str] = {
    "research_digest": (
        "This is a RESEARCH DIGEST trigger. Lead with the specific finding "
        "(trial size, % improvement, source). Frame as 'something relevant to your patients/customers'. "
        "End with a low-friction CTA: offer to pull the abstract or draft a patient-education message."
    ),
    "perf_spike": (
        "This is a PERFORMANCE SPIKE trigger. Merchant's views/calls went UP. "
        "Celebrate briefly, then pivot to 'let's capitalize on this momentum' with one specific action. "
        "Use loss-aversion: 'this window won't last'. CTA: binary YES/STOP."
    ),
    "perf_dip": (
        "This is a PERFORMANCE DIP trigger. Calls or views dropped. "
        "Be empathetic but data-driven. Name the specific metric and delta. "
        "Offer one concrete fix (offer activation, post update, profile update). CTA: binary YES/STOP."
    ),
    "milestone_reached": (
        "This is a MILESTONE trigger. Merchant crossed a meaningful number (reviews, customers, etc). "
        "Celebrate with the exact number. Suggest the next milestone + what action gets them there. "
        "Social proof: 'Top 3 in your locality do X after crossing this mark'."
    ),
    "dormant_with_vera": (
        "Merchant hasn't replied in 14+ days. Use CURIOSITY to re-engage — ask a question about their "
        "business ('What's your most-asked service this week?'). No CTA pressure. Just open a door."
    ),
    "review_theme_emerged": (
        "A review theme emerged (e.g., 'wait time'). Name the theme explicitly. "
        "Offer to help address it (draft a reply, update description). CTA: binary YES/STOP."
    ),
    "competitor_opened": (
        "A new competitor opened nearby. Use loss-aversion gently. "
        "Suggest one differentiator action (activate an offer, add photos, get more reviews). "
        "Don't name the competitor if not in context. CTA: binary YES/STOP."
    ),
    "festival_upcoming": (
        "A festival/seasonal event is coming. Use the specific festival name and date. "
        "Offer to draft a festival-themed post or campaign. "
        "Keep tone celebratory but peer-level, not promotional hype. CTA: binary YES/STOP."
    ),
    "recall_due": (
        "A CUSTOMER RECALL is due. Message is FROM THE MERCHANT to the customer. "
        "Use customer's name, the specific service due (from relationship history), and offer real slots if available. "
        "send_as = merchant_on_behalf. Language must match customer's language_pref."
    ),
    "customer_lapsed_soft": (
        "A customer hasn't visited in a while. Draft a warm re-engagement message from the merchant. "
        "Name the last service and how long ago. Offer an incentive from the merchant's active offers. "
        "send_as = merchant_on_behalf."
    ),
    "appointment_tomorrow": (
        "Customer has an appointment tomorrow. Send a friendly reminder from the merchant. "
        "Include appointment time, what to expect, any prep needed. "
        "send_as = merchant_on_behalf."
    ),
    "renewal_due": (
        "Merchant's subscription is expiring soon. Remind them of days remaining. "
        "Highlight what they'd lose (visibility, leads, Vera support). "
        "Effort-externalize: 'I can renew in 2 mins — just say YES'. CTA: binary YES/STOP."
    ),
    "curious_ask_due": (
        "Weekly curiosity-ask cadence. Ask the merchant ONE genuine business question "
        "('What offer gets the most calls this month?', 'Any big events planned?'). "
        "No CTA. Just start a conversation."
    ),
    "chronic_refill_due": (
        "For pharmacies: a patient's chronic medication refill is likely due. "
        "Mention the approximate refill window (don't fabricate exact dates). "
        "send_as = merchant_on_behalf."
    ),
    "trial_followup": (
        "Customer tried the merchant for the first time. Follow up with a satisfaction check. "
        "Ask if they'd like to book again, mention a relevant offer. "
        "send_as = merchant_on_behalf."
    ),
}

DEFAULT_TRIGGER_INSTRUCTION = (
    "Compose a relevant, specific, compelling WhatsApp message for this trigger. "
    "Anchor on one concrete fact from the contexts."
)


SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant for merchant growth. You compose short, high-compulsion WhatsApp messages for Indian merchants.

CORE RULES:
1. Use ONLY data from the provided contexts — never fabricate numbers, citations, competitor names, or offers not listed.
2. Match the merchant's language preference (hi-en mix = Hindi-English code-mix, hi = Hindi, en = English).
3. Peer/colleague tone — NOT promotional hype. "Dr. Meera" not "AMAZING DEAL!!".
4. One primary CTA per message: either binary YES/STOP, open_ended question, or none (for pure info).
5. Lead with the hook — no preambles like "I hope you're doing well" or re-introductions after first message.
6. Keep messages concise (3-6 sentences) unless the trigger demands detail.
7. Always include at least ONE compulsion lever: specificity (real numbers), loss-aversion, social proof, curiosity, effort-externalization, or single-binary-commit.
8. For customer-facing messages (recall, lapsed, appointment): send_as = "merchant_on_behalf".
9. For merchant-facing messages: send_as = "vera".
10. Do NOT use service+percentage discounts ("20% off"). Use service+price ("Dental Cleaning @ ₹299").

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{
  "body": "the WhatsApp message text",
  "cta": "yes_stop" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "copied from trigger or auto-generated",
  "rationale": "1-2 sentence explanation of choices made"
}"""


def _build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
    conversation_history: list[dict] | None = None,
) -> str:
    trigger_kind = trigger.get("kind", "")
    trigger_instr = TRIGGER_INSTRUCTIONS.get(trigger_kind, DEFAULT_TRIGGER_INSTRUCTION)

    parts = [
        f"TRIGGER INSTRUCTION: {trigger_instr}",
        "",
        "CATEGORY CONTEXT:",
        json.dumps(_slim_category(category), ensure_ascii=False, indent=2),
        "",
        "MERCHANT CONTEXT:",
        json.dumps(_slim_merchant(merchant), ensure_ascii=False, indent=2),
        "",
        "TRIGGER CONTEXT:",
        json.dumps(trigger, ensure_ascii=False, indent=2),
    ]

    if customer:
        parts += ["", "CUSTOMER CONTEXT:", json.dumps(customer, ensure_ascii=False, indent=2)]

    if conversation_history:
        parts += ["", "CONVERSATION HISTORY (last 5 turns):"]
        for turn in conversation_history[-5:]:
            role = turn.get("from", turn.get("from_role", "?"))
            msg = turn.get("body", turn.get("msg", turn.get("message", "")))
            parts.append(f"  [{role.upper()}]: {msg}")

    parts += ["", "Now compose the message. Respond ONLY with the JSON object."]
    return "\n".join(parts)


def _slim_category(cat: dict) -> dict:
    """Return a trimmed category dict — only the fields most useful for composition."""
    return {
        "slug": cat.get("slug"),
        "voice": cat.get("voice"),
        "offer_catalog": cat.get("offer_catalog", [])[:5],
        "peer_stats": cat.get("peer_stats"),
        "digest": cat.get("digest", [])[:3],
        "seasonal_beats": cat.get("seasonal_beats", [])[:3],
        "trend_signals": cat.get("trend_signals", [])[:2],
        "patient_content_library": cat.get("patient_content_library", [])[:2],
    }


def _slim_merchant(m: dict) -> dict:
    """Return a trimmed merchant dict — only fields useful for composition."""
    return {
        "merchant_id": m.get("merchant_id"),
        "category_slug": m.get("category_slug"),
        "identity": m.get("identity"),
        "subscription": m.get("subscription"),
        "performance": m.get("performance"),
        "offers": m.get("offers", [])[:5],
        "conversation_history": m.get("conversation_history", [])[-3:],
        "customer_aggregate": m.get("customer_aggregate"),
        "signals": m.get("signals", []),
        "review_themes": m.get("review_themes", []),
    }


def _parse_llm_output(raw: str, trigger: dict) -> dict:
    """Extract JSON from LLM response, with fallback."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find first { ... }
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            # Ensure required keys exist
            result.setdefault("suppression_key", trigger.get("suppression_key", f"auto_{trigger.get('id', 'unknown')}"))
            result.setdefault("rationale", "Composed from provided contexts.")
            result.setdefault("cta", "open_ended")
            result.setdefault("send_as", "vera")
            # Validate cta values
            if result["cta"] not in ("yes_stop", "open_ended", "none"):
                result["cta"] = "open_ended"
            return result
        except json.JSONDecodeError:
            pass
    # Fallback — return a safe empty response
    return {
        "body": "Ek minute — kuch important share karna tha. Kya aap abhi baat kar sakte hain?",
        "cta": "yes_stop",
        "send_as": "vera",
        "suppression_key": trigger.get("suppression_key", "fallback"),
        "rationale": "Fallback message — LLM output could not be parsed.",
    }


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Main composition function.

    Inputs are dicts loaded from the dataset JSON.
    Returns: {body, cta, send_as, suppression_key, rationale}
    """
    client = _get_client()
    user_prompt = _build_user_prompt(category, merchant, trigger, customer, conversation_history)

    last_exc = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return _parse_llm_output(raw, trigger)
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str or "429" in err_str:
                print(f"[WARN] {model} rate-limited, trying next model...")
                last_exc = e
                time.sleep(1)
                continue
            raise
    # All models exhausted
    raise last_exc or RuntimeError("All models rate-limited")


def compose_reply(
    category: dict,
    merchant: dict,
    trigger: dict | None,
    conversation_history: list[dict],
    merchant_message: str,
    customer: dict | None = None,
) -> dict:
    """
    Compose a reply to a merchant's message within an ongoing conversation.
    Returns same schema as compose().
    """
    client = _get_client()

    # Build a reply-specific prompt
    history_text = "\n".join(
        f"  [{t.get('from', t.get('from_role', '?')).upper()}]: {t.get('body', t.get('msg', t.get('message', '')))}"
        for t in conversation_history[-6:]
    )

    user_prompt = f"""You are mid-conversation with a merchant. The merchant just replied.

MERCHANT'S LATEST MESSAGE: "{merchant_message}"

CONVERSATION SO FAR:
{history_text}

MERCHANT CONTEXT:
{json.dumps(_slim_merchant(merchant), ensure_ascii=False, indent=2)}

CATEGORY CONTEXT (voice only):
{json.dumps({'slug': category.get('slug'), 'voice': category.get('voice')}, ensure_ascii=False)}

REPLY RULES:
- If merchant said YES/agreed/go ahead: move to ACTION MODE — confirm what you'll do next, be specific.
- If merchant asked a question: answer from context data, then offer the next step.
- If merchant seems uninterested/said stop: reply with action "end".
- If this looks like a canned auto-reply (exact repeat or generic "thank you for contacting"): reply with action "end" after one retry.
- Keep reply short (2-3 sentences max).
- No re-introductions.

Respond ONLY with JSON: {{"action": "send"|"wait"|"end", "body": "...", "cta": "yes_stop"|"open_ended"|"none", "rationale": "..."}}
If action is "wait", add "wait_seconds": <integer>.
If action is "end", body is optional (brief polite close).
"""

    last_exc = None
    response = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            break
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str or "429" in err_str:
                last_exc = e
                time.sleep(1)
                continue
            raise
    if response is None:
        raise last_exc or RuntimeError("All models rate-limited")

    raw = response.choices[0].message.content or ""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            result.setdefault("action", "send")
            result.setdefault("cta", "open_ended")
            if result.get("action") == "wait":
                result.setdefault("wait_seconds", 900)
            return result
        except json.JSONDecodeError:
            pass

    return {"action": "send", "body": "Samajh gaya. Aage kya karna hai batao — main ready hoon.", "cta": "open_ended", "rationale": "Fallback reply"}
