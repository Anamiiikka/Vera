"""
composer.py — Vera message composer.

compose(category, merchant, trigger, customer?) → {body, cta, send_as, suppression_key, rationale}

The LLM provider is pluggable via env vars (see llm.py). Groq is the default.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm import complete_json


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


def _iter_catalog_items(category: dict):
    """Yield (bucket_name, item_dict) for every id-bearing item in a category."""
    for bucket in (
        "digest",
        "patient_content_library",
        "offer_catalog",
        "seasonal_beats",
        "trend_signals",
    ):
        for item in category.get(bucket, []) or []:
            if isinstance(item, dict):
                yield bucket, item


def _resolve_trigger_refs(category: dict, merchant: dict, trigger: dict) -> dict:
    """Resolve the IDs a trigger references into their full objects.

    Triggers point at facts by id (payload.top_item_id, offer ids, content ids)
    rather than embedding them. Without resolution the composer can't cite the
    specific fact — and slimming may truncate it away entirely. This returns the
    exact objects the message must anchor on.
    """
    payload = trigger.get("payload", {}) or {}

    # Gather every id-looking value from the payload (scalars + lists).
    ref_ids: set[str] = set()
    for key, val in payload.items():
        if not ("id" in key.lower() or key in ("top_item", "item", "content", "offer")):
            continue
        if isinstance(val, str):
            ref_ids.add(val)
        elif isinstance(val, list):
            ref_ids.update(v for v in val if isinstance(v, str))

    resolved: dict[str, Any] = {}
    for bucket, item in _iter_catalog_items(category):
        if item.get("id") in ref_ids:
            resolved.setdefault(bucket, []).append(item)

    # Referenced merchant offers (some triggers point at a specific offer).
    for offer in merchant.get("offers", []) or []:
        if isinstance(offer, dict) and offer.get("id") in ref_ids:
            resolved.setdefault("merchant_offer", []).append(offer)

    return resolved


def _build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
    conversation_history: list[dict] | None = None,
) -> str:
    trigger_kind = trigger.get("kind", "")
    trigger_instr = TRIGGER_INSTRUCTIONS.get(trigger_kind, DEFAULT_TRIGGER_INSTRUCTION)

    resolved = _resolve_trigger_refs(category, merchant, trigger)

    parts = [f"TRIGGER INSTRUCTION: {trigger_instr}"]

    if resolved:
        parts += [
            "",
            "PRIMARY FACT TO ANCHOR ON (the trigger references this exact item — "
            "cite its concrete numbers/source; do NOT anchor on anything else):",
            json.dumps(resolved, ensure_ascii=False, indent=2),
        ]

    parts += [
        "",
        "CATEGORY CONTEXT:",
        json.dumps(_slim_category(category, resolved), ensure_ascii=False, indent=2),
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


def _slim_bucket(cat: dict, bucket: str, limit: int, resolved: dict | None) -> list:
    """First `limit` items of a bucket, plus any resolved (trigger-referenced)
    item from that bucket that the truncation would have dropped."""
    items = list(cat.get(bucket, []) or [])
    kept = items[:limit]
    if resolved:
        kept_ids = {i.get("id") for i in kept if isinstance(i, dict)}
        for item in resolved.get(bucket, []):
            if item.get("id") not in kept_ids:
                kept.append(item)
    return kept


def _slim_category(cat: dict, resolved: dict | None = None) -> dict:
    """Return a trimmed category dict — only the fields most useful for composition.

    Any item the trigger explicitly references (`resolved`) is force-included even
    if it falls outside the truncation window, so the anchor fact is never lost.
    """
    return {
        "slug": cat.get("slug"),
        "voice": cat.get("voice"),
        "offer_catalog": _slim_bucket(cat, "offer_catalog", 5, resolved),
        "peer_stats": cat.get("peer_stats"),
        "digest": _slim_bucket(cat, "digest", 3, resolved),
        "seasonal_beats": _slim_bucket(cat, "seasonal_beats", 3, resolved),
        "trend_signals": _slim_bucket(cat, "trend_signals", 2, resolved),
        "patient_content_library": _slim_bucket(cat, "patient_content_library", 2, resolved),
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


def _lang_of(merchant: dict | None, customer: dict | None) -> str:
    """Return 'hi-en' if Hindi is in scope, else 'en'."""
    if customer:
        pref = (customer.get("identity", {}) or {}).get("language_pref", "")
        if "hi" in pref.lower():
            return "hi-en"
    if merchant:
        langs = (merchant.get("identity", {}) or {}).get("languages", ["en"])
        if "hi" in langs:
            return "hi-en"
    return "en"


def _extract_json(raw: str) -> dict | None:
    """Best-effort JSON extraction: strip fences, then scan for the first
    balanced {...} object (tolerant of trailing prose the model may add)."""
    raw = re.sub(r"```(?:json)?", "", raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = raw.find("{", start + 1)
    return None


def _context_aware_fallback(
    trigger: dict,
    merchant: dict | None,
    category: dict | None,
    customer: dict | None,
) -> dict:
    """A last-resort message that still names the merchant and the trigger reason
    instead of emitting a generic canned line (which scores ~0 on every rubric
    dimension). Only used when the LLM is unreachable or unparseable."""
    merchant = merchant or {}
    name = (merchant.get("identity", {}) or {}).get("name", "").split(",")[0].strip()
    first = name.split()[0] if name else "there"
    lang = _lang_of(merchant, customer)
    kind_label = trigger.get("kind", "").replace("_", " ").strip() or "an update"
    send_as = "merchant_on_behalf" if customer else "vera"

    if lang == "hi-en":
        body = (
            f"{first}, ek quick baat — aapke {kind_label} ko lekar kuch relevant tha. "
            "Kya main details bhej doon?"
        )
    else:
        body = (
            f"{first}, quick one — there's something relevant to your {kind_label}. "
            "Want me to share the details?"
        )
    return {
        "body": body,
        "cta": "yes_stop",
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", f"auto_{trigger.get('id', 'unknown')}"),
        "rationale": "Context-aware fallback — LLM output unavailable; kept merchant name + trigger reason.",
    }


def _parse_llm_output(
    raw: str,
    trigger: dict,
    merchant: dict | None = None,
    category: dict | None = None,
    customer: dict | None = None,
) -> dict:
    """Extract JSON from LLM response, with a context-aware fallback."""
    result = _extract_json(raw)
    if isinstance(result, dict) and result.get("body", "").strip():
        result.setdefault(
            "suppression_key",
            trigger.get("suppression_key", f"auto_{trigger.get('id', 'unknown')}"),
        )
        result.setdefault("rationale", "Composed from provided contexts.")
        result.setdefault("cta", "open_ended")
        result.setdefault("send_as", "merchant_on_behalf" if customer else "vera")
        if result["cta"] not in ("yes_stop", "open_ended", "none"):
            result["cta"] = "open_ended"
        return result
    return _context_aware_fallback(trigger, merchant, category, customer)


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
    user_prompt = _build_user_prompt(category, merchant, trigger, customer, conversation_history)
    try:
        raw = complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=600)
        return _parse_llm_output(raw, trigger, merchant, category, customer)
    except Exception as e:  # degrade gracefully instead of 500-ing the tick
        print(f"[WARN] compose failed for {trigger.get('id')}: {e}")
        return _context_aware_fallback(trigger, merchant, category, customer)


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
- If the merchant went off-topic or hostile: stay on-mission politely in ONE short line, then re-offer the original next step. Do not take on unrelated tasks (GST filing, etc.).
- Keep reply short (2-3 sentences max).
- No re-introductions.
- NEVER repeat an earlier message verbatim — advance the conversation with a new, specific next step.

Respond ONLY with JSON: {{"action": "send"|"wait"|"end", "body": "...", "cta": "yes_stop"|"open_ended"|"none", "rationale": "..."}}
If action is "wait", add "wait_seconds": <integer>.
If action is "end", body is optional (brief polite close).
"""

    raw = None
    try:
        raw = complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=400)
    except Exception as e:
        print(f"[REPLY WARN] compose_reply failed: {e}")

    lang = _lang_of(merchant, customer)
    if raw is not None:
        result = _extract_json(raw)
        if isinstance(result, dict) and (
            result.get("action") in ("send", "wait", "end") or (result.get("body") or "").strip()
        ):
            action = result.get("action") if result.get("action") in ("send", "wait", "end") else "send"
            result["action"] = action
            result.setdefault("cta", "open_ended")
            if action == "wait":
                result.setdefault("wait_seconds", 900)
            return result

    # Reachability/parse fallback — acknowledge and keep the door open.
    body = (
        "Samajh gaya. Aage kya karna hai batao — main ready hoon."
        if lang == "hi-en"
        else "Got it. Tell me how you'd like to proceed — I'm ready."
    )
    return {"action": "send", "body": body, "cta": "open_ended", "rationale": "Fallback reply — LLM output unavailable."}
