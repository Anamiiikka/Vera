"""
conversation_handlers.py — Multi-turn conversation state machine for Vera

Handles:
- Auto-reply detection
- Intent transition (qualifying → action mode)
- Graceful exit signals
- Conversation state tracking
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConvState(str, Enum):
    OPENING = "opening"
    ENGAGED = "engaged"
    ACTION_MODE = "action_mode"
    WAITING = "waiting"
    AUTO_REPLY_DETECTED = "auto_reply_detected"
    CLOSED = "closed"


# ── Auto-reply detection ──────────────────────────────────────────────────────

# Canned phrases that indicate a WhatsApp Business auto-reply
AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "aapki jaankari ke liye",
    "i am an automated",
    "main ek automated",
    "bahut-bahut shukriya",
    "hamari team tak pahuncha",
    "we will get back to you",
    "hum aapko jald",
    "our team will reach",
    "this is an automated message",
    "you have reached an automated",
    "auto reply",
    "auto-reply",
    "out of office",
]

# Exit/disinterest signals
EXIT_PHRASES = [
    "not interested",
    "no thanks",
    "stop",
    "unsubscribe",
    "nahi chahiye",
    "nahi chahte",
    "busy hoon",
    "abhi nahi",
    "mat bhejo",
    "block",
    "do not contact",
    "don't contact",
]

# Strong intent signals → switch to action mode
INTENT_PHRASES = [
    r"\byes\b",
    r"\byep\b",
    r"\byeah\b",
    r"\bsure\b",
    r"\bhaan\b",
    r"\bhaanji\b",
    r"\bhan\b",
    r"\bha\b",
    r"\bji\b",
    r"\btheek hai\b",
    r"\bthik hai\b",
    r"\bchalo\b",
    r"\bchalega\b",
    r"\bkaro\b",
    r"\bkar do\b",
    r"\bkar dijiye\b",
    r"\bgo ahead\b",
    r"\blet'?s do\b",
    r"\blets do\b",
    r"\bproceed\b",
    r"\bstart\b",
    r"\bsend\b",
    r"\bsend it\b",
    r"\bbhejo\b",
    r"\bbhej do\b",
    r"\bkijiye\b",
    r"\bi want\b",
    r"\bi'?m interested\b",
    r"\binterested\b",
    r"\bjoin\b",
    r"\bsign me up\b",
    r"\bdo it\b",
    r"\bok\b",
    r"\bokay\b",
    r"\bplease\b",
    r"\bpls\b",
    r"\bset it up\b",
    r"\bcall me\b",
]

# Short affirmatives that must never be misread as an auto-reply or exit.
SHORT_AFFIRMATIVES = {
    "yes", "ok", "okay", "k", "haan", "haanji", "han", "ha", "ji", "hn",
    "yep", "yeah", "sure", "done", "great", "cool", "thanks", "thank you",
    "go", "go ahead", "do it", "send", "send it", "start", "proceed",
    "theek hai", "thik hai", "chalo", "chalega", "kar do", "bhejo",
}


def is_auto_reply(message: str) -> bool:
    """Return True if the message looks like a WhatsApp Business canned auto-reply.

    Conservative on purpose: a false positive here silently kills a live
    conversation, so we only flag messages that match a known canned phrase.
    Verbatim-repeat detection (in decide_on_reply) catches the rest — a genuine
    auto-reply repeats the same text, a real merchant does not.
    """
    msg_lower = message.lower().strip()
    if msg_lower in SHORT_AFFIRMATIVES:
        return False
    for phrase in AUTO_REPLY_PHRASES:
        if phrase in msg_lower:
            return True
    return False


def is_exit_signal(message: str) -> bool:
    """Return True if merchant/customer wants to disengage."""
    msg_lower = message.lower().strip()
    if msg_lower in SHORT_AFFIRMATIVES:
        return False
    return any(phrase in msg_lower for phrase in EXIT_PHRASES)


def is_intent_signal(message: str) -> bool:
    """Return True if merchant has given a clear go-ahead."""
    msg_lower = message.lower().strip()
    if msg_lower in SHORT_AFFIRMATIVES:
        return True
    return any(re.search(pattern, msg_lower) for pattern in INTENT_PHRASES)


# ── Conversation State Object ─────────────────────────────────────────────────

@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    trigger_id: Optional[str]
    state: ConvState = ConvState.OPENING
    turns: list[dict] = field(default_factory=list)
    auto_reply_count: int = 0
    last_bot_body: str = ""
    send_as: str = "vera"

    def add_bot_turn(self, body: str, cta: str = "open_ended") -> None:
        self.turns.append({"from": "vera", "body": body, "cta": cta})
        self.last_bot_body = body

    def add_human_turn(self, message: str, from_role: str = "merchant") -> None:
        self.turns.append({"from": from_role, "message": message})

    def transition(self, new_state: ConvState) -> None:
        self.state = new_state

    def is_closed(self) -> bool:
        return self.state == ConvState.CLOSED

    def get_history(self) -> list[dict]:
        return self.turns


# ── Reply decision logic ──────────────────────────────────────────────────────

def decide_on_reply(
    state: ConversationState,
    merchant_message: str,
) -> dict:
    """
    Without calling the LLM, determine if we can make a routing decision.

    Returns:
        {"decision": "auto_reply_retry" | "exit" | "intent" | "normal" | "close_after_retry"}

    Order matters: a clear go-ahead ("ok let's do it") must win over the
    auto-reply heuristic, otherwise we ask another qualifying question — the
    exact intent-handoff failure the challenge penalizes.
    """
    # 1) Explicit disengagement always wins.
    if is_exit_signal(merchant_message):
        return {"decision": "exit"}

    # 2) A verbatim repeat of what the human already said = canned auto-reply,
    #    regardless of content. Look back over prior human turns, not just [-2].
    prior_human = [
        t.get("message", "").strip()
        for t in state.turns[:-1]
        if t.get("from") not in ("vera", "merchant_on_behalf")
    ]
    is_repeat = merchant_message.strip() != "" and merchant_message.strip() in prior_human

    # 3) Clear intent → action mode (checked BEFORE the auto-reply heuristic so
    #    short affirmatives like "go ahead" / "haan bhejo" route correctly).
    if not is_repeat and is_intent_signal(merchant_message) and state.state in (
        ConvState.OPENING,
        ConvState.ENGAGED,
        ConvState.WAITING,
    ):
        state.transition(ConvState.ACTION_MODE)
        return {"decision": "intent"}

    # 4) Canned auto-reply (known phrase or verbatim repeat).
    if is_repeat or is_auto_reply(merchant_message):
        state.auto_reply_count += 1
        if state.auto_reply_count >= 2 or is_repeat:
            return {"decision": "close_after_retry"}
        return {"decision": "auto_reply_retry"}

    # 5) Anything else → let the LLM handle it with full context.
    return {"decision": "normal"}


# ── Canned responses for routing decisions ────────────────────────────────────

def auto_reply_retry_message(merchant_name: str, lang: str = "en") -> dict:
    """One more attempt after detecting auto-reply."""
    if "hi" in lang:
        body = f"{merchant_name.split()[0]}, lagta hai aap abhi busy hain. Koi baat nahi — jab time mile tab baat karte hain. Ek cheez batao: aapke is hafte ka sabse popular service kaunsa raha? 😊"
    else:
        body = f"Looks like you might be away, {merchant_name.split()[0]}. No rush — just curious: what's been your most popular service this week?"
    return {
        "action": "send",
        "body": body,
        "cta": "open_ended",
        "rationale": "Auto-reply detected; one soft retry with curiosity hook before graceful exit.",
    }


def graceful_exit_message(lang: str = "en") -> dict:
    """Polite exit after auto-reply loop or exit signal."""
    if "hi" in lang:
        body = "Koi baat nahi, samajh gayi. Jab bhi zarurat ho, main yahaan hoon. Best wishes! 🙂"
    else:
        body = "Got it — no worries at all. Whenever you need anything, I'm here. Take care! 🙂"
    return {
        "action": "end",
        "body": body,
        "rationale": "Graceful exit after auto-reply loop / exit signal detected.",
    }


def intent_ack_message(merchant_name: str, trigger_kind: str, lang: str = "en") -> str:
    """Quick action-mode acknowledgment before composing next step."""
    name_short = merchant_name.split()[0]
    if "hi" in lang:
        return f"Perfect, {name_short}! Main abhi set kar deti hoon —"
    return f"Great, {name_short}! Let me set that up —"
