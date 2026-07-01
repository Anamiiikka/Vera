"""
llm.py — Pluggable LLM provider for Vera.

The composer talks to the model through a single function, `complete_json`,
so the underlying provider can be swapped with env vars — no code change.
Groq stays the DEFAULT, so behavior is unchanged unless you opt in.

Configure via environment:

  LLM_PROVIDER        groq | openai | anthropic | gemini      (default: groq)
  LLM_MODEL           override the primary model               (optional)
  LLM_FALLBACK_MODEL  model to try if the primary rate-limits  (optional)
  LLM_TEMPERATURE     sampling temperature                     (default: 0.0)

  # API keys (only the one for your chosen provider is needed)
  GROQ_API_KEY
  OPENAI_API_KEY      (+ optional OPENAI_BASE_URL for OpenAI-compatible hosts)
  ANTHROPIC_API_KEY
  GEMINI_API_KEY  or  GOOGLE_API_KEY

Every provider is instructed to return a single JSON object; the caller
(composer._extract_json) is tolerant of stray prose, so we optimize for
determinism (temperature 0) over strict formatting.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

# Per-provider (primary, fallback) model defaults. Override with LLM_MODEL /
# LLM_FALLBACK_MODEL. Frontier defaults follow "latest and most capable".
_DEFAULT_MODELS: dict[str, tuple[str, str | None]] = {
    "groq": ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
    "openai": ("gpt-4o", "gpt-4o-mini"),
    "anthropic": ("claude-sonnet-5", "claude-haiku-4-5-20251001"),
    "gemini": ("gemini-2.0-flash", None),
}

_RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "429", "quota", "overloaded", "529")

# Cached SDK clients, keyed by provider.
_clients: dict[str, object] = {}


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "groq").strip().lower()


def _temperature() -> float:
    try:
        return float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    except ValueError:
        return 0.0


def _models(provider: str) -> list[str]:
    primary_def, fallback_def = _DEFAULT_MODELS.get(provider, (None, None))
    primary = os.environ.get("LLM_MODEL", primary_def or "")
    fallback = os.environ.get("LLM_FALLBACK_MODEL", fallback_def or "")
    models = [m for m in (primary, fallback) if m]
    if not models:
        raise RuntimeError(
            f"No model configured for provider '{provider}'. Set LLM_MODEL."
        )
    return models


def _is_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return any(m in s for m in _RATE_LIMIT_MARKERS)


# ── Provider implementations (lazy imports so unused SDKs need not be installed) ─

def _call_groq(system: str, user: str, model: str, temp: float, max_tokens: int) -> str:
    from groq import Groq

    if "groq" not in _clients:
        _clients["groq"] = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    client = _clients["groq"]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temp,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def _call_openai(system: str, user: str, model: str, temp: float, max_tokens: int) -> str:
    from openai import OpenAI

    if "openai" not in _clients:
        kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "")}
        base = os.environ.get("OPENAI_BASE_URL")
        if base:
            kwargs["base_url"] = base
        _clients["openai"] = OpenAI(**kwargs)
    client = _clients["openai"]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temp,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(system: str, user: str, model: str, temp: float, max_tokens: int) -> str:
    from anthropic import Anthropic

    if "anthropic" not in _clients:
        _clients["anthropic"] = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    client = _clients["anthropic"]
    # Prefill the assistant turn with "{" to force a JSON object response.
    resp = client.messages.create(
        model=model,
        system=system,
        max_tokens=max_tokens,
        temperature=temp,
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": "{"},
        ],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    return "{" + text


def _call_gemini(system: str, user: str, model: str, temp: float, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    if "gemini" not in _clients:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        _clients["gemini"] = genai.Client(api_key=key)
    client = _clients["gemini"]
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temp,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    return resp.text or ""


_DISPATCH: dict[str, Callable[[str, str, str, float, int], str]] = {
    "groq": _call_groq,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
}


def complete_json(system: str, user: str, max_tokens: int = 600) -> str:
    """Send one system+user prompt and return the raw model text (expected JSON).

    Tries the primary model, then the fallback model on rate-limit errors.
    Raises the last exception if every model fails.
    """
    provider = _provider()
    call = _DISPATCH.get(provider)
    if call is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(_DISPATCH)}."
        )

    temp = _temperature()
    last_exc: Exception | None = None
    for model in _models(provider):
        try:
            return call(system, user, model, temp, max_tokens)
        except Exception as e:  # noqa: BLE001 — provider SDKs raise varied types
            last_exc = e
            if _is_rate_limit(e):
                print(f"[LLM] {provider}:{model} rate-limited, trying next model...")
                time.sleep(1)
                continue
            raise
    raise last_exc or RuntimeError(f"All {provider} models failed")


def active_model_label() -> str:
    """Human-readable 'provider:model' for /v1/metadata."""
    provider = _provider()
    try:
        primary = _models(provider)[0]
    except Exception:
        primary = "unconfigured"
    return f"{provider}:{primary}"
