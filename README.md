# Vera Bot — magicpin AI Challenge Submission

## Approach

**Architecture**: FastAPI HTTP server + Groq LLaMA-3.3-70B composer + multi-turn state machine.

### Composition Strategy

Every `compose()` call goes through four layers:

1. **Trigger routing**: Each trigger `kind` (e.g., `research_digest`, `perf_dip`, `recall_due`) gets a specialized instruction block prepended to the prompt. This ensures the message anchors on the trigger's specific value rather than being generic.

2. **Reference resolution**: Triggers point at facts by id (`payload.top_item_id`, offer ids, content ids) rather than embedding them. We resolve those ids against the category/merchant contexts and surface the exact object as a `PRIMARY FACT TO ANCHOR ON` block — and force-include it in the slimmed context so truncation can never drop the fact the message must cite. This keeps Specificity + Trigger-relevance high, including on the post-submission digest injections.

3. **Context slimming**: We inject only the most relevant fields from each context (not the entire JSON) to keep token count low and focus the LLM's attention on actionable signals — merchant CTR, active offers, conversation history, peer stats, category voice.

4. **Output validation + graceful degradation**: Post-process enforces the correct CTA enum (`yes_stop`/`open_ended`/`none`), extracts JSON via a balanced-brace scan (tolerant of trailing prose), and on total LLM failure emits a **context-aware** fallback that still names the merchant and the trigger reason — never a generic canned line.

### Compulsion Levers Used (per message)

We target at least **2 of 8** levers per message:
- **Specificity**: Always cite a real number from context (CTR, review count, patient cohort size, trial N).
- **Loss aversion**: For dip/competitor/renewal triggers.
- **Social proof**: Peer stats from category context.
- **Effort externalization**: "I've drafted X — just say YES."
- **Curiosity**: Dormant/curious-ask triggers.
- **Single binary CTA**: YES/STOP for action triggers; open-ended for information triggers.

### Multi-turn Handling

A `ConversationState` object tracks each conversation. On each `/v1/reply` the router checks, **in this order**:
- **Exit detection**: "stop/not interested/nahi chahiye" → polite `action: end`.
- **Intent detection** (before auto-reply): "yes/haan/ok/go ahead/haan bhejo" → switch to action mode immediately. Intent is checked *before* the auto-reply heuristic, so short affirmatives are never misread as canned replies (the classic intent-handoff failure).
- **Auto-reply detection**: known canned phrases **or** a verbatim repeat of the merchant's own earlier text → one soft retry, then graceful exit. Deliberately conservative — a false positive here silently kills a live conversation.
- **Normal replies**: forwarded to the LLM with full conversation history (last 6 turns) and an explicit "never repeat a message verbatim / stay on-mission" instruction.

An **anti-repetition guard** on every outbound reply ensures the bot never sends the same body twice in a conversation (the judge penalizes verbatim repeats).

### Language

Match `identity.languages` — Hindi-English code-mix preferred when `hi` in languages list.

### Model

`llama-3.3-70b-versatile` on Groq, with `llama-3.1-8b-instant` as a rate-limit fallback. Temperature=0 for determinism. `/v1/tick` composes candidates **concurrently** (bounded worker pool, urgency-ordered) inside a 25s budget so a busy tick never blows the 30s judge timeout.

---

## Tradeoffs

| Decision | Why |
|---|---|
| Groq + LLaMA over GPT-4o | 10–15× faster API calls; free tier sufficient; determinism at temp=0 |
| In-memory context store | Simplest; avoids Redis setup; fine for 60-min test window |
| Single-prompt vs. multi-step | Single prompt is faster and stays within timeout; a retrieval step would add 2–3s |
| Slim context injection | Avoids exceeding context window; focuses LLM on actionable fields |

---

## What Additional Context Would Have Helped Most

1. **Merchant's recent customer reviews** (full text) — the `review_themes` field in merchant context lists topics but not the actual text. With full reviews we could quote exact customer language back to the merchant ("3 customers mentioned 'gentle touch' this week — that's your differentiator").

2. **Historical message open/reply rates per trigger kind** — knowing which trigger types get the best real-world response rates would let us prioritize the tick's action queue more precisely.

3. **Merchant's actual WhatsApp chat history** (not just Vera turns) — understanding the merchant's communication style (how they talk to customers) would let us better match their voice when composing `merchant_on_behalf` messages.

---

## Files

| File | Purpose |
|---|---|
| `bot.py` | FastAPI server — 5 judge endpoints |
| `composer.py` | Groq LLM composition logic |
| `conversation_handlers.py` | Multi-turn state machine |
| `generate_submission.py` | Script to generate submission.jsonl |
| `submission.jsonl` | 30 pre-composed test outputs |
| `requirements.txt` | Python dependencies |
