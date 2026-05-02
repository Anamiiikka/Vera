# Vera Bot — magicpin AI Challenge Submission

## Approach

**Architecture**: FastAPI HTTP server + Groq LLaMA-3.3-70B composer + multi-turn state machine.

### Composition Strategy

Every `compose()` call goes through three layers:

1. **Trigger routing**: Each trigger `kind` (e.g., `research_digest`, `perf_dip`, `recall_due`) gets a specialized instruction block prepended to the prompt. This ensures the message anchors on the trigger's specific value rather than being generic.

2. **Context slimming**: We inject only the most relevant fields from each context (not the entire JSON) to keep token count low and focus the LLM's attention on actionable signals — merchant CTR, active offers, conversation history, peer stats, category voice.

3. **Output validation**: Post-process enforces correct CTA enum (`yes_stop`/`open_ended`/`none`), strips markdown fences, and falls back gracefully if JSON parsing fails.

### Compulsion Levers Used (per message)

We target at least **2 of 8** levers per message:
- **Specificity**: Always cite a real number from context (CTR, review count, patient cohort size, trial N).
- **Loss aversion**: For dip/competitor/renewal triggers.
- **Social proof**: Peer stats from category context.
- **Effort externalization**: "I've drafted X — just say YES."
- **Curiosity**: Dormant/curious-ask triggers.
- **Single binary CTA**: YES/STOP for action triggers; open-ended for information triggers.

### Multi-turn Handling

A `ConversationState` object tracks each conversation. On each `/v1/reply`:
- **Auto-reply detection**: canned-phrase matching + exact-text repetition counter → graceful exit after 2 attempts.
- **Intent detection**: regex patterns for "yes/haan/ok/go ahead" → switch to action mode immediately (no more qualifying questions).
- **Exit detection**: "stop/not interested/nahi chahiye" → polite `action: end`.
- **Normal replies**: forwarded to LLM with full conversation history (last 6 turns).

### Language

Match `identity.languages` — Hindi-English code-mix preferred when `hi` in languages list.

### Model

`llama-3.3-70b-versatile` on Groq. Temperature=0 for determinism. ~1.5–2.5s per call — comfortably within the 30s judge timeout.

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
