# Vera-beating bot — submission README

**Team**: Vanshika (solo)
**Model**: pluggable, temperature=0. Set ONE of these env vars:
- `GEMINI_API_KEY` — Google Gemini (`gemini-2.0-flash` by default). Genuinely
  free tier, no card required — get a key at aistudio.google.com/apikey.
- `GROQ_API_KEY` — Groq (`llama-3.3-70b-versatile` by default). Also free,
  no card required — console.groq.com.
- `ANTHROPIC_API_KEY` — Claude (`claude-sonnet-4-6` by default). No free
  tier, needs prepaid credit.

If none is set (or a call fails), the bot falls back to a deterministic
template composer — the server never crashes or times out, it just writes
plainer copy. Override the model name with `LLM_MODEL` if you want a
different model from the same provider.

## Files

| File | Purpose |
|---|---|
| `bot.py` | FastAPI server — the 5 required endpoints |
| `composer.py` | The actual "brain" — `compose()` (first-touch) and `decide_reply()` (multi-turn) |
| `conversation_handlers.py` | Optional `respond(state, merchant_message)` wrapper per §7.4 |
| `generate_submission.py` | Produces `submission.jsonl` from the 30 canonical test pairs |
| `submission.jsonl` | 30 composed messages, one per canonical test pair |
| `requirements.txt` | `fastapi`, `uvicorn`, `anthropic`, `pydantic` |
| `Procfile` | For Render/Railway/Heroku-style buildpacks |

## Approach

One LLM call per message, with the 4-context framework serialized straight
into the prompt (category, merchant, trigger, customer), plus a strict
system prompt encoding every rule from the brief: specificity-first,
category-matched voice, single CTA in the last sentence, don't fabricate,
prefer social-proof / ask-the-merchant levers (called out as production
Vera's biggest miss), never repeat a body verbatim.

Multi-turn control flow (`/v1/reply`) is **rule-based, not LLM-based**,
because these are the moments that need to be reliable every time, not just
"probably right":

- **Auto-reply detection** — every inbound message is normalized and
  counted per conversation. A second occurrence of the same text (or of a
  known canned-reply phrase like "our team will respond shortly") triggers
  one direct probe; a third occurrence ends the conversation instead of
  burning more turns. This is checked *before* anything is sent to the LLM.
- **Intent transition** — a regex bank of English + Hindi commitment
  phrases ("let's do it", "kar do", "go ahead", "sign me up") forces
  `action_mode`. The LLM is asked to write in that mode; if the response
  still looks like a qualifying question, a safe hard-coded fallback line
  is substituted so the conversation never regresses to pitch mode.
- **STOP / opt-out** — always wins, checked first, always ends the
  conversation regardless of anything else in flight.
- **"Not now"** — a small phrase bank ("later", "thodi der", "busy") maps
  to `action: wait` instead of pushing again.

The LLM is only used to *write the prose* once the control-flow decision is
already made — this keeps the parts the judge's replay tests specifically
probe (auto-reply hell, intent transition, hostile handling) deterministic
and testable, rather than hoping the model gets it right zero-shot every
time.

## Tradeoffs

- **No retrieval / embeddings.** With only 5 categories and small digest
  lists, the full digest is just serialized into the prompt rather than
  retrieved — simpler, and nothing in the dataset is large enough to need
  chunking. Would revisit if digest lists were long.
- **In-memory state only.** Fine for a single 60-minute test window; would
  swap for Redis/SQLite for a real multi-day deployment (the brief notes
  the harness doesn't restart the bot mid-test, so this is a fair scope cut).
- **Rules over prompting for control flow.** Slightly less "elegant" than
  asking the LLM to decide send/wait/end itself, but far more reliable for
  exactly the scenarios (§Phase 4 replay) that carry outsized scoring weight
  and that a single bad zero-shot judgment call would tank.
- **Fallback quality ceiling.** The template fallback path (no LLM) is
  intentionally plain — it exists purely so the bot never 500s or times
  out, not to score well on its own. Submission quality depends on the LLM
  path actually running, which needs `ANTHROPIC_API_KEY` set wherever this
  is deployed.

## What additional context would have helped most

- A real example of a *good* multi-turn conversation reaching a booked
  outcome (all the brief's excerpts stop at 2-3 turns) — would sharpen the
  "what's the next best step after acceptance" logic.
- Category-specific taboo/compliance detail beyond dentists (the brief
  fully spells out dentist compliance risk but is thinner on, say,
  pharmacy regulatory language).
- Confirmation of whether `available_slots`/offer data can be trusted as
  exhaustive, or whether the merchant might have more offers not reflected
  in the pushed context (affects how confidently the bot can say "your only
  active offer is X").
