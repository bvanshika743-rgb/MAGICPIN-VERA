"""
composer.py — the "brain" of the Vera-beating bot.

Two public entry points:
    compose(category, merchant, trigger, customer=None) -> dict
        First-touch / proactive message. Used by /v1/tick.

    decide_reply(state, from_role, message) -> dict
        Multi-turn reply given conversation state + the latest inbound
        message. Used by /v1/reply and by conversation_handlers.respond().

Design goals (per challenge-brief.md):
- Specificity: anchor on a concrete, verifiable fact from the contexts.
- Category fit: voice/vocab/offer format matches the vertical.
- Merchant fit: personalized to this merchant's actual numbers/state/history.
- Trigger relevance: message clearly says *why now*.
- Engagement compulsion: uses at least one lever (curiosity, social proof,
  loss aversion, effort externalization, reciprocity, asking-the-merchant,
  single binary CTA).
- Never fabricate data that isn't in the contexts.
- Detect auto-replies and exit gracefully instead of burning turns.
- Detect explicit intent ("let's do it") and switch to action mode instantly.
- Respect STOP / opt-out immediately.
- Never repeat a previous body verbatim.

The LLM call is the primary path. If no API key is configured, or the call
fails/times out, a deterministic template fallback keeps the server alive
and still produces a reasonably on-brief message (this matters because the
judge disqualifies bots that time out or 500).
"""

from __future__ import annotations

import os
import re
import json
import random
import urllib.request
import urllib.error
from typing import Any, Optional

# --------------------------------------------------------------------------
# LLM client. Pluggable across free and paid providers; falls back to
# deterministic templates if none is configured or a call fails.
#
# Pick ONE by setting its env var (checked in this priority order):
#   GEMINI_API_KEY     -> Google Gemini (genuinely free tier, no card needed
#                          — get a key at aistudio.google.com/apikey)
#   GROQ_API_KEY       -> Groq (free tier, no card needed — console.groq.com)
#   ANTHROPIC_API_KEY  -> Claude (no free tier, needs prepaid credit)
# Optionally override the model name with LLM_MODEL.
# --------------------------------------------------------------------------

_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

_client = None  # kept for backwards-compat truthiness checks elsewhere
_PROVIDER = None
MODEL = os.environ.get("LLM_MODEL", "")

if _GEMINI_KEY:
    _PROVIDER = "gemini"
    MODEL = MODEL or "gemini-2.0-flash"
    _client = True
elif _GROQ_KEY:
    _PROVIDER = "groq"
    MODEL = MODEL or "llama-3.3-70b-versatile"
    _client = True
elif _ANTHROPIC_KEY:
    _PROVIDER = "anthropic"
    MODEL = MODEL or "claude-sonnet-4-6"
    try:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        _client = True
    except Exception:
        _client = None
        _PROVIDER = None
else:
    MODEL = MODEL or "template-fallback"


def _call_llm(system: str, user: str, max_tokens: int = 500) -> Optional[str]:
    """Return raw text from the LLM, or None on any failure (caller must
    fall back to templates — never raise, never block past the budget)."""
    if _PROVIDER is None:
        return None
    try:
        if _PROVIDER == "gemini":
            return _call_gemini(system, user, max_tokens)
        if _PROVIDER == "groq":
            return _call_groq(system, user, max_tokens)
        if _PROVIDER == "anthropic":
            return _call_anthropic(system, user, max_tokens)
    except Exception as e:
        if os.environ.get("LLM_DEBUG"):
            import sys, traceback
            print(f"[LLM_DEBUG] {_PROVIDER} call failed: {e!r}", file=sys.stderr)
            traceback.print_exc()
        return None
    return None


def _call_gemini(system: str, user: str, max_tokens: int) -> Optional[str]:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL}:generateContent?key={_GEMINI_KEY}")
    body = json.dumps({
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {e.code}: {err_body}") from None
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini returned no usable candidate: {json.dumps(data)[:500]}") from None


def _call_groq(system: str, user: str, max_tokens: int) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    body = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_GROQ_KEY}",
    })
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _call_anthropic(system: str, user: str, max_tokens: int) -> Optional[str]:
    resp = _anthropic_client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    return "\n".join(parts).strip()


def _parse_json_block(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(json)?", "", t.strip(), flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t.strip()).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------
# Shared framework knowledge baked into every prompt (from challenge-brief.md)
# --------------------------------------------------------------------------

SYSTEM_RULES = """You are composing a single WhatsApp message for "Vera", magicpin's
merchant-marketing assistant — but you must be BETTER than production Vera.

Hard rules:
1. Specificity wins. Anchor on a concrete, verifiable fact from the contexts
   given to you (a number, a date, a headline, a peer stat). Never write
   generic filler like "10% off" or "grow your business" when a specific
   service+price or number is available.
2. Category fit. Match the category's voice exactly (tone, allowed
   vocabulary, taboo words). Clinical categories (dentists, doctors) need a
   peer/colleague tone, never promotional/hype.
3. Merchant fit. Personalize to THIS merchant's actual numbers, offers,
   conversation history and signals. Honor their language preference
   (Hindi-English code-mix is fine and often preferred for Indian merchants
   — match what the merchant/category context implies).
4. Trigger relevance. State plainly *why you're messaging now* — tie it to
   the specific trigger.
5. Engagement compulsion. Use at least one lever: curiosity, loss aversion,
   social proof, effort externalization ("I've drafted X, just say go"),
   reciprocity, asking the merchant a direct question, or a single binary
   CTA (YES/STOP). Prefer social proof and "asking the merchant" — these are
   underused by production Vera.
6. Single primary CTA. One call-to-action, in the last sentence. Never
   multiple-choice CTAs ("reply YES for X, NO for Y").
7. Never fabricate. If a fact (a citation, a competitor name, a number)
   is not present in the contexts you were given, do not invent it. If the
   trigger payload is thin/placeholder, write a shorter, still-relevant
   message rather than making things up.
8. No long preambles ("I hope you're doing well..."), no re-introducing
   yourself after the first message, no promotional hype tone for clinical
   categories.
9. Never repeat a message verbatim that was already sent in this
   conversation.

Respond with STRICT JSON only, no markdown fences, no commentary:
{"body": "...", "cta": "open_ended" | "binary_yes_stop" | "none", "rationale": "..."}
"""


def _trim(obj: Any, max_len: int = 4000) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= max_len else s[:max_len] + "...<truncated>"


# --------------------------------------------------------------------------
# First-touch composition — /v1/tick path
# --------------------------------------------------------------------------

def compose(category: dict, merchant: dict, trigger: dict,
            customer: Optional[dict] = None) -> dict:
    send_as = "merchant_on_behalf" if (customer and trigger.get("scope") == "customer") else "vera"

    user_prompt = f"""Compose the NEXT message. This is a first-touch (no prior turns in
this conversation), so if there's no prior merchant reply in `conversation_history`
treat the WhatsApp 24h template rule loosely (we won't call Meta directly — write
natural free-form copy; a real deployment would slot this into a template).

send_as = "{send_as}"  ({"message TO the merchant's own customer, drafted for the merchant's WhatsApp number" if send_as == "merchant_on_behalf" else "message TO the merchant, from Vera"})

CATEGORY CONTEXT:
{_trim(category)}

MERCHANT CONTEXT:
{_trim(merchant)}

TRIGGER CONTEXT (this is WHY you're messaging right now):
{_trim(trigger)}

CUSTOMER CONTEXT (only relevant if send_as is merchant_on_behalf):
{_trim(customer) if customer else "none"}

Write the single best message now, as JSON per the schema."""

    raw = _call_llm(SYSTEM_RULES, user_prompt)
    parsed = _parse_json_block(raw) if raw else None

    if not parsed or "body" not in parsed or not parsed["body"]:
        parsed = _fallback_compose(category, merchant, trigger, customer, send_as)

    body = parsed["body"].strip()
    cta = parsed.get("cta", "open_ended")
    rationale = parsed.get("rationale", "composed from category+merchant+trigger context")

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", trigger.get("id", "unknown")),
        "rationale": rationale,
    }


def _fallback_compose(category: dict, merchant: dict, trigger: dict,
                       customer: Optional[dict], send_as: str) -> dict:
    """Deterministic, no-LLM template path. Used when the LLM is unavailable
    or fails — keeps the bot alive and still reasonably on-brief."""
    name = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "there")
    kind = trigger.get("kind", "update")
    payload = trigger.get("payload", {}) or {}
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_line = offers[0]["title"] if offers else None

    if kind == "research_digest" and category.get("digest"):
        item = category["digest"][0]
        body = (f"{name}, {item.get('source', 'this week\u2019s digest')} — "
                f"{item.get('title', 'a relevant update')}. "
                f"Worth a look — want me to pull the details?")
    elif kind == "perf_dip":
        perf = merchant.get("performance", {})
        body = (f"{name}, your calls are down this week (was {perf.get('calls', 'N/A')} last 30d). "
                f"Want me to check what's changed on your listing?")
    elif kind == "perf_spike":
        perf = merchant.get("performance", {})
        body = (f"{name}, quick update — your profile views jumped this week "
                f"({perf.get('views', 'N/A')} in the last 30 days). Want the breakdown?")
    elif kind == "recall_due" and customer:
        cname = customer.get("identity", {}).get("name", "your customer")
        slots = payload.get("available_slots", [])
        slot_txt = " or ".join(s.get("label", "") for s in slots[:2]) if slots else "a slot that works"
        offer_txt = f" {offer_line}." if offer_line else ""
        body = (f"Hi {cname}, it's been a while since your last visit — "
                f"your recall is due. {slot_txt} works?{offer_txt} Reply to book.")
    elif offer_line:
        body = f"{name}, quick one — your {offer_line} offer is live. Want me to push it to a few nearby searches this week?"
    else:
        body = f"{name}, checking in — anything on your profile you'd like help with this week?"

    return {"body": body, "cta": "open_ended", "rationale": f"template fallback for kind={kind}"}


# --------------------------------------------------------------------------
# Multi-turn conversation handling — /v1/reply path
# --------------------------------------------------------------------------

INTENT_PATTERNS = [
    r"\blet'?s do it\b", r"\bgo ahead\b", r"\bstart karo\b", r"\bkar do\b",
    r"\bkardo\b", r"\bshuru kar(o|iye|do)\b", r"\bsign me up\b", r"\bi'?m in\b",
    r"\byes,? (i want to join|please proceed|do it)\b", r"\bhaan chaliye\b",
    r"\btheek hai,? (karo|kijiye|start)\b", r"\bconfirm(ed)? kar do\b",
    r"\bok(ay)?,? (lets|let'?s) do it\b", r"\bwhat'?s next\b", r"\bwhats next\b",
]

STOP_PATTERNS = [
    r"\bstop\b", r"\bunsubscribe\b", r"\bnot interested\b",
    r"\bdon'?t (message|contact|text) me\b", r"\bspam\b", r"\bband karo\b",
    r"\bmat bhejo\b", r"\buseless\b",
]

WAIT_PATTERNS = [
    r"\blater\b", r"\bnot now\b", r"\bbusy\b", r"\bcall you back\b",
    r"\bthodi der\b", r"\bbaad me\b", r"\bbaad mein\b", r"\bfursat\b",
]

AUTO_REPLY_HINTS = [
    "thank you for contacting", "will respond shortly", "automated assistant",
    "we will get back to you", "out of office",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def new_conversation_state(merchant_id: str, customer_id: Optional[str],
                            send_as: str, trigger_id: Optional[str],
                            first_body: str) -> dict:
    return {
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": send_as,
        "trigger_id": trigger_id,
        "turns": 1,
        "ended": False,
        "action_mode": False,
        "inbound_counts": {},
        "sent_bodies": [first_body] if first_body else [],
        "history": ([{"role": "bot", "text": first_body}] if first_body else []),
    }


def decide_reply(state: dict, message: str, category: Optional[dict] = None,
                  merchant: Optional[dict] = None, trigger: Optional[dict] = None,
                  customer: Optional[dict] = None) -> dict:
    """Returns {"action": "send"|"wait"|"end", body?, cta?, wait_seconds?, rationale}."""
    norm = _normalize(message)
    state.setdefault("inbound_counts", {})
    state.setdefault("history", [])
    state.setdefault("sent_bodies", [])

    count = state["inbound_counts"].get(norm, 0) + 1
    state["inbound_counts"][norm] = count
    state["history"].append({"role": "other", "text": message})

    # 1. Explicit opt-out always wins.
    if any(re.search(p, norm) for p in STOP_PATTERNS):
        state["ended"] = True
        return {"action": "end", "rationale": "Merchant/customer asked to stop — respecting opt-out immediately."}

    # 2. Auto-reply detection — same verbatim text seen 3+ times, or matches
    #    a known canned-reply phrase and has already repeated once.
    looks_auto = any(h in norm for h in AUTO_REPLY_HINTS)
    if count >= 3 or (looks_auto and count >= 2):
        state["ended"] = True
        return {"action": "end", "rationale": "Same message repeated verbatim — auto-reply detected, exiting to avoid wasting turns."}

    if count == 2 or (looks_auto and count == 1):
        body = "Samajh gayi — just so I don't waste your time: is this the owner/manager, or should I loop them in directly?"
        state["sent_bodies"].append(body)
        state["history"].append({"role": "bot", "text": body})
        return {"action": "send", "body": body, "cta": "open_ended",
                "rationale": "Possible auto-reply — probing once before exiting, per anti-auto-reply-pollution guidance."}

    # 3. Explicit intent -> switch to action mode immediately, no more qualifying.
    if any(re.search(p, norm) for p in INTENT_PATTERNS):
        state["action_mode"] = True
        body = _compose_reply_llm(state, message, category, merchant, trigger, customer, mode="action")
        if _looks_like_qualifying(body):
            body = "Done — setting this up now. I'll confirm here the moment it's live, no more questions needed from you."
        state["sent_bodies"].append(body)
        state["history"].append({"role": "bot", "text": body})
        return {"action": "send", "body": body, "cta": "none",
                "rationale": "Merchant signaled explicit intent — switched from pitch/qualification to action immediately."}

    # 4. Wants time -> back off, don't burn a turn.
    if any(re.search(p, norm) for p in WAIT_PATTERNS):
        return {"action": "wait", "wait_seconds": 1800,
                "rationale": "Merchant asked for time; backing off 30 minutes instead of pushing."}

    # 5. Normal in-conversation reply.
    body = _compose_reply_llm(state, message, category, merchant, trigger, customer, mode="normal")
    if _normalize(body) in {_normalize(b) for b in state["sent_bodies"]}:
        body = body.rstrip(".") + " — anything I can clarify?"
    state["sent_bodies"].append(body)
    state["history"].append({"role": "bot", "text": body})
    return {"action": "send", "body": body, "cta": "open_ended",
            "rationale": "Continuing the conversation based on the merchant's latest message."}


_QUALIFYING_HINTS = ["would you", "do you", "can you tell", "what if", "how about", "could you"]


def _looks_like_qualifying(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in _QUALIFYING_HINTS)


def _compose_reply_llm(state: dict, message: str, category: Optional[dict],
                        merchant: Optional[dict], trigger: Optional[dict],
                        customer: Optional[dict], mode: str) -> str:
    history_txt = "\n".join(f"[{h['role'].upper()}] {h['text']}" for h in state.get("history", [])[-8:])

    if mode == "action":
        mode_instructions = (
            "MODE: ACTION. The merchant just gave explicit go-ahead. Do NOT ask "
            "another qualifying question. Confirm you're proceeding and state the "
            "concrete next step in one line (e.g. what you're setting up / drafting "
            "/ sending next). No question marks unless it's a genuinely necessary "
            "logistics detail (e.g. which of two slots)."
        )
    else:
        mode_instructions = (
            "MODE: NORMAL. Continue the conversation naturally, staying on-mission "
            "(profile/marketing/offers help for this merchant). If the message is "
            "hostile, apologize briefly and stay polite without being submissive. "
            "If it's off-topic (e.g. asking for help with GST/taxes/unrelated "
            "personal favors), politely note that's outside what you can help with "
            "here and redirect back to one concrete thing you can do for them."
        )

    user_prompt = f"""Conversation so far:
{history_txt}

Latest inbound message: "{message}"

{mode_instructions}

CATEGORY CONTEXT: {_trim(category) if category else "none"}
MERCHANT CONTEXT: {_trim(merchant) if merchant else "none"}
TRIGGER CONTEXT (original reason for this conversation): {_trim(trigger) if trigger else "none"}
CUSTOMER CONTEXT: {_trim(customer) if customer else "none"}

Never repeat any body already sent in this conversation verbatim. Reply as JSON: {{"body": "...", "cta": "..."}}"""

    raw = _call_llm(SYSTEM_RULES, user_prompt, max_tokens=300)
    parsed = _parse_json_block(raw) if raw else None
    if parsed and parsed.get("body"):
        return parsed["body"].strip()

    # Fallback templates by mode
    if mode == "action":
        return "Done — setting this up now. I'll confirm here as soon as it's live."
    low = message.lower()
    if any(w in low for w in ["sorry", "abuse", "useless", "spam", "stupid"]):
        return "Sorry to have bothered you — happy to step back. If it's ever useful, just say the word."
    return "Got it — let me know if there's anything specific you'd like me to check or draft next."
