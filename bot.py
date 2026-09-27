"""
bot.py — magicpin AI Challenge ("Vera") candidate bot.

Implements the 5-endpoint HTTP contract from challenge-testing-brief.md:
    POST /v1/context   — receive/store a context push (category|merchant|customer|trigger)
    POST /v1/tick      — periodic wake-up; bot may proactively send
    POST /v1/reply     — receive a merchant/customer reply; bot must respond synchronously
    GET  /v1/healthz   — liveness probe
    GET  /v1/metadata  — bot identity

Run locally:
    ANTHROPIC_API_KEY=sk-... uvicorn bot:app --host 0.0.0.0 --port 8080

Deploy: any host that gives you a public URL works (Render, Railway, Fly,
Replit, a VM, ngrok). Set ANTHROPIC_API_KEY (or leave unset to run in
template-fallback mode) as an environment variable on that host.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import composer

app = FastAPI(title="Vera-beating bot")
START = time.time()

# --------------------------------------------------------------------------
# In-memory stores. Fine for a challenge run; swap for Redis/SQLite for
# anything long-lived across restarts.
# --------------------------------------------------------------------------

# (scope, context_id) -> {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> composer conversation state dict
conversations: dict[str, dict] = {}

# suppression_key -> True, once a first-touch has been sent for it (avoid
# re-sending the same trigger twice across ticks)
sent_suppression_keys: set[str] = set()

TEAM_NAME = os.environ.get("TEAM_NAME", "Vanshika")
TEAM_MEMBERS = [os.environ.get("TEAM_MEMBER_1", "Vanshika")]
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
BOT_VERSION = "1.0.0"
SUBMITTED_AT = os.environ.get("SUBMITTED_AT", datetime.now(timezone.utc).isoformat())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_payload(scope: str, context_id: Optional[str]) -> Optional[dict]:
    if not context_id:
        return None
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _category_for_merchant(merchant: Optional[dict]) -> Optional[dict]:
    if not merchant:
        return None
    return _get_payload("category", merchant.get("category_slug"))


# --------------------------------------------------------------------------
# 2.4 GET /v1/healthz
# --------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
    }


# --------------------------------------------------------------------------
# 2.5 GET /v1/metadata
# --------------------------------------------------------------------------

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": composer.MODEL,
        "approach": (
            "Single LLM composer over the 4-context framework (category/merchant/"
            "trigger/customer), temperature=0, JSON-schema output. A rule layer "
            "handles auto-reply detection (verbatim-repeat counting), explicit "
            "intent transitions, STOP/opt-out, and wait-for-later — the LLM only "
            "writes prose once the control-flow decision is made. Deterministic "
            "template fallback if the LLM is unavailable or times out."
        ),
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": SUBMITTED_AT,
    }


# --------------------------------------------------------------------------
# 2.1 POST /v1/context
# --------------------------------------------------------------------------

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"unknown scope {body.scope!r}"}

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


# --------------------------------------------------------------------------
# 2.2 POST /v1/tick
# --------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers[:20]:  # respect the 20-actions-per-tick cap
        trigger = _get_payload("trigger", trg_id)
        if not trigger:
            continue

        supp_key = trigger.get("suppression_key", trg_id)
        if supp_key in sent_suppression_keys:
            continue  # dedup — already messaged for this trigger

        merchant_id = trigger.get("merchant_id")
        merchant = _get_payload("merchant", merchant_id)
        category = _category_for_merchant(merchant)
        if not merchant or not category:
            continue  # don't compose without the context we need

        customer_id = trigger.get("customer_id")
        customer = _get_payload("customer", customer_id) if customer_id else None

        composed = composer.compose(category, merchant, trigger, customer)

        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        conversations[conversation_id] = composer.new_conversation_state(
            merchant_id=merchant_id,
            customer_id=customer_id,
            send_as=composed["send_as"],
            trigger_id=trg_id,
            first_body=composed["body"],
        )
        sent_suppression_keys.add(supp_key)

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("owner_first_name", "")],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        })

        if len(actions) >= 20:
            break

    return {"actions": actions}


# --------------------------------------------------------------------------
# 2.3 POST /v1/reply
# --------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: Optional[int] = None


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    state = conversations.get(body.conversation_id)
    if state is None:
        # Unknown conversation (e.g. server restarted) — start fresh state
        # rather than erroring out, so the judge always gets a valid response.
        state = composer.new_conversation_state(
            merchant_id=body.merchant_id, customer_id=body.customer_id,
            send_as="vera", trigger_id=None, first_body="",
        )
        conversations[body.conversation_id] = state

    if state.get("ended"):
        return {"action": "end", "rationale": "Conversation already ended."}

    state["turns"] = state.get("turns", 0) + 1

    merchant = _get_payload("merchant", body.merchant_id)
    category = _category_for_merchant(merchant)
    customer = _get_payload("customer", body.customer_id) if body.customer_id else None
    trigger = _get_payload("trigger", state.get("trigger_id")) if state.get("trigger_id") else None

    result = composer.decide_reply(
        state, body.message, category=category, merchant=merchant,
        trigger=trigger, customer=customer,
    )

    out: dict[str, Any] = {"action": result["action"], "rationale": result.get("rationale", "")}
    if result["action"] == "send":
        out["body"] = result["body"]
        out["cta"] = result.get("cta", "open_ended")
    elif result["action"] == "wait":
        out["wait_seconds"] = result.get("wait_seconds", 1800)

    return out


@app.get("/")
async def root():
    return {"ok": True, "service": "vera-beating-bot", "see": "/v1/healthz"}
