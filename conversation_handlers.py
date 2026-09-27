"""
conversation_handlers.py — optional deliverable (challenge-brief.md §7.4).

Thin wrapper around composer.decide_reply so the exact signature the brief
asks for exists as a standalone importable function, on top of the HTTP
/v1/reply path that bot.py exposes for the live judge harness.

    def respond(state: ConversationState, merchant_message: str) -> dict

`state` here is the same plain-dict conversation state produced by
composer.new_conversation_state(...) / mutated in place by decide_reply.
Callers that also want category/merchant/trigger/customer context taken
into account should attach them to state before calling, e.g.:

    state["_category"], state["_merchant"], state["_trigger"], state["_customer"] = ...
"""

from __future__ import annotations

import composer


def respond(state: dict, merchant_message: str) -> dict:
    """Given the conversation so far + the merchant's latest message,
    produce the reply. Returns {"action": "send"|"wait"|"end", ...}."""
    return composer.decide_reply(
        state,
        merchant_message,
        category=state.get("_category"),
        merchant=state.get("_merchant"),
        trigger=state.get("_trigger"),
        customer=state.get("_customer"),
    )
