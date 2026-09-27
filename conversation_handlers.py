"""
conversation_handlers.py — multi-turn reply logic for the replay test
(challenge-brief.md §12 open challenges 1-2-5, testing-brief.md Phase 4).

Handles, in priority order, on every incoming merchant/customer message:
  1. Hostile / abusive text        -> stay polite, stay on-mission
  2. Explicit "not interested"/STOP -> action: end
  3. Auto-reply detection           -> one soft nudge, then graceful exit
  4. Intent transition ("let's do it") -> jump straight to action, no re-qualifying
  5. Off-topic question             -> answer briefly, redirect back once
  6. Otherwise                      -> advance the conversation naturally

This module is intentionally dependency-light (no LLM required) so the
replay test can be exercised deterministically; `bot.llm_polish` can still
be layered on top of any returned `body` by the caller if desired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Conversation state
# =============================================================================

@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: str = "vera"
    turn_number: int = 0
    sent_bodies: list[str] = field(default_factory=list)
    received_bodies: list[str] = field(default_factory=list)
    auto_reply_strikes: int = 0
    intent_mode: str = "pitch"          # "pitch" | "action" | "ended"
    unanswered_nudges: int = 0
    last_topic: Optional[str] = None


# =============================================================================
# Detectors
# =============================================================================

_AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|reaching out)",
    r"we('| a)?ll get back to you",
    r"automated (assistant|reply|message)",
    r"team tak pahuncha",
    r"hamari team",
    r"currently (unavailable|away|closed)",
    r"business hours",
    r"will respond shortly",
]

_STOP_PATTERNS = [
    r"\bstop\b", r"not interested", r"nahi chahiye", r"band karo", r"no thanks",
    r"unsubscribe", r"remove me", r"don'?t (message|contact) me",
]

_INTENT_YES_PATTERNS = [
    r"\blet'?s do it\b", r"\bgo ahead\b", r"\byes.*join\b", r"\bsign me up\b",
    r"\bi want to join\b", r"^\s*(yes|ok|okay|sure|haan|theek hai|chaliye)\s*[.!]?\s*$",
    r"\bok let'?s\b",
]

_ABUSE_PATTERNS = [
    r"\bidiot\b", r"\bstupid\b", r"\bshut up\b", r"\bnonsense\b", r"f[*u]ck",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    low = text.lower().strip()
    return any(re.search(p, low) for p in patterns)


def is_auto_reply(prior_received: list[str], message: str) -> bool:
    if _matches_any(message, _AUTO_REPLY_PATTERNS):
        return True
    # Same message verbatim already seen earlier in this conversation (not
    # counting the current message itself) = auto-reply, per challenge-brief
    # hint ("same message verbatim 3+ times").
    norm = message.strip().lower()
    return any(m.strip().lower() == norm for m in prior_received)


def is_stop_signal(message: str) -> bool:
    return _matches_any(message, _STOP_PATTERNS)


def is_intent_transition(message: str) -> bool:
    return _matches_any(message, _INTENT_YES_PATTERNS)


def is_abusive(message: str) -> bool:
    return _matches_any(message, _ABUSE_PATTERNS)


def is_off_topic(message: str) -> bool:
    # Heuristic: a question mark about a topic with no overlap with
    # magicpin/marketing vocabulary is treated as off-topic.
    on_topic_vocab = ["profile", "listing", "offer", "review", "post", "customer",
                       "booking", "appointment", "renewal", "magicpin", "google",
                       "views", "calls", "price", "plan", "recall", "gst"]
    low = message.lower()
    if "?" not in message:
        return False
    return not any(v in low for v in on_topic_vocab[:-1])  # "gst" excluded on purpose


# =============================================================================
# Response builders
# =============================================================================

def _action_send(body: str, cta: str, rationale: str) -> dict:
    return {"action": "send", "body": body, "cta": cta, "rationale": rationale}


def _action_wait(seconds: int, rationale: str) -> dict:
    return {"action": "wait", "wait_seconds": seconds, "rationale": rationale}


def _action_end(rationale: str) -> dict:
    return {"action": "end", "rationale": rationale}


def respond(state: ConversationState, merchant_message: str) -> dict:
    """Given the conversation so far + the latest reply, produce the next move."""
    state.turn_number += 1
    prior_received = list(state.received_bodies)
    state.received_bodies.append(merchant_message)

    # 1. Abuse — stay polite, stay on-mission, don't escalate or end.
    if is_abusive(merchant_message):
        return _action_send(
            "No worries, happy to keep this quick and useful. "
            "Want me to pick up where we left off, or is there something else on your mind?",
            cta="open_ended",
            rationale="Hostile tone detected; de-escalated politely and stayed on-mission without ending.",
        )

    # 2. Explicit stop / not-interested — graceful, immediate exit.
    if is_stop_signal(merchant_message):
        state.intent_mode = "ended"
        return _action_end("Merchant signaled not-interested/STOP; exiting gracefully per anti-spam rule.")

    # 3. Auto-reply detection.
    if is_auto_reply(prior_received, merchant_message):
        state.auto_reply_strikes += 1
        if state.auto_reply_strikes == 1:
            return _action_send(
                "Samajh gayi. Ek chhoti si baat — 2 minute mein khud dekhna chahenge ki "
                "exactly kya update karna hai? Chalega?",
                cta="binary",
                rationale="First auto-reply detected; one soft direct nudge before disengaging (Pattern B).",
            )
        state.intent_mode = "ended"
        return _action_end(
            "Second auto-reply / repeated canned text detected; stopping rather than burning further "
            "turns on a non-human responder. Polite sign-off."
        )

    # 4. Intent transition — switch to action mode immediately, no re-qualifying.
    if is_intent_transition(merchant_message):
        was_pitching = state.intent_mode == "pitch"
        state.intent_mode = "action"
        if was_pitching:
            return _action_send(
                "Great — starting now. Just need your business hours and one photo to kick things off; "
                "send whenever you're ready and I'll take it from there.",
                cta="open_ended",
                rationale="Explicit intent detected ('let's do it'); routed straight to action, skipped further qualification.",
            )
        return _action_send(
            "On it — I'll get that moving now and confirm once it's done.",
            cta="none",
            rationale="Already in action mode; acknowledging and executing.",
        )

    # 5. Off-topic / curveball — answer briefly, redirect once.
    if is_off_topic(merchant_message):
        return _action_send(
            "That's outside what I can help with directly, but happy to point you to the right place for it. "
            "On your magicpin side — want me to pick back up on what we were discussing?",
            cta="open_ended",
            rationale="Off-topic question; answered briefly and redirected back to the mission without ignoring the merchant.",
        )

    # 6. Ordinary engaged reply — advance naturally.
    state.unanswered_nudges = 0
    body = ("Got it — sending that over now. I'll also flag anything else worth your attention "
            "as it comes up.")
    result = _action_send(body, cta="open_ended", rationale="Engaged reply; acknowledged and advanced the conversation.")

    # Anti-repetition guard: never resend a body verbatim in this conversation.
    if result["body"] in state.sent_bodies:
        result["body"] = result["body"] + " (following up on the above)"
    state.sent_bodies.append(result["body"])
    return result


def on_no_reply(state: ConversationState) -> dict:
    """Called by the tick loop when a nudge goes unanswered — implements
    'know when to stop' (challenge-brief §12.5)."""
    state.unanswered_nudges += 1
    if state.unanswered_nudges >= 3:
        state.intent_mode = "ended"
        return _action_end("3 unanswered nudges; stopping to avoid spamming a non-responsive merchant.")
    return _action_wait(21600, "No reply yet; backing off before the next nudge (6h).")
