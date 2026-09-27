"""
server.py — HTTP harness required by challenge-testing-brief.md §2.

Implements all 5 endpoints:
    POST /v1/context     — idempotent context push (category/merchant/customer/trigger)
    POST /v1/tick        — periodic wake-up; bot may proactively initiate
    POST /v1/reply       — synchronous reply to a merchant/customer turn
    GET  /v1/healthz     — liveness probe
    GET  /v1/metadata    — bot identity
    POST /v1/teardown    — optional; wipes state (privacy rule §11)

Run:
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import bot
import conversation_handlers as ch

app = FastAPI(title="Vera-beater bot")
START = time.time()

# ---------------------------------------------------------------------------
# In-memory state. Keyed exactly as the testing brief describes.
# ---------------------------------------------------------------------------
contexts: dict[tuple[str, str], dict] = {}          # (scope, context_id) -> {"version", "payload"}
conversations: dict[str, ch.ConversationState] = {}  # conversation_id -> state
sent_suppression_keys: set[str] = set()              # cross-conversation dedup
first_message_sent: set[tuple[str, Optional[str]]] = set()  # (merchant_id, customer_id) -> template used already

TEAM_NAME = "Vera-Beater"
MODEL_NAME = "claude-sonnet-4-6 (LLM-polish, optional) + deterministic rules composer"


# =============================================================================
# Helpers
# =============================================================================

def _get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _counts() -> dict[str, int]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# =============================================================================
# GET /v1/healthz
# =============================================================================

@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": _counts()}


# =============================================================================
# GET /v1/metadata
# =============================================================================

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": ["Claude + operator"],
        "model": MODEL_NAME,
        "approach": (
            "Deterministic, per-trigger-kind template composer with an anti-hallucination "
            "anchor resolver: real trigger payload wins; placeholder/thin triggers fall back "
            "to the strongest verifiable fact in merchant/customer/category context, or a "
            "generic asking-the-merchant hook. Optional temperature=0 LLM polish pass "
            "re-validates before use so quality only improves, never regresses into "
            "fabrication or off-voice copy."
        ),
        "contact_email": "team@example.com",
        "version": "1.0.0",
        "submitted_at": _now_iso(),
    }


# =============================================================================
# POST /v1/context
# =============================================================================

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"unknown scope '{body.scope}'"}
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": _now_iso()}


# =============================================================================
# POST /v1/tick
# =============================================================================

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _resolve_quad(trigger_id: str) -> Optional[tuple[dict, dict, dict, Optional[dict]]]:
    trigger = _get_ctx("trigger", trigger_id)
    if not trigger:
        return None
    merchant_id = trigger.get("merchant_id")
    merchant = _get_ctx("merchant", merchant_id) if merchant_id else None
    if not merchant:
        return None
    category = _get_ctx("category", merchant.get("category_slug"))
    if not category:
        return None
    customer = None
    customer_id = trigger.get("customer_id")
    if customer_id:
        customer = _get_ctx("customer", customer_id)
    return category, merchant, trigger, customer


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers[:20]:  # tick action cap
        quad = _resolve_quad(trg_id)
        if not quad:
            continue
        category, merchant, trigger, customer = quad

        suppression_key = trigger.get("suppression_key", trg_id)
        if suppression_key in sent_suppression_keys:
            continue  # dedup: don't resend the same logical nudge

        merchant_id = merchant.get("merchant_id")
        customer_id = customer.get("customer_id") if customer else None
        target_key = (merchant_id, customer_id)
        is_first = target_key not in first_message_sent

        result = bot.compose(category, merchant, trigger, customer)
        if not result.get("body"):
            continue  # restraint: don't send an empty/failed composition

        conversation_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:8]}"
        state = ch.ConversationState(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            send_as=result["send_as"],
        )
        state.sent_bodies.append(result["body"])
        conversations[conversation_id] = state

        sent_suppression_keys.add(suppression_key)
        first_message_sent.add(target_key)

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result["send_as"],
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1" if is_first else None,
            "template_params": [merchant.get("identity", {}).get("name", merchant_id)] if is_first else None,
            "body": result["body"],
            "cta": result["cta"],
            "suppression_key": suppression_key,
            "rationale": result["rationale"],
        })
    return {"actions": actions}


# =============================================================================
# POST /v1/reply
# =============================================================================

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    state = conversations.get(body.conversation_id)
    if state is None:
        state = ch.ConversationState(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id or "",
            customer_id=body.customer_id,
        )
        conversations[body.conversation_id] = state

    result = ch.respond(state, body.message)

    if result["action"] == "send":
        # Anti-repetition guard at the transport layer too.
        if result["body"] in state.sent_bodies[:-1]:
            return {"action": "send", "body": result["body"] + " (as mentioned above)",
                    "cta": result.get("cta", "open_ended"), "rationale": result["rationale"]}
        return result
    return result


# =============================================================================
# POST /v1/teardown (optional, privacy rule §11)
# =============================================================================

@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    first_message_sent.clear()
    return {"status": "wiped"}
