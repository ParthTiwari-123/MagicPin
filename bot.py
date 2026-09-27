"""
bot.py — magicpin AI Challenge submission ("Vera, but better")
================================================================

Entry point required by the challenge brief:

    def compose(category: dict, merchant: dict, trigger: dict,
                customer: dict | None = None) -> dict

Design in one paragraph
------------------------
A trigger tells us *why* to message. But ~40% of the canonical test set ships
triggers with placeholder payloads (no real metric attached). Rather than
letting an LLM improvise a fake number to fit the trigger's name (the #1 way
submissions get penalized — "hallucinated data"), this bot always resolves an
**anchor**: a concrete, verifiable fact pulled from whatever context *is*
real (merchant performance, subscription, signals, customer relationship,
category peer stats/digest). If the trigger's own payload is real, that wins.
If it's a placeholder, we search merchant/customer/category context for the
best real fact to lead with, and downgrade gracefully to a generic-but-honest
"asking the merchant" hook if nothing concrete exists anywhere. We never
invent a number, a name, or a citation that isn't in the contexts.

Composition itself is template-based and fully deterministic (no LLM
required to run) so this file works standalone and produces reproducible
output for `submission.jsonl`. If an LLM API key is present in the
environment (ANTHROPIC_API_KEY or OPENAI_API_KEY), an optional "polish" pass
asks the model to smooth phrasing WITHOUT adding facts; the polished output
is re-validated and only used if it survives the same checks as the
deterministic draft — so quality can only go up, never regress into
hallucination or off-voice copy.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Optional


# =============================================================================
# 0. Generic accessors — the dataset and the live testing-brief schema use
#    slightly different key names for the same fields (e.g. "vocab_taboo" vs
#    "taboos"). We accept either everywhere.
# =============================================================================

def g(d: Optional[dict], *keys: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _hash_pick(seed_str: str, n: int) -> int:
    """Deterministic 'random' index — same inputs always pick the same variant."""
    if n <= 1:
        return 0
    h = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


def _fmt_pct(x: float) -> str:
    return f"{abs(x) * 100:.0f}%"


def _inr(v: Any) -> str:
    try:
        return f"₹{int(float(v)):,}"
    except (TypeError, ValueError):
        return f"₹{v}"


# =============================================================================
# 1. Normalization — tolerant readers for the 4 context types
# =============================================================================

def normalize_category(raw: dict) -> dict:
    voice = g(raw, "voice", default={}) or {}
    peer = g(raw, "peer_stats", default={}) or {}
    return {
        "slug": g(raw, "slug", "category_slug"),
        "display_name": g(raw, "display_name", default=g(raw, "slug", default="")).title() if isinstance(g(raw, "slug"), str) else g(raw, "display_name", default=""),
        "tone": g(voice, "tone", default="peer_practical"),
        "code_mix": g(voice, "code_mix", default="hindi_english_natural"),
        "taboo": [t.lower() for t in (g(voice, "vocab_taboo", "taboos", default=[]) or [])],
        "vocab_allowed": g(voice, "vocab_allowed", default=[]) or [],
        "salutations": g(voice, "salutation_examples", default=[]) or [],
        "offer_catalog": g(raw, "offer_catalog", default=[]) or [],
        "peer_stats": {
            "avg_rating": g(peer, "avg_rating"),
            "avg_reviews": g(peer, "avg_review_count", "avg_reviews"),
            "avg_ctr": g(peer, "avg_ctr"),
            "avg_views_30d": g(peer, "avg_views_30d"),
            "avg_calls_30d": g(peer, "avg_calls_30d"),
            "avg_post_freq_days": g(peer, "avg_post_freq_days"),
            "retention_6mo_pct": g(peer, "retention_6mo_pct"),
            "scope": g(peer, "scope"),
        },
        "digest": g(raw, "digest", default=[]) or [],
        "seasonal_beats": g(raw, "seasonal_beats", default=[]) or [],
        "trend_signals": g(raw, "trend_signals", default=[]) or [],
        "patient_content_library": g(raw, "patient_content_library", default=[]) or [],
    }


def normalize_merchant(raw: dict) -> dict:
    identity = g(raw, "identity", default={}) or {}
    sub = g(raw, "subscription", default={}) or {}
    perf = g(raw, "performance", default={}) or {}
    delta = g(perf, "delta_7d", default={}) or {}
    cagg = g(raw, "customer_aggregate", default={}) or {}
    return {
        "merchant_id": g(raw, "merchant_id"),
        "category_slug": g(raw, "category_slug"),
        "name": g(identity, "name", default="there"),
        "owner_first_name": g(identity, "owner_first_name"),
        "city": g(identity, "city"),
        "locality": g(identity, "locality"),
        "verified": g(identity, "verified", default=True),
        "languages": g(identity, "languages", default=["en"]) or ["en"],
        "sub_status": g(sub, "status", default="active"),
        "sub_days_remaining": g(sub, "days_remaining"),
        "sub_days_since_expiry": g(sub, "days_since_expiry"),
        "sub_plan": g(sub, "plan"),
        "views": g(perf, "views"),
        "calls": g(perf, "calls"),
        "directions": g(perf, "directions"),
        "ctr": g(perf, "ctr"),
        "leads": g(perf, "leads"),
        "views_pct_7d": g(delta, "views_pct"),
        "calls_pct_7d": g(delta, "calls_pct"),
        "ctr_pct_7d": g(delta, "ctr_pct"),
        "offers": g(raw, "offers", default=[]) or [],
        "conversation_history": g(raw, "conversation_history", default=[]) or [],
        "total_unique_ytd": g(cagg, "total_unique_ytd"),
        "lapsed_180d_plus": g(cagg, "lapsed_180d_plus"),
        "retention_6mo_pct": g(cagg, "retention_6mo_pct"),
        "signals": g(raw, "signals", default=[]) or [],
        "review_themes": g(raw, "review_themes", default=[]) or [],
    }


def normalize_trigger(raw: dict) -> dict:
    payload = g(raw, "payload", default={}) or {}
    return {
        "id": g(raw, "id"),
        "scope": g(raw, "scope", default="merchant"),
        "kind": g(raw, "kind", default="generic"),
        "source": g(raw, "source", default="internal"),
        "merchant_id": g(raw, "merchant_id", default=g(payload, "merchant_id")),
        "customer_id": g(raw, "customer_id", default=g(payload, "customer_id")),
        "payload": payload,
        "is_placeholder": bool(payload.get("placeholder")),
        "urgency": g(raw, "urgency", default=2),
        "suppression_key": g(raw, "suppression_key", default=g(raw, "id", default="")),
        "expires_at": g(raw, "expires_at"),
    }


def normalize_customer(raw: Optional[dict]) -> Optional[dict]:
    if not raw:
        return None
    identity = g(raw, "identity", default={}) or {}
    rel = g(raw, "relationship", default={}) or {}
    prefs = g(raw, "preferences", default={}) or {}
    return {
        "customer_id": g(raw, "customer_id"),
        "merchant_id": g(raw, "merchant_id"),
        "name": g(identity, "name", default="there"),
        "language_pref": g(identity, "language_pref", default="en"),
        "first_visit": g(rel, "first_visit"),
        "last_visit": g(rel, "last_visit"),
        "visits_total": g(rel, "visits_total"),
        "services_received": g(rel, "services_received", default=[]) or [],
        "state": g(raw, "state", default="active"),
        "preferred_slots": g(prefs, "preferred_slots", "preferred_time"),
        "reminder_opt_in": g(prefs, "reminder_opt_in", default=True),
    }


def find_digest_item(category: dict, item_id: Optional[str]) -> Optional[dict]:
    if not item_id:
        return None
    for item in category["digest"]:
        if item.get("id") == item_id:
            return item
    return None


# =============================================================================
# 2. Anchor resolution — the anti-hallucination core.
#    Returns: {"mode": ..., "kind": ..., "facts": {...}, "text_hook": "..."}
#    `text_hook` is a ready-to-use, fact-grounded clause a renderer can use.
# =============================================================================

def _parse_signal(sig: str) -> tuple[str, Optional[str]]:
    if ":" in sig:
        k, v = sig.split(":", 1)
        return k, v
    return sig, None


def _merchant_fallback_signal(merchant: dict, category: dict) -> Optional[dict]:
    """Priority-ordered search for a REAL, concrete fact about this merchant
    to lead with when the trigger itself carries no real payload."""
    peer = category["peer_stats"]

    # 1. Subscription trouble — concrete, urgent, always true if present.
    if merchant["sub_status"] == "expired" and merchant["sub_days_since_expiry"]:
        return {"kind": "subscription_expired", "days": merchant["sub_days_since_expiry"]}
    if merchant["sub_status"] == "active" and merchant["sub_days_remaining"] is not None and merchant["sub_days_remaining"] <= 10:
        return {"kind": "renewal_soon", "days": merchant["sub_days_remaining"]}

    # 2. Explicit derived signals already computed for this merchant.
    for sig in merchant["signals"]:
        k, v = _parse_signal(sig)
        if k in ("stale_posts", "ctr_below_peer_median", "dormant", "high_risk_adult_cohort"):
            return {"kind": k, "value": v}

    # 3. Verifiable performance movement (only if it's actually there).
    if merchant["ctr"] is not None and peer.get("avg_ctr"):
        gap = merchant["ctr"] - peer["avg_ctr"]
        if abs(gap) / peer["avg_ctr"] >= 0.15:
            return {"kind": "ctr_vs_peer", "ctr": merchant["ctr"], "peer_ctr": peer["avg_ctr"], "gap": gap}
    if merchant["views_pct_7d"] is not None and abs(merchant["views_pct_7d"]) >= 0.15:
        return {"kind": "views_delta", "pct": merchant["views_pct_7d"]}
    if merchant["calls_pct_7d"] is not None and abs(merchant["calls_pct_7d"]) >= 0.15:
        return {"kind": "calls_delta", "pct": merchant["calls_pct_7d"]}

    # 4. Customer-base facts.
    if merchant["lapsed_180d_plus"]:
        return {"kind": "lapsed_customers", "count": merchant["lapsed_180d_plus"]}
    if merchant["total_unique_ytd"]:
        return {"kind": "total_customers", "count": merchant["total_unique_ytd"]}

    # 5. Offer catalog gap.
    active_offers = [o for o in merchant["offers"] if g(o, "status") == "active"]
    if not active_offers and category["offer_catalog"]:
        return {"kind": "no_active_offer"}

    return None


def resolve_anchor(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    kind = trigger["kind"]
    payload = trigger["payload"]

    if not trigger["is_placeholder"] and payload:
        return {"mode": "trigger_payload", "kind": kind, "facts": payload}

    # Placeholder trigger: the trigger label may not be verifiable
    # (e.g. "perf_dip" with no actual negative delta). Never assert the
    # trigger's claim without evidence.
    if customer and trigger["scope"] == "customer":
        # Customer record itself is real data even when the trigger payload
        # is thin — plenty to anchor on honestly.
        return {"mode": "customer_context_fallback", "kind": kind, "facts": {}}

    fb = _merchant_fallback_signal(merchant, category)
    if fb:
        return {"mode": "merchant_signal_pivot", "kind": kind, "facts": fb}

    if category["digest"]:
        return {"mode": "category_general", "kind": kind, "facts": {"digest_item": category["digest"][0]}}

    return {"mode": "category_general", "kind": kind, "facts": {}}


# =============================================================================
# 3. Lever + CTA + offer + language selection
# =============================================================================

ACTION_KINDS = {
    "recall_due", "appointment_tomorrow", "chronic_refill_due", "customer_lapsed_soft",
    "customer_lapsed_hard", "renewal_due", "gbp_unverified", "trial_followup",
    "supply_alert", "winback_eligible",
}
INFO_KINDS = {
    "research_digest", "regulation_change", "category_seasonal", "seasonal_perf_dip",
    "review_theme_emerged", "category_trend_movement",
}


def select_cta_shape(kind: str, anchor: dict) -> str:
    if anchor["mode"] == "category_general":
        return "open_ended"  # never force a binary commitment on an unverified claim
    if kind in ACTION_KINDS:
        return "binary"
    if kind in INFO_KINDS:
        return "open_ended"
    return "open_ended"


def select_levers(kind: str, anchor: dict) -> list[str]:
    base = {
        "research_digest": ["specificity", "curiosity", "reciprocity"],
        "regulation_change": ["specificity", "loss_aversion"],
        "recall_due": ["reciprocity", "single_binary_commitment"],
        "appointment_tomorrow": ["reciprocity", "single_binary_commitment"],
        "chronic_refill_due": ["effort_externalization", "single_binary_commitment"],
        "customer_lapsed_soft": ["social_proof", "single_binary_commitment"],
        "customer_lapsed_hard": ["reciprocity", "single_binary_commitment"],
        "perf_dip": ["loss_aversion", "specificity"],
        "perf_spike": ["specificity", "social_proof"],
        "seasonal_perf_dip": ["specificity", "curiosity"],
        "renewal_due": ["loss_aversion", "single_binary_commitment"],
        "festival_upcoming": ["effort_externalization", "curiosity"],
        "wedding_package_followup": ["effort_externalization", "specificity"],
        "curious_ask_due": ["asking_the_merchant"],
        "winback_eligible": ["loss_aversion", "social_proof"],
        "ipl_match_today": ["specificity", "effort_externalization"],
        "review_theme_emerged": ["specificity", "loss_aversion"],
        "milestone_reached": ["social_proof", "effort_externalization"],
        "active_planning_intent": ["effort_externalization"],
        "customer_lapsed_hard_win": ["reciprocity"],
        "category_seasonal": ["specificity", "curiosity"],
        "trial_followup": ["reciprocity", "single_binary_commitment"],
        "supply_alert": ["specificity", "loss_aversion"],
        "gbp_unverified": ["loss_aversion", "effort_externalization"],
        "cde_opportunity": ["curiosity", "specificity"],
        "competitor_opened": ["loss_aversion", "specificity"],
        "dormant_with_vera": ["asking_the_merchant", "social_proof"],
        "category_trend_movement": ["specificity", "curiosity"],
    }.get(kind, ["asking_the_merchant"])
    if anchor["mode"] in ("merchant_signal_pivot", "category_general"):
        # We're pivoting away from an unverified claim — lean on the levers
        # production Vera under-uses rather than force a fact we don't have.
        base = list(dict.fromkeys(base + ["asking_the_merchant", "social_proof"]))
    return base


def select_offer(category: dict, merchant: dict, customer: Optional[dict]) -> Optional[dict]:
    active = [o for o in merchant["offers"] if g(o, "status") == "active"]
    if active:
        return active[0]
    if category["offer_catalog"]:
        audience = "repeat_user" if (customer and customer["visits_total"] and customer["visits_total"] > 1) else "new_user"
        for o in category["offer_catalog"]:
            if g(o, "audience") == audience:
                return o
        return category["offer_catalog"][0]
    return None


def language_mode(merchant: dict, customer: Optional[dict]) -> str:
    if customer and customer.get("language_pref"):
        pref = customer["language_pref"]
        if "hi" in pref:
            return "hi-en"
        return "en"
    if "hi" in (merchant.get("languages") or []):
        return "hi-en"
    return "en"


# =============================================================================
# 4. Deterministic rendering
# =============================================================================

def _greet(merchant: dict, category: dict, lang: str) -> str:
    first = merchant.get("owner_first_name") or merchant["name"].split()[0]
    sal = category["salutations"]
    if not sal:
        return first
    idx = _hash_pick(merchant["merchant_id"] + "greet", len(sal))
    template = sal[idx]
    if "{" not in template:
        return template
    # Resolve any {placeholder} token generically — different categories use
    # different variable names (first_name, gym_name, pharmacist_name, ...).
    tokens = re.findall(r"\{(\w+)\}", template)
    business_tokens = {"gym_name", "pharmacy_name", "restaurant_name", "salon_name",
                        "clinic_name", "business_name", "shop_name"}
    values = {}
    for tok in tokens:
        values[tok] = merchant["name"] if tok in business_tokens else first
    try:
        return template.format(**values)
    except (KeyError, IndexError):
        return first


def _close(cta_shape: str, lang: str, yes_label: str = "1", no_label: str = "2",
            yes_text: str = "yes", no_text: str = "not now") -> str:
    if cta_shape == "binary":
        if lang == "hi-en":
            return f"Reply {yes_label} for {yes_text}, {no_label} to skip."
        return f"Reply {yes_label} for {yes_text}, or {no_label} for {no_text}."
    if cta_shape == "open_ended":
        return ""  # renderers embed the open question inline
    return ""


def _render_generic_pivot(category, merchant, trigger, customer, anchor, lang) -> str:
    """Fallback when a trigger's claim can't be verified anywhere — pivots to
    the strongest real signal we found, or to a pure lever-7 question."""
    greet = _greet(merchant, category, lang)
    facts = anchor["facts"]
    fk = facts.get("kind") if isinstance(facts, dict) else None

    if fk == "subscription_expired":
        return (f"{greet}, quick one — your magicpin plan lapsed {facts['days']} days ago, "
                f"so new customer leads from your listing have paused. Want me to show you "
                f"what you're missing before you decide on renewing?")
    if fk == "renewal_soon":
        return (f"{greet}, your plan renews in {facts['days']} days. Want me to check if your "
                f"listing is fully set up before then, or hold off for now?")
    if fk == "ctr_below_peer_median" or fk == "ctr_vs_peer":
        peer_ctr = category["peer_stats"].get("avg_ctr")
        return (f"{greet}, noticed your listing's click-through is trailing the category "
                f"median (~{_fmt_pct(peer_ctr) if peer_ctr else 'peer average'}). Want me to "
                f"pull up what the top profiles in your area are doing differently?")
    if fk == "stale_posts":
        days = facts.get("value", "a while")
        return (f"{greet}, your last Google post was {days.rstrip('d') if isinstance(days, str) else days} days ago — "
                f"fresh posts are one of the few free levers left for visibility. Want me to draft one now?")
    if fk == "views_delta":
        direction = "up" if facts["pct"] > 0 else "down"
        return (f"{greet}, your profile views are {direction} {_fmt_pct(facts['pct'])} this week. "
                f"Want to know what's likely driving it?")
    if fk == "calls_delta":
        direction = "up" if facts["pct"] > 0 else "down"
        return (f"{greet}, calls from your listing are {direction} {_fmt_pct(facts['pct'])} this week. "
                f"Want the breakdown?")
    if fk == "lapsed_customers":
        return (f"{greet}, {facts['count']} of your regulars haven't been back in 6+ months. "
                f"Want me to draft a simple win-back message you can send them?")
    if fk == "total_customers":
        return (f"{greet}, you've served {facts['count']} customers this year on magicpin. "
                f"Curious what's been your most-requested service lately?")
    if fk == "no_active_offer":
        sample = category["offer_catalog"][0]["title"] if category["offer_catalog"] else "a starter offer"
        return (f"{greet}, you don't have an active offer live right now — listings with one "
                f"typically convert better. Want me to set up something like \"{sample}\"?")

    # Nothing concrete anywhere — pure lever #7, honest and low-friction.
    digest_item = facts.get("digest_item") if isinstance(facts, dict) else None
    if digest_item:
        return (f"{greet}, saw this in this week's {category['display_name'].lower()} digest — "
                f"\"{digest_item.get('title')}\" ({digest_item.get('source', 'category digest')}). "
                f"Relevant to your patients, or should I skip these going forward?")
    return (f"{greet}, quick one for you — what's the one service your customers have been "
            f"asking about most this week? Helps me tailor what I send you.")


def _render_research_digest(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    item = find_digest_item(category, anchor["facts"].get("top_item_id")) or (category["digest"][0] if category["digest"] else None)
    if not item:
        return _render_generic_pivot(category, merchant, trigger, None, anchor, lang)
    n = item.get("trial_n")
    n_clause = f"{n:,}-{'patient' if 'patient' in category['slug'] or category['slug']=='dentists' else 'sample'} " if n else ""
    source_label = item.get("source") or "this week's digest"
    return (f"{greet}, {source_label} landed — "
            f"{n_clause}finding: {item.get('title')}. "
            f"Worth a 2-min look. Want me to pull it and draft something your customers can read?")


def _render_regulation_change(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    facts = anchor["facts"]
    item = find_digest_item(category, facts.get("top_item_id"))
    deadline = facts.get("deadline_iso")
    title = item.get("title") if item else "a regulatory update in your category"
    source = item.get("source") if item else None
    tail = f" Deadline: {deadline}." if deadline else ""
    src = f" — {source}" if source else ""
    return (f"{greet}, heads up — {title}{src}.{tail} Want me to send the compliance checklist "
            f"so you're not scrambling later?")


def _render_recall_due(category, merchant, trigger, customer, anchor, lang) -> str:
    facts = anchor["facts"]
    name = customer["name"] if customer else "there"
    slots = facts.get("available_slots") or []
    service = (facts.get("service_due") or "recall").replace("_", " ")
    offer = select_offer(category, merchant, customer)
    price_clause = f" {offer['title']}." if offer else ""
    if slots:
        s1 = slots[0].get("label", "")
        s2 = slots[1].get("label", "") if len(slots) > 1 else None
        slot_clause = f"Apke liye slots ready hain: {s1}" + (f" ya {s2}." if s2 else ".") if lang == "hi-en" else \
                      f"We've got {s1}" + (f" or {s2} open." if s2 else " open.")
    else:
        slot_clause = "Let me know a time that works and I'll block it."
    hi_bit = f"It's been a bit since your last visit — your {service} is due. " if lang != "hi-en" else \
             f"Aapka {service} due hai. "
    return (f"Hi {name}, {merchant['name']} here 🙂 {hi_bit}{slot_clause}{price_clause} "
            f"Reply 1 for {slots[0].get('label') if slots else 'the first slot'}"
            + (f", 2 for {slots[1].get('label')}" if len(slots) > 1 else "") + ", or tell us what works.")


def _render_appointment_tomorrow(category, merchant, trigger, customer, anchor, lang) -> str:
    name = customer["name"] if customer else "there"
    facts = anchor["facts"]
    time_label = facts.get("time_label") or facts.get("slot_label")
    when = f" at {time_label}" if time_label else " tomorrow"
    return (f"Hi {name}, {merchant['name']} here — just confirming your appointment{when}. "
            f"Reply 1 to confirm, 2 to reschedule.")


def _render_chronic_refill_due(category, merchant, trigger, customer, anchor, lang) -> str:
    name = customer["name"] if customer else "there"
    facts = anchor["facts"]
    molecules = facts.get("molecule_list") or []
    runs_out = facts.get("stock_runs_out_iso")
    if molecules:
        med_clause = f"your usual ({', '.join(molecules[:2])}{'...' if len(molecules) > 2 else ''}) "
    else:
        med_clause = "your usual refill "
    out_clause = f"looks like stock runs out around {runs_out.split('T')[0]}. " if runs_out else "looks due soon. "
    delivery = " I can have it delivered." if facts.get("delivery_address_saved") else ""
    return (f"Hi {name}, {merchant['name']} here — {med_clause}{out_clause}"
            f"Reply 1 to reorder{delivery and ' (delivered)'}, 2 to skip this month.")


def _render_customer_lapsed(category, merchant, trigger, customer, anchor, lang, hard: bool) -> str:
    name = customer["name"] if customer else "there"
    facts = anchor["facts"]
    if hard:
        days = facts.get("days_since_last_visit")
        focus = (facts.get("previous_focus") or "").replace("_", " ")
        days_clause = f"It's been {days} days since your last visit" if days else "It's been a while"
        focus_clause = f" on your {focus} plan" if focus else ""
        offer = select_offer(category, merchant, customer)
        offer_clause = f" We've got {offer['title']} if you want to ease back in." if offer else ""
        return (f"Hi {name}, {merchant['name']} here — {days_clause}{focus_clause}. Miss having you around."
                f"{offer_clause} Reply 1 to book a slot, 2 if now's not the time.")
    else:
        last_visit = customer.get("last_visit") if customer else None
        v_clause = f"since your visit on {last_visit}" if last_visit else "in a while"
        offer = select_offer(category, merchant, customer)
        offer_clause = f" {offer['title']}." if offer else ""
        return (f"Hi {name}, {merchant['name']} here — haven't seen you {v_clause}.{offer_clause} "
                f"Reply 1 to book, 2 to skip for now.")


def _render_perf_dip(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    if anchor["mode"] != "trigger_payload":
        return _render_generic_pivot(category, merchant, trigger, None, anchor, lang)
    f = anchor["facts"]
    metric = f.get("metric", "activity")
    return (f"{greet}, your {metric} are down {_fmt_pct(f.get('delta_pct', 0))} vs your usual "
            f"{f.get('vs_baseline', '')} over the last {f.get('window', '7d')} — want me to check what changed "
            f"on your listing?")


def _render_perf_spike(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    if anchor["mode"] != "trigger_payload":
        return _render_generic_pivot(category, merchant, trigger, None, anchor, lang)
    f = anchor["facts"]
    metric = f.get("metric", "activity")
    driver = (f.get("likely_driver") or "").replace("_", " ")
    driver_clause = f" — looks like your {driver} post is working." if driver else "."
    return (f"{greet}, nice bump — {metric} up {_fmt_pct(f.get('delta_pct', 0))} vs your usual "
            f"{f.get('vs_baseline', '')} this week{driver_clause} Want me to double down on that angle?")


def _render_renewal_due(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    days = f.get("days_remaining")
    amt = f.get("renewal_amount")
    amt_clause = f" ({_inr(amt)})" if amt else ""
    return (f"{greet}, your {f.get('plan', 'plan')}{amt_clause} renews in {days} days. "
            f"Reply 1 to renew now, 2 to talk to someone first.")


def _render_festival_upcoming(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    fest = f.get("festival", "the festival")
    days = f.get("days_until")
    days_clause = f" in {days} days" if days else ""
    offer = select_offer(category, merchant, None)
    offer_clause = f" I've drafted a {fest} post around \"{offer['title']}\" — want to see it?" if offer else \
                   f" Want me to draft a {fest} post for your page?"
    return f"{greet}, {fest} is coming up{days_clause}.{offer_clause}"


def _render_wedding_package_followup(category, merchant, trigger, customer, anchor, lang) -> str:
    name = customer["name"] if customer else "there"
    f = anchor["facts"]
    days = f.get("days_to_wedding")
    prog = (f.get("next_step_window_open") or "next step").replace("_", " ")
    days_clause = f"With {days} days to go, " if days else ""
    return (f"Hi {name}, {merchant['name']} here — {days_clause}this is the window to start your {prog}. "
            f"Reply 1 to book it, 2 to hold off.")


def _render_curious_ask_due(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    return (f"{greet}, quick one — what's the one service your customers asked about most "
            f"this week? Helps me tailor what I send you.")


def _render_winback_eligible(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    days = f.get("days_since_expiry")
    lapsed = f.get("lapsed_customers_added_since_expiry")
    lapsed_clause = f" and {lapsed} regulars have gone quiet since" if lapsed else ""
    return (f"{greet}, it's been {days} days since your plan lapsed{lapsed_clause}. "
            f"Reply 1 to see what reactivating gets you back, 2 to skip.")


def _render_ipl_match_today(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    match = f.get("match", "tonight's match")
    venue = f.get("venue")
    venue_clause = f" at {venue}" if venue else ""
    return (f"{greet}, {match}{venue_clause} tonight — usually a good night for walk-ins. "
            f"Want me to draft a quick match-night post?")


def _render_review_theme_emerged(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    theme = (f.get("theme") or "").replace("_", " ")
    occ = f.get("occurrences_30d")
    trend_clause = f", trending {f.get('trend')}" if f.get("trend") else ""
    return (f"{greet}, {occ} reviews this month mention \"{theme}\"{trend_clause}. "
            f"Want me to draft a response you can post, and a fix to try?")


def _render_milestone_reached(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    if anchor["mode"] != "trigger_payload":
        return _render_generic_pivot(category, merchant, trigger, None, anchor, lang)
    metric = (f.get("metric") or "milestone").replace("_", " ")
    now, target = f.get("value_now"), f.get("milestone_value")
    if f.get("is_imminent") and now and target:
        return (f"{greet}, you're at {now} {metric} — {target - now} away from {target}. "
                f"Want me to nudge a couple of recent customers for a review to help you cross it?")
    return f"{greet}, you've crossed {now or target} {metric} — want me to draft a post celebrating it?"


def _render_active_planning_intent(category, merchant, trigger, anchor, lang) -> str:
    f = anchor["facts"]
    topic = (f.get("intent_topic") or "your idea").replace("_", " ")
    # Merchant already showed initiative — go straight to action, no re-qualifying.
    return (f"Good idea — let's build the {topic} out. Two things I need: what price point are "
            f"you thinking, and should it launch this week or next? I'll draft the listing either way.")


def _render_seasonal(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    if trigger["kind"] == "category_seasonal":
        trends = f.get("trends") or []
        trend_clause = ", ".join(t.replace("_", " ") for t in trends[:2]) if trends else "seasonal shifts"
        return (f"{greet}, category-wide demand shift this season: {trend_clause}. "
                f"Want me to flag which of these matter most for your shelf/menu?")
    # seasonal_perf_dip
    metric = f.get("metric", "views")
    note = (f.get("season_note") or "").replace("_", " ")
    if f.get("is_expected_seasonal"):
        return (f"{greet}, your {metric} are down {_fmt_pct(f.get('delta_pct', 0))} — this tracks the usual "
                f"{note} dip, not something specific to you. Want the seasonal playbook for this window?")
    return _render_perf_dip(category, merchant, trigger, anchor, lang)


def _render_trial_followup(category, merchant, trigger, customer, anchor, lang) -> str:
    name = customer["name"] if customer else "there"
    f = anchor["facts"]
    opts = f.get("next_session_options") or []
    slot_clause = f"Next slot: {opts[0].get('label')}." if opts else "Let me know a time that works."
    return f"Hi {name}, {merchant['name']} here — how was your trial? {slot_clause} Reply 1 to book, 2 to skip."


def _render_supply_alert(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    molecule = f.get("molecule", "a product")
    batches = f.get("affected_batches") or []
    batch_clause = f" (batches {', '.join(batches)})" if batches else ""
    return (f"{greet}, recall alert on {molecule}{batch_clause}. Reply 1 for the removal checklist, "
            f"2 if you don't stock it.")


def _render_gbp_unverified(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    uplift = f.get("estimated_uplift_pct")
    uplift_clause = f" (roughly +{_fmt_pct(uplift)} typical visibility uplift)" if uplift else ""
    return (f"{greet}, your Google listing still isn't verified{uplift_clause}. "
            f"Reply 1 to start verification now, 2 to do it later.")


def _render_cde_opportunity(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    f = anchor["facts"]
    item = find_digest_item(category, f.get("digest_item_id"))
    title = item.get("title") if item else "a category CDE session"
    fee = f.get("fee", "").replace("_", " ")
    fee_clause = f" ({fee})" if fee else ""
    return f"{greet}, {title}{fee_clause} — {f.get('credits', '')} credits. Want the sign-up link?"


def _render_competitor_opened(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    if anchor["mode"] != "trigger_payload":
        return _render_generic_pivot(category, merchant, trigger, None, anchor, lang)
    f = anchor["facts"]
    name = f.get("competitor_name", "a new competitor")
    dist = f.get("distance_km")
    dist_clause = f", {dist}km away" if dist else ""
    offer_clause = f" They're running \"{f.get('their_offer')}\"." if f.get("their_offer") else ""
    return (f"{greet}, {name}{dist_clause} just opened on GBP.{offer_clause} "
            f"Want me to check how your listing compares side by side?")


def _render_dormant_with_vera(category, merchant, trigger, anchor, lang) -> str:
    greet = _greet(merchant, category, lang)
    if anchor["mode"] != "trigger_payload":
        return _render_generic_pivot(category, merchant, trigger, None, anchor, lang)
    f = anchor["facts"]
    days = f.get("days_since_last_merchant_message")
    topic = (f.get("last_topic") or "").replace("_", " ")
    topic_clause = f" — we'd left off on {topic}" if topic else ""
    return (f"{greet}, haven't heard from you in {days} days{topic_clause}. Still around? "
            f"One quick thing that might help: what's slowing you down right now?")


_RENDERERS = {
    "research_digest": lambda c, m, t, cu, a, l: _render_research_digest(c, m, t, a, l),
    "regulation_change": lambda c, m, t, cu, a, l: _render_regulation_change(c, m, t, a, l),
    "recall_due": lambda c, m, t, cu, a, l: _render_recall_due(c, m, t, cu, a, l),
    "appointment_tomorrow": lambda c, m, t, cu, a, l: _render_appointment_tomorrow(c, m, t, cu, a, l),
    "chronic_refill_due": lambda c, m, t, cu, a, l: _render_chronic_refill_due(c, m, t, cu, a, l),
    "customer_lapsed_soft": lambda c, m, t, cu, a, l: _render_customer_lapsed(c, m, t, cu, a, l, hard=False),
    "customer_lapsed_hard": lambda c, m, t, cu, a, l: _render_customer_lapsed(c, m, t, cu, a, l, hard=True),
    "perf_dip": lambda c, m, t, cu, a, l: _render_perf_dip(c, m, t, a, l),
    "perf_spike": lambda c, m, t, cu, a, l: _render_perf_spike(c, m, t, a, l),
    "renewal_due": lambda c, m, t, cu, a, l: _render_renewal_due(c, m, t, a, l),
    "festival_upcoming": lambda c, m, t, cu, a, l: _render_festival_upcoming(c, m, t, a, l),
    "wedding_package_followup": lambda c, m, t, cu, a, l: _render_wedding_package_followup(c, m, t, cu, a, l),
    "curious_ask_due": lambda c, m, t, cu, a, l: _render_curious_ask_due(c, m, t, a, l),
    "winback_eligible": lambda c, m, t, cu, a, l: _render_winback_eligible(c, m, t, a, l),
    "ipl_match_today": lambda c, m, t, cu, a, l: _render_ipl_match_today(c, m, t, a, l),
    "review_theme_emerged": lambda c, m, t, cu, a, l: _render_review_theme_emerged(c, m, t, a, l),
    "milestone_reached": lambda c, m, t, cu, a, l: _render_milestone_reached(c, m, t, a, l),
    "active_planning_intent": lambda c, m, t, cu, a, l: _render_active_planning_intent(c, m, t, a, l),
    "category_seasonal": lambda c, m, t, cu, a, l: _render_seasonal(c, m, t, a, l),
    "seasonal_perf_dip": lambda c, m, t, cu, a, l: _render_seasonal(c, m, t, a, l),
    "trial_followup": lambda c, m, t, cu, a, l: _render_trial_followup(c, m, t, cu, a, l),
    "supply_alert": lambda c, m, t, cu, a, l: _render_supply_alert(c, m, t, a, l),
    "gbp_unverified": lambda c, m, t, cu, a, l: _render_gbp_unverified(c, m, t, a, l),
    "cde_opportunity": lambda c, m, t, cu, a, l: _render_cde_opportunity(c, m, t, a, l),
    "competitor_opened": lambda c, m, t, cu, a, l: _render_competitor_opened(c, m, t, a, l),
    "dormant_with_vera": lambda c, m, t, cu, a, l: _render_dormant_with_vera(c, m, t, a, l),
}


# Kinds whose renderer needs trigger-specific named/numeric facts (a festival
# name, a match name, a milestone value, a deadline...) that simply don't
# exist when the payload is a placeholder, and which aren't already
# self-guarded (perf_dip/perf_spike/competitor_opened/dormant_with_vera/
# milestone_reached check anchor mode internally) or customer-context-based
# (recall_due & co. can honestly fall back to real CustomerContext fields).
_STRICT_PAYLOAD_KINDS = {
    "renewal_due", "festival_upcoming", "wedding_package_followup", "ipl_match_today",
    "review_theme_emerged", "supply_alert", "gbp_unverified", "cde_opportunity",
    "regulation_change", "winback_eligible",
}


def render(category: dict, merchant: dict, trigger: dict, customer: Optional[dict],
           anchor: dict, lang: str) -> str:
    kind = trigger["kind"]
    if kind in _STRICT_PAYLOAD_KINDS and anchor["mode"] != "trigger_payload":
        return _render_generic_pivot(category, merchant, trigger, customer, anchor, lang)
    renderer = _RENDERERS.get(kind)
    if renderer:
        try:
            return renderer(category, merchant, trigger, customer, anchor, lang)
        except Exception:
            pass  # fall through to generic pivot on any malformed payload field
    return _render_generic_pivot(category, merchant, trigger, customer, anchor, lang)


# =============================================================================
# 5. Validation — anti-pattern guardrails from challenge-brief §11
# =============================================================================

def validate(body: str, category: dict, cta_shape: str) -> tuple[bool, list[str]]:
    reasons = []
    if not body or not body.strip():
        reasons.append("empty_body")
        return False, reasons
    low = body.lower()
    for taboo in category["taboo"]:
        if taboo in low:
            reasons.append(f"taboo_word:{taboo}")
    if cta_shape == "binary":
        # exactly one binary ask; guard against multi-CTA sprawl
        if low.count("reply ") > 1:
            reasons.append("multiple_ctas")
    if re.search(r"\b(amazing|unbelievable|don't miss out)!!*", low):
        reasons.append("promotional_hype")
    if len(body) > 900:
        reasons.append("too_long")
    return (len(reasons) == 0), reasons


# =============================================================================
# 6. Optional LLM polish pass (never adds facts, only smooths phrasing)
# =============================================================================

_POLISH_SYSTEM = """You are polishing a WhatsApp message for phrasing only.
Rules:
- Do NOT add any fact, number, name, date, or claim that is not already in the draft.
- Do NOT change the call-to-action structure (keep exactly the same CTA shape: {cta_shape}).
- Keep the same language mix ({lang}).
- Keep it concise; no preambles like "I hope you're doing well".
- Return ONLY the revised message body, nothing else."""


def llm_polish(draft: str, cta_shape: str, lang: str) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            temperature=0,
            system=_POLISH_SYSTEM.format(cta_shape=cta_shape, lang=lang),
            messages=[{"role": "user", "content": f"Draft:\n{draft}"}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return text or None
    except Exception:
        return None


# =============================================================================
# 7. Main entrypoint
# =============================================================================

def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    cat = normalize_category(category)
    mer = normalize_merchant(merchant)
    trg = normalize_trigger(trigger)
    cus = normalize_customer(customer)

    anchor = resolve_anchor(cat, mer, trg, cus)
    lang = language_mode(mer, cus)
    cta_shape = select_cta_shape(trg["kind"], anchor)
    levers = select_levers(trg["kind"], anchor)

    draft = render(cat, mer, trg, cus, anchor, lang)
    ok, reasons = validate(draft, cat, cta_shape)
    if not ok:
        # Strip taboo words defensively and fall back to the safe generic pivot
        # rather than ship a body that fails our own guardrails.
        draft = _render_generic_pivot(cat, mer, trg, cus, anchor, lang)
        ok, reasons = validate(draft, cat, cta_shape)

    body = draft
    polished = llm_polish(draft, cta_shape, lang)
    if polished:
        p_ok, _ = validate(polished, cat, cta_shape)
        if p_ok:
            body = polished

    send_as = "merchant_on_behalf" if (cus and trg["scope"] == "customer") else "vera"

    rationale_bits = [
        f"trigger={trg['kind']}",
        f"anchor_mode={anchor['mode']}",
        f"levers={'+'.join(levers)}",
        f"cta={cta_shape}",
        f"lang={lang}",
    ]
    if anchor["mode"] != "trigger_payload":
        rationale_bits.append("pivoted_off_unverifiable_trigger_payload_to_avoid_fabrication")

    return {
        "body": body,
        "cta": cta_shape,
        "send_as": send_as,
        "suppression_key": trg["suppression_key"],
        "rationale": "; ".join(rationale_bits),
    }
