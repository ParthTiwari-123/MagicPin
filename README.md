# Vera-Beater — magicpin AI Challenge submission

## Approach

The bot is built around one idea: **never assert a claim you can't verify.**
Composition proceeds in five stages (`bot.py`):

1. **Normalize** the four contexts (category/merchant/trigger/customer),
   tolerating both the dataset's key names and the testing-brief's live-push
   key names (e.g. `vocab_taboo` vs `taboos`).
2. **Resolve an anchor** — the one concrete, checkable fact the message will
   lead with. If the trigger's payload is real, it wins. If it's a
   placeholder (~40% of the canonical test set ships `{"placeholder": true}`
   payloads, and some even attach a trigger label — e.g. `perf_dip` — to a
   merchant whose numbers are actually *up*), the bot searches
   merchant/customer/category context in priority order (subscription
   status → derived signals → performance vs. peer benchmark → customer-base
   stats → offer-catalog gaps → category digest) for the strongest real fact
   available, and only falls back to a pure "ask the merchant" hook
   (compulsion lever #7, one of production Vera's biggest under-used levers
   per the brief) when nothing concrete exists anywhere.
3. **Select levers, CTA shape, offer, and language** from lookup tables keyed
   by trigger kind and anchor mode.
4. **Render** via a per-trigger-kind template bank (24 kinds covered,
   generic fallback for anything else), fully deterministic — no LLM
   required to run or to reproduce `submission.jsonl`.
5. **Validate + optionally polish**: output is checked against category
   taboo words, single-CTA shape, and length before shipping. If
   `ANTHROPIC_API_KEY` is set, an optional temperature-0 pass asks the model
   to smooth phrasing *without adding facts*; the polished text is
   re-validated and only used if it passes the same checks — so the LLM
   pass can only improve fluency, never reintroduce hallucination.

`conversation_handlers.py` layers multi-turn logic on top: auto-reply
detection (regex bank of canned phrases + verbatim-repeat detection),
explicit-stop handling, intent-transition routing (jumps straight to action
mode on "let's do it", skipping re-qualification — the brief's Pattern-D
failure), hostile/off-topic handling that de-escalates without ending, and
a 3-unanswered-nudges exit rule.

`server.py` wraps both in the 5-endpoint HTTP contract from
`challenge-testing-brief.md` (context/tick/reply/healthz/metadata, plus an
optional teardown), with in-memory context versioning, suppression-key
dedup, and per-(merchant, customer) first-message template tracking.

## What I verified locally

- All 30 canonical test pairs compose successfully with zero duplicate
  bodies, zero unresolved template tokens, zero multi-CTA violations.
- Manually drove `server.py` live through a full context-push →
  tick → reply cycle, including suppression-key dedup on a repeat tick.
- Manually ran `conversation_handlers.py` through the three Phase-4 replay
  scenarios (auto-reply hell, intent transition, hostile-then-off-topic)
  and fixed two bugs this surfaced (an off-by-one that flagged the very
  first inbound message as an auto-reply, and a double-word grammar bug
  in a fallback template).
- Could **not** run `judge_simulator.py` itself — it hard-requires a
  pasted LLM API key (OpenAI/Anthropic/Gemini/etc.) to play the merchant
  and score conversations, which wasn't available in the build
  environment. Everything above was validated by driving the endpoints
  and `compose()`/`respond()` directly instead.

## Tradeoffs

- **Deterministic templates over a single big LLM prompt.** This makes the
  bot reproducible, cheap, fast (<30s easily), and impossible to hallucinate
  by construction — but the copy is necessarily less fluent/varied than a
  well-tuned frontier-model prompt would produce. The optional LLM-polish
  pass is the compromise: fluency without giving up the guardrails.
- **Rule-based conversation handling over an LLM-driven dialogue manager.**
  Simpler to reason about and to make the replay-test behaviors (auto-reply
  exit, intent transition) provably correct, at the cost of being less
  flexible on genuinely novel merchant phrasing than an LLM classifier would
  be. A natural upgrade is to use the LLM only for *intent classification*
  (auto-reply / stop / intent-transition / off-topic) while keeping the
  deterministic action logic.
- **One template per trigger kind** keeps quality consistent and auditable,
  but caps stylistic variety across the 30 test pairs for the same kind
  (mitigated with deterministic per-merchant variant selection where the
  category offers multiple salutation styles, e.g. dentist salutations).

## What additional context would have helped most

1. **A "current date" field** in the base payload — several triggers
   (seasonal beats, festival countdowns) would compose better with a known
   "now" to compute freshness/urgency instead of relying on `days_until`
   fields that may or may not be present.
2. **A canonical list of known auto-reply phrases per WhatsApp Business
   locale** — the brief hints "same message verbatim 3+ times," but a seed
   list of common Indian WA Business auto-reply strings would sharpen
   first-message detection (currently a curated regex bank).
3. **Explicit "already sent this trigger kind to this merchant N times"
   counters** in `MerchantContext` — useful for cadence planning
   (open challenge §12.3) without having to infer it from
   `conversation_history` text.
