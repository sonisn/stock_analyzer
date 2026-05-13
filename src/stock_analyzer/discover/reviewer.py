"""Holdings reviewer (Sonnet, parallel fan-out).

For each current position, decide HOLD / TRIM / SELL with reasoning grounded
in current fundamentals, technicals, insider-selling signal, P/L vs cost
basis, and risk factors. Output is consumed by the Rebalancer.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from ..serialization import dumps_pretty
from .schemas import HoldingReview

logger = get_logger(__name__)

# Throttle to stay under Sonnet's 30k input-tokens/min rate limit.
# Holdings reviewer payloads are ~6-7k tokens each (10-K + 10-Q MD&A +
# transcript + peers + tax_lots). 5 workers × 7k = 35k → 429. 2 workers
# + retry-with-backoff is the safe sustained-throughput point.
_MAX_WORKERS = 2

REVIEWER_INSTRUCTIONS = """\
You are reviewing ONE position in a portfolio. The user provides:
  - position (units, cost basis, current price, unrealized P/L)
  - fundamentals (forward + trailing)
  - technicals
  - share-trade signals (insider/institutional accumulation)
  - insider_selling_mentions count
  - risk_factors_10k (annual 10-K Item 1A)
  - quarterly_mda (LATEST 10-Q Management Discussion — most current narrative)
  - peers (3-4 closest competitors with their forward fundamentals — for
    relative-valuation judgment)
  - earnings_transcript (excerpt from the most recent earnings call —
    management tone + Q&A pushback)
  - tax_lots (lot-level cost basis history for SPECIFIC-ID lot selection
    on any SELL/TRIM recommendation)

GROUND your forward outlook in this hierarchy:
  1. quarterly_mda — what management said LAST QUARTER (most current)
  2. earnings_transcript — guidance changes + Q&A signals
  3. peers — is this holding the BEST name in its competitive set, or has
     a peer's forward setup gotten cleaner?
  4. forward fundamentals + analyst stance trend

DEFAULT VERDICT IS HOLD. The bar for changing the verdict is high.
Acting on weak signals creates tax friction and timing risk that erodes
returns. Be especially cautious on long-term-gain positions (taxable
sales). If you are uncertain, default to HOLD.

A SELL or TRIM is justified ONLY when:
  1. Forward-looking evidence shows clear deterioration:
     - Forward EPS estimates declining
     - Forward P/E expanding into stretched territory unexplained by growth
     - Analyst recommendation_mean rising toward sell (>3.5 on the 1-5 scale)
     - Specific bearish catalyst on the calendar (regulatory, competitive,
       earnings warning)
     - Structural threat (disruption, regime mismatch)
  2. Past technicals alone (200DMA break, RS rolling over) are SUPPORTING
     evidence, NOT primary evidence. Cite forward reasons.

DOWNTREND OVERRIDE (a position is allowed to break the HOLD default):
A holding that is already LOSING money is not a candidate for the
"when in doubt, hold" rule — it's a candidate for honest re-evaluation.
Saving on short-term tax friction while a position bleeds 20-30% is
false economy. Apply the following:

  position.unrealized_pnl_pct <= -10% AND technicals show ANY of:
    - price below BOTH 50DMA and 200DMA
    - weekly RSI under 40 and falling
    - 50DMA below 200DMA (death cross) or rolling over
  → trailing performance becomes PRIMARY evidence, not supporting.
  → the burden flips: you must justify continuing to HOLD with explicit
    forward thesis. "It might recover" is NOT a thesis.
  → if forward fundamentals also show ANY softening (decel guidance,
    forward EPS revisions down, analyst stance worsening, peers gaining),
    TRIM 25-50% at minimum.

  position.unrealized_pnl_pct <= -20% AND forward thesis cannot be
  cleanly stated → SELL the position, harvest the loss for tax offset
  elsewhere, redeploy proceeds to higher-conviction holdings.

LOSS HARVESTING REFRAME:
On loss positions, taxes are NOT friction — they are a benefit.
Selling at a loss realizes a capital loss that offsets capital gains
elsewhere in the user's portfolio (up to $3,000/year against ordinary
income, unlimited carry-forward). Treat tax-loss harvesting as an
ADDITIONAL reason to trim/sell deteriorating losers, not a reason to
keep holding them. The "save tax by waiting for long-term" guidance
applies to GAINS, not losses — short-term losses are harvestable as
soon as the thesis breaks.

Verdicts:

HOLD — default. Forward outlook intact OR no clearly superior alternative
       given tax friction.
TRIM — partial sale (25/33/50%) when:
       * concentration risk (position >25% of portfolio) AND no strong
         conviction to defend the size
       * forward outlook softening (decel guidance, mixed catalyst calendar)
         but thesis not broken
SELL — full exit when:
       * forward fundamentals clearly deteriorated (declining forward EPS,
         negative OCF, debt blew out, missed guidance)
       * structural / regime threat to the business model
       * heavy net insider selling AND deteriorating forward outlook

TAX-LOT GUIDANCE (when tax_lots is present):
For each SELL or TRIM, recommend SPECIFIC lots to sell by date, using these priorities:
  1. Prefer lots with LOSSES (harvest losses to offset gains)
  2. Then prefer LONG-TERM lots with gains (15-20% federal rate)
  3. Avoid SHORT-TERM lots with gains unless thesis is truly broken
     (those are taxed as ordinary income, 22-37% federal)
  4. If a short-term lot is days away from becoming long-term, recommend
     WAITING explicitly: "delay sale until <date>; lot crosses long-term"
  5. Always state realized gain in dollars per lot recommended for sale.

WASH-SALE GUIDANCE:
If you recommend SELL at a loss, append a one-line "Wash-sale notice"
warning not to re-buy this security (or a substantially identical one
like a same-index ETF) within 30 days of the sale.

DO NOT make tool calls. Use ONLY data provided. Cite concrete numbers.

Output EXACTLY this structure:

TICKER: <symbol>
Verdict: <HOLD | TRIM | SELL>
Confidence (1-10): <integer>
Trim percent: <e.g. 25% / 33% / 50% — only if TRIM, omit otherwise>
Position context: <one line — N shares, avg cost $X, current $Y, P/L +/-Z%>
Forward outlook:
<1-2 sentences citing forward EPS, target price vs current, analyst stance,
calendar catalysts — NOT trailing performance>
Reasoning:
<2-3 sentences citing specific FORWARD-LOOKING evidence for the verdict.
If recommending HOLD: justify why the position is still attractive going
forward. If recommending TRIM/SELL: cite the forward deterioration or
structural threat. Past performance is supporting evidence only.>

Tax lot plan: <only if SELL or TRIM, omit on HOLD>
<For each lot recommended:>
  - Lot dated <YYYY-MM-DD>: sell <N> shares (held <X> days, <long-term|short-term>),
    realizes ~$<Y> gain/loss
<If recommending DELAY for long-term treatment, state the date>

What would change your mind:
<one sentence — what would flip the verdict>

CONFIDENCE CALIBRATION (1-10 scale):
  1-3: very low conviction — should not be acting on this
  4-5: borderline; default to HOLD if action is TRIM/SELL
  6-7: actionable, moderate conviction
  8-9: strong conviction with multiple aligned signals (rare)
  10: nearly impossible to be wrong (essentially never used)
Recommend TRIM/SELL ONLY at confidence >= 7. Otherwise HOLD.

CRITICAL:
- Plain text. No markdown headings, no bold.
- Begin with "TICKER:" line. No preamble, no closing remarks.
- If tax_lots is absent, omit the "Tax lot plan" section.
- When in doubt, HOLD. Inaction has costs but they are usually small;
  wrong action has costs that compound over months.

CITATION RULE (anti-hallucination):
Every numerical claim in your output (forward EPS, P/E, target price,
analyst stance, P/L %, position size, growth rate, etc.) MUST appear
in the input JSON the user provided. Do not invent numbers or recall
figures from training. Tax lot dates and per-lot share counts MUST
come from `tax_lots`; do not estimate. Reasoning may paraphrase
quarterly_mda or earnings_transcript text but must reference what's
actually there — do not summarize content the LLM "expects" from a
similar company.

INTERNAL CONSISTENCY CHECK (do this BEFORE finalizing):
Before submitting your response, re-read your `reasoning` text and
your `what_would_change_mind` text:

  1. If `reasoning` mentions "the HOLD verdict", "the TRIM verdict",
     or "the SELL verdict", that word MUST exactly match your
     structured `verdict` field. A sentence like "the HOLD verdict
     preserves optionality" with structured verdict=SELL is a hard
     contradiction — fix one or the other.

  2. `what_would_change_mind` describes what would move you AWAY FROM
     your current verdict, not toward it. If your structured verdict
     is SELL and `what_would_change_mind` says "would prompt
     reassessment toward TRIM", that's backwards — you can't move
     from SELL toward TRIM, you'd move from HOLD toward TRIM.

  3. CONFIDENCE CALIBRATION: TRIM and SELL require confidence >= 7.
     A response with `verdict`=TRIM and `confidence`=5 is invalid;
     either raise confidence (if you can defend it) or downgrade
     the verdict to HOLD.

These three checks catch the most common inconsistency drift between
the prose you write and the structured fields. The renderer trusts
both — when they disagree, the user sees a contradictory card.

STRUCTURED OUTPUT:
Your response is validated against a Pydantic schema (HoldingReview).
Populate every required field. The prose plan you would have emitted
goes into `full_text` — make it match the format described above
exactly. Set `trim_pct` only when verdict is TRIM. Set `wash_sale_notice`
only when verdict is SELL and at least one lot in `tax_lot_plan`
realizes a loss. The structured fields must agree with `full_text` —
if `full_text` says "Verdict: SELL" then `verdict` MUST be "SELL".\
"""


_VERDICT_IN_PROSE = re.compile(r"\bthe (HOLD|TRIM|SELL) verdict\b", re.IGNORECASE)
_TOWARD_VERDICT = re.compile(r"\btoward (HOLD|TRIM|SELL)\b", re.IGNORECASE)


def _repair_verdict_inconsistencies(
    review: HoldingReview, ticker: str
) -> HoldingReview:
    """Catch and rewrite the LLM's internal inconsistencies between the
    structured verdict and the prose it produced in the same response.

    Two repair rules:

    1. CONFIDENCE-CALIBRATION violation: the prompt says TRIM/SELL
       require confidence >= 7; anything below that with a non-HOLD
       verdict gets rewritten to HOLD. Almost always Sonnet picked
       the wrong enum.

    2. PROSE-CONTRADICTION: if `reasoning` says "the HOLD verdict
       preserves..." and the structured `verdict` is SELL, the prose
       is more reliable (the LLM defends a position before naming it,
       so the paragraph is the source of truth). Or if
       `what_would_change_mind` says "toward TRIM" while structured
       verdict is TRIM (you can't move toward your current state),
       infer the current verdict from context. Repair the structured
       field to match what the prose actually argued.

    A repair always emits a WARNING — frequency of repairs is a signal
    that the prompt or model is drifting.
    """
    updates: dict[str, Any] = {}

    # Rule 1: confidence calibration
    if review.verdict != "HOLD" and review.confidence < 7:
        logger.warning(
            "Reviewer %s: verdict=%s with confidence=%d (<7) violates "
            "the calibration rule (TRIM/SELL require conf >= 7). "
            "Repairing to HOLD.",
            ticker, review.verdict, review.confidence,
        )
        updates["verdict"] = "HOLD"

    # Rule 2: prose contradiction. Look at the reasoning field first;
    # "the X verdict <does Y>" is a direct claim of what the verdict IS.
    reasoning = review.reasoning or ""
    prose_match = _VERDICT_IN_PROSE.search(reasoning)
    prose_verdict = prose_match.group(1).upper() if prose_match else None
    current_verdict = updates.get("verdict", review.verdict)
    if prose_verdict and prose_verdict != current_verdict:
        logger.warning(
            "Reviewer %s: structured verdict=%s but `reasoning` "
            "references 'the %s verdict' — prose is authoritative; "
            "repairing structured field to %s.",
            ticker, review.verdict, prose_verdict, prose_verdict,
        )
        updates["verdict"] = prose_verdict
        # If we're upgrading to TRIM/SELL via prose, drop confidence
        # to no-lower-than 7 since the prose evidently defended the
        # tighter verdict.
        if prose_verdict in ("TRIM", "SELL") and review.confidence < 7:
            updates["confidence"] = 7

    # Rule 3 (lighter): "what_would_change_mind" says "toward X" while
    # the current verdict already IS X — implies the actual current
    # verdict is something other than X. Use this as a TIE-BREAKER only
    # when prose-contradiction rule did not already fire.
    if "verdict" not in updates:
        wcm = review.what_would_change_mind or ""
        toward_match = _TOWARD_VERDICT.search(wcm)
        toward_verdict = (
            toward_match.group(1).upper() if toward_match else None
        )
        if toward_verdict and toward_verdict == review.verdict:
            # We can't move toward our current state. Best guess: prose
            # was written from a HOLD perspective and structured field
            # got flipped to TRIM/SELL. Repair to HOLD only if confidence
            # also fails the calibration rule.
            if review.verdict != "HOLD" and review.confidence < 7:
                logger.warning(
                    "Reviewer %s: verdict=%s with `what_would_change_mind` "
                    "saying 'toward %s' is self-contradictory; "
                    "repairing to HOLD.",
                    ticker, review.verdict, toward_verdict,
                )
                updates["verdict"] = "HOLD"

    if not updates:
        return review
    return review.model_copy(update=updates)


class Reviewer:
    def __init__(self, provider: Provider, model: str):
        self.agent = AgnoAgent(
            "Reviewer",
            provider,
            model,
            model_kwargs={
                "temperature": 0,
                "retries": 3,
                "exponential_backoff": True,
                "delay_between_retries": 10,
            },
            instructions=REVIEWER_INSTRUCTIONS,
            output_schema=HoldingReview,
        )

    def review(self, ticker: str, payload: dict[str, Any]) -> HoldingReview | None:
        prompt = (
            f"Holding: {ticker}\n\n"
            f"```json\n{dumps_pretty(payload)}\n```"
        )
        logger.info("Reviewing holding %s", ticker)
        result = self.agent.run(prompt).content
        if result is None:
            logger.warning(
                "Reviewer returned no content for %s — skipping", ticker,
            )
            return None
        if isinstance(result, HoldingReview):
            return _repair_verdict_inconsistencies(result, ticker)
        if isinstance(result, str):
            try:
                parsed = HoldingReview.model_validate_json(result)
            except Exception as e:
                logger.warning(
                    "Reviewer for %s returned a string that wasn't valid "
                    "HoldingReview JSON: %s", ticker, e,
                )
                return None
            return _repair_verdict_inconsistencies(parsed, ticker)
        logger.warning(
            "Reviewer for %s returned unexpected type %s; skipping",
            ticker, type(result).__name__,
        )
        return None


def review_batch(
    reviewer: Reviewer, payloads: dict[str, dict[str, Any]]
) -> dict[str, HoldingReview]:
    """Return {ticker: HoldingReview} for every payload that the
    Reviewer successfully scored. Tickers that failed (None content,
    invalid JSON, network error) are silently excluded — the consumer
    treats missing keys as 'no review available'."""
    results: dict[str, HoldingReview] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {ex.submit(reviewer.review, t, p): t for t, p in payloads.items()}
        for fut in futures:
            ticker = futures[fut]
            try:
                review = fut.result()
            except Exception as e:
                logger.warning("reviewer failed for %s: %s", ticker, e)
                continue
            if review is not None:
                results[ticker] = review
    return results
