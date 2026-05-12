"""Holdings reviewer (Sonnet, parallel fan-out).

For each current position, decide HOLD / TRIM / SELL with reasoning grounded
in current fundamentals, technicals, insider-selling signal, P/L vs cost
basis, and risk factors. Output is consumed by the Rebalancer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from ..serialization import dumps_pretty

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
  wrong action has costs that compound over months.\
"""


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
        )

    def review(self, ticker: str, payload: dict[str, Any]) -> str:
        prompt = (
            f"Holding: {ticker}\n\n"
            f"```json\n{dumps_pretty(payload)}\n```"
        )
        logger.info("Reviewing holding %s", ticker)
        content = self.agent.run(prompt).content
        if not content:
            logger.warning(
                "Reviewer returned empty content for %s — using empty review",
                ticker,
            )
            return ""
        return content


def review_batch(
    reviewer: Reviewer, payloads: dict[str, dict[str, Any]]
) -> dict[str, str]:
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {ex.submit(reviewer.review, t, p): t for t, p in payloads.items()}
        for fut in futures:
            ticker = futures[fut]
            try:
                results[ticker] = fut.result()
            except Exception as e:
                logger.warning("reviewer failed for %s: %s", ticker, e)
    return results
