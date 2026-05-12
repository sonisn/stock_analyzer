"""Holdings reviewer (Sonnet, parallel fan-out).

For each current position, decide HOLD / TRIM / SELL with reasoning grounded
in current fundamentals, technicals, insider-selling signal, P/L vs cost
basis, and risk factors. Output is consumed by the Rebalancer.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import AgnoAgent, Provider
from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 5

REVIEWER_INSTRUCTIONS = """\
You are reviewing ONE position in a portfolio. The user provides the
aggregate position (units, cost basis, current price, unrealized P/L),
current fundamentals, technicals, news, latest 10-K risk factors, an
insider-selling signal, share-trade signals (insider/institutional
accumulation), and — most importantly for tax-aware rebalancing —
the user's LOT-LEVEL tax history under `tax_lots`.

The user has asked for an AGGRESSIVE rebalance — actively flag SELLs and
TRIMs where a meaningfully better alternative exists, not only when the
thesis is broken outright. BUT respect the tax cost of acting: a SELL
that crystallizes a large short-term gain may be worse than holding for
a few more weeks until the lot seasons into long-term treatment.

Decide one of:

HOLD — thesis still intact AND no clearly better redeployment available
TRIM — partial sale (specify percentage 25/33/50) because:
  * position has appreciated significantly and capital can be redeployed
  * fundamentals softening, de-risk by trimming
  * concentration risk: position too large relative to portfolio
SELL — full exit because:
  * fundamentals clearly deteriorated (neg growth, OCF flipped negative)
  * technical breakdown (price <200DMA, 50DMA crossed below 200DMA)
  * heavy insider selling on top of weak technicals
  * thesis intact but a clearly superior alternative exists

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
Trim percent: <e.g. 25% / 33% / 50% — only if TRIM, omit otherwise>
Position context: <one line — N shares, avg cost $X, current $Y, P/L +/-Z%>
Reasoning:
<2-3 sentences citing specific data>

Tax lot plan: <only if SELL or TRIM, omit on HOLD>
<For each lot recommended:>
  - Lot dated <YYYY-MM-DD>: sell <N> shares (held <X> days, <long-term|short-term>),
    realizes ~$<Y> gain/loss
<If recommending DELAY for long-term treatment, state the date>

What would change your mind:
<one sentence — what would flip the verdict>

CRITICAL:
- Plain text. No markdown headings, no bold.
- Begin with "TICKER:" line. No preamble, no closing remarks.
- If tax_lots is absent, omit the "Tax lot plan" section.\
"""


class Reviewer:
    def __init__(self, provider: Provider, model: str):
        self.agent = AgnoAgent(
            "Reviewer", provider, model, instructions=REVIEWER_INSTRUCTIONS
        )

    def review(self, ticker: str, payload: dict[str, Any]) -> str:
        prompt = (
            f"Holding: {ticker}\n\n"
            f"```json\n{json.dumps(payload, default=str, indent=2)}\n```"
        )
        logger.info("Reviewing holding %s", ticker)
        return self.agent.run(prompt).content


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
