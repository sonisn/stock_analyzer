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
position (units, cost basis, current price, unrealized P/L), current
fundamentals, technicals, news, the latest 10-K risk factors (if available),
and an insider-selling signal.

The user has asked for an AGGRESSIVE rebalance — you should actively flag
SELLs and TRIMs where a meaningfully better alternative could exist, not
only when the thesis is broken outright.

Decide one of:

HOLD — thesis still intact AND no clearly better redeployment available
TRIM — recommend partial sale (specify a percentage 25/33/50) because:
  * position has appreciated significantly (>50% gain) and capital can
    be redeployed
  * fundamentals are softening (decel growth, margin compression, rising debt)
    but not catastrophically — de-risk by trimming
  * concentration risk: position is becoming too large relative to portfolio
SELL — recommend full exit because:
  * fundamentals clearly deteriorated (negative growth, OCF flipped negative,
    debt/equity blew out, earnings miss)
  * technical breakdown (price below 200DMA, 50DMA crossed below 200DMA)
  * heavy insider selling on top of weak technicals
  * thesis is intact but a clearly superior alternative exists (cite the
    opportunity-cost argument explicitly)

DO NOT make tool calls. Use ONLY the data provided. Be specific about WHY,
citing concrete numbers from the data.

Output EXACTLY this structure:

TICKER: <symbol>
Verdict: <HOLD | TRIM | SELL>
Trim percent: <e.g. 25% / 33% / 50% — only if TRIM, omit otherwise>
Position context: <one line — N shares, avg cost $X, current $Y, P/L +/-Z%, held N months>
Reasoning:
<2-3 sentences citing specific data — fundamental change, technical signal,
relative weakness, or opportunity-cost argument. Concrete numbers required.>

What would change your mind:
<one sentence — what would flip the verdict (e.g. "next-quarter revenue
re-accelerates above 15%")>

CRITICAL:
- Plain text. No markdown headings, no bold.
- Begin with "TICKER:" line. No preamble, no closing remarks.\
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
