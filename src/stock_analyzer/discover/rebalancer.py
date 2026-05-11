"""Portfolio rebalance decision (Opus + extended thinking, single call).

Synthesizes per-holding HOLD/TRIM/SELL verdicts from the Reviewer + new
discover picks from the Ranker + cash available + macro regime into a
coherent action list, ordered for execution: SELLs/TRIMs raise cash, BUYs
deploy it.

Aggressive churn: explicitly encouraged in the prompt — recommend SELLs
where a meaningfully better alternative exists, even if the existing
holding is fine in isolation.
"""
from __future__ import annotations

from ..llm import AgnoAgent, Provider
from ..logging import get_logger

logger = get_logger(__name__)

REBALANCER_INSTRUCTIONS = """\
You are a portfolio manager producing a concrete rebalance action list.
The user provides:
- HOLD/TRIM/SELL verdicts for each current holding (with reasoning + P/L)
- 5 new discover picks with bull theses, bear cases, and conviction
- Current cash balance available at the broker
- Macro regime summary

The user has asked for an AGGRESSIVE rebalance — actively churn into
better ideas. SELLs and TRIMs are encouraged where a meaningfully better
alternative exists, even if the existing holding is technically OK.

DO NOT make tool calls. Use ONLY the data provided. Reason about WHOLE
PORTFOLIO health, not each ticker in isolation.

Hard constraints:
- Total BUYs must NOT exceed (SELL proceeds + TRIM proceeds + available cash).
- No single position should exceed ~25% of post-rebalance portfolio value.
- No leverage, no options, no shorts.
- Order actions by execution: SELLs first, TRIMs second, BUYs last.
- For each SELL/TRIM, flag if the gain would be short-term (<12 months held).

Output EXACTLY this format:

REBALANCE PLAN

Summary:
<2-3 sentences on the big shift this rebalance makes and why>

Cash math:
SELL proceeds: ~$<approx>
TRIM proceeds: ~$<approx>
Available cash: $<from input>
Total BUY budget: ~$<sum>

---
Action 1: SELL <TICKER> (full position, raises ~$X)
Reasoning: <one or two sentences citing concrete data>
Tax note: <"short-term gain" | "long-term gain" | "loss" | "—">
---
Action 2: TRIM <TICKER> by <pct>% (raises ~$X)
Reasoning: <one or two sentences>
Tax note: <as above>
---
[...as many SELL/TRIM as needed...]
---
Action N: BUY <TICKER> (~$X, ~<pct>% of new capital)
Reasoning: <one sentence, citing which discover pick this is and its conviction>
---
[...as many BUYs as needed...]

Concentration check:
<one paragraph: after these actions, what is the largest single position
(% of portfolio), what are the top-3 sector weights, flag if anything
exceeds the 25% single-name cap>

Risk summary:
<one paragraph: net change in portfolio risk profile. Is this rebalance
defensive, neutral, or risk-on? Cite specific evidence>

CRITICAL:
- Plain text only. No markdown headings or bold.
- Order: SELLs → TRIMs → BUYs.
- Sum constraint: BUYs total ≤ proceeds + cash.
- If a holding has a SELL verdict but the math would over-deploy proceeds,
  still recommend the SELL and let cash accumulate; do not invent BUYs
  beyond budget.\
"""


class Rebalancer:
    def __init__(
        self, provider: Provider, model: str, *, thinking_budget: int = 12000
    ):
        self.agent = AgnoAgent(
            "Rebalancer",
            provider,
            model,
            model_kwargs={
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
                "max_tokens": thinking_budget + 8000,
            },
            instructions=REBALANCER_INSTRUCTIONS,
        )

    def decide(
        self,
        holdings_reviews: dict[str, str],
        picks_text: str,
        cash_available: float | None,
        macro_summary: str = "",
    ) -> str:
        reviews_block = "\n\n".join(
            f"=== {ticker} ===\n{text}" for ticker, text in holdings_reviews.items()
        )
        cash_line = (
            f"Available cash: ${cash_available:,.0f}"
            if cash_available is not None
            else "Available cash: unknown (size BUYs from SELL+TRIM proceeds only)"
        )
        macro_block = f"Macro regime:\n{macro_summary}\n\n" if macro_summary else ""
        prompt = (
            f"{macro_block}"
            f"{cash_line}\n\n"
            f"Current holdings reviews ({len(holdings_reviews)}):\n\n{reviews_block}\n\n"
            f"New discover picks:\n\n{picks_text}"
        )
        logger.info(
            "Generating rebalance plan with Opus + extended thinking "
            "(%d holdings, cash=%s)",
            len(holdings_reviews),
            f"${cash_available:,.0f}" if cash_available is not None else "unknown",
        )
        return self.agent.run(prompt).content
