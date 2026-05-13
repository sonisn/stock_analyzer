"""Portfolio sizing (Opus, single call).

Allocate new capital across picks given conviction scores, fragility ranks
from the red-team, and the user's current holdings (for sector concentration).
"""
from __future__ import annotations

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from .schemas import SizerOutput

logger = get_logger(__name__)

SIZER_INSTRUCTIONS = """\
You are a portfolio manager allocating new capital across a set of picks
that have already passed bull + bear analysis. The user provides the picks,
their bear-case fragility ranks, their conviction scores, the user's
current holdings, and either a cash budget (dollars) or a request for
percentage allocations.

DO NOT make tool calls. Use ONLY the data provided.

For each pick, output:

---
TICKER: <symbol>
Allocation: <dollars if budget given, else % of new capital>
Rationale: <1-2 sentences citing conviction, fragility, correlation to existing holdings>
---

End with a "Concentration warnings:" block listing any sector or theme
where the new picks + existing holdings would exceed 30% combined.

Allocation principles to follow:
- The user message includes an EXPECTED RETURN TABLE — pre-computed
  E[return] = Σ(probability × scenario_return) from the ranker's
  bull/base/bear scenarios. This is the PRIMARY ranking signal:
  size proportional to expected return.
- Higher conviction (and thus typically higher EV) → larger position,
  up to ~30% of new capital
- Higher fragility (bear-case rank 1-2) → smaller position, even if
  EV is high (high EV with high dispersion = risky bet)
- Highly correlated picks (same sector/theme) → underweight one or split
- Never recommend more than 35% in any single pick

CRITICAL:
- Plain text only. No markdown headings or bold.
- Allocations must sum to 100% (or the full dollar budget).

CITATION RULE (anti-hallucination):
Cite the SPECIFIC inputs that justify each sizing: "NVDA conviction 8
+ fragility 3 → 30%" — not "NVDA is hot, size big." Conviction and
fragility numbers must come from the picks / bear-case inputs the
user provided. If you cite a sector concentration percentage, derive
it explicitly from holdings_summary; don't estimate.

STRUCTURED OUTPUT:
Your response is validated against a Pydantic schema (SizerOutput).
Populate `allocations` with one Allocation per pick (ticker, rationale,
plus EITHER allocation_pct OR allocation_usd depending on whether a
cash budget was provided). Populate `concentration_warnings` with
the same warnings you list at the end of the prose. Put the full
prose plan in `full_text`. Structured fields must match the prose.\
"""


class Sizer:
    def __init__(
        self, provider: Provider, model: str, *, effort: str = "medium"
    ):
        # Adaptive thinking is sufficient for sizing (constraint optimization,
        # not open-ended reasoning). Medium effort.
        self.agent = AgnoAgent(
            "Sizer",
            provider,
            model,
            model_kwargs={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "max_tokens": 4000,
                "temperature": 0,
            },
            instructions=SIZER_INSTRUCTIONS,
            output_schema=SizerOutput,
        )

    def allocate(
        self,
        picks_text: str,
        bear_case_text: str,
        holdings_summary: str,
        cash_budget: float | None,
        ev_table: str = "",
    ) -> SizerOutput:
        budget_line = (
            f"Cash budget: ${cash_budget:,.0f}"
            if cash_budget is not None
            else "No dollar budget — output percentages of new capital."
        )
        ev_block = (
            f"Expected return table (deterministic, computed from your "
            f"ranker's probability-weighted scenarios):\n{ev_table}\n\n"
            if ev_table else ""
        )
        prompt = (
            f"{budget_line}\n\n"
            f"{ev_block}"
            f"Current holdings:\n{holdings_summary or '(none)'}\n\n"
            f"Picks (with bull theses):\n{picks_text}\n\n"
            f"Bear cases:\n{bear_case_text}"
        )
        logger.info("Sizing picks with Opus")
        result = self.agent.run(prompt).content
        if result is None:
            raise RuntimeError("Sizer returned no content.")
        if isinstance(result, SizerOutput):
            return result
        if isinstance(result, str):
            return SizerOutput.model_validate_json(result)
        raise RuntimeError(
            f"Sizer returned unexpected type {type(result).__name__}."
        )
