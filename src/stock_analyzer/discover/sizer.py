"""Portfolio sizing (Opus, single call).

Allocate new capital across picks given conviction scores, fragility ranks
from the red-team, the ranker's consensus agreement ratio (when the ranker
ran more than one round), and the user's current holdings (for sector
concentration).
"""

from __future__ import annotations

from ..llm import AgnoAgent, Provider, reasoning_model_kwargs, run_with_fallback
from ..logging import get_logger
from ..models.llm import Allocation, CorrelatedPair, SizerOutput

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
- If a "Consensus agreement" block is provided, it's the fraction of
  independent ranker rounds (different providers/models) that
  independently picked this ticker. Unanimous agreement (e.g. 3/3) is a
  real conviction signal on top of the stated conviction score — size
  toward the top of what conviction/fragility already justify. Bare-
  majority agreement (e.g. 2/3) means at least one independent model
  disagreed with including this pick at all — size toward the bottom of
  the justified range, even if conviction/EV look strong.
- Highly correlated picks (same sector/theme) → underweight one or split
- Never recommend more than 35% in any single pick
- If a "Correlated pairs" block is provided, those pairs share a driver
  closely enough that a deterministic check will cap their COMBINED
  allocation at 35% after you respond — size each pair with that cap in
  mind up front (e.g. split roughly evenly, or clearly favor the
  higher-conviction one) rather than letting the automatic clamp make
  the call for you
- If a "Risk-parity weights" block is provided, it's a third sizing
  input (alongside EV and conviction/fragility): the inverse-volatility
  weight each pick would get under equal risk contribution. Use it to
  temper EV/conviction, not override them — a high-conviction, high-EV
  pick with high realized volatility should size somewhat smaller than
  EV alone implies, and a low-volatility pick can size somewhat larger.
  Blend this judgment; don't mechanically copy the risk-parity weight.

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


def _build_agent(provider: Provider, model: str, effort: str) -> AgnoAgent:
    # Adaptive thinking is sufficient for sizing (constraint optimization,
    # not open-ended reasoning). Medium effort.
    return AgnoAgent(
        "Sizer",
        provider,
        model,
        model_kwargs=reasoning_model_kwargs(provider, effort, max_tokens=4000),
        instructions=SIZER_INSTRUCTIONS,
        output_schema=SizerOutput,
    )


class Sizer:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        effort: str = "medium",
        fallback: tuple[Provider, str] | None = None,
    ):
        self.provider = provider
        self.effort = effort
        self.fallback = fallback
        self.agent = _build_agent(provider, model, effort)

    def allocate(
        self,
        picks_text: str,
        bear_case_text: str,
        holdings_summary: str,
        cash_budget: float | None,
        ev_table: str = "",
        agreement_block: str = "",
        correlated_pairs: list[CorrelatedPair] | None = None,
        risk_parity_block: str = "",
        earnings_block: str = "",
    ) -> SizerOutput:
        budget_line = (
            f"Cash budget: ${cash_budget:,.0f}"
            if cash_budget is not None
            else "No dollar budget — output percentages of new capital."
        )
        ev_block = (
            f"Expected return table (deterministic, computed from your "
            f"ranker's probability-weighted scenarios):\n{ev_table}\n\n"
            if ev_table
            else ""
        )
        agreement_block_text = (
            f"Consensus agreement (fraction of independent ranker rounds "
            f"that picked each ticker):\n{agreement_block}\n\n"
            if agreement_block
            else ""
        )
        correlated_pairs_block = (
            "Correlated pairs (combined allocation will be capped at 35%):\n"
            + "\n".join(
                f"  {p.ticker_a} + {p.ticker_b}: {p.shared_driver}" for p in correlated_pairs
            )
            + "\n\n"
            if correlated_pairs
            else ""
        )
        risk_parity_block_text = (
            f"Risk-parity weights (inverse-volatility, equal risk "
            f"contribution — a third sizing input, see instructions):\n"
            f"{risk_parity_block}\n\n"
            if risk_parity_block
            else ""
        )
        earnings_block_text = (
            f"Earnings within the next few days (a deterministic check will cap "
            f"each of these at a small starter position after you respond — size "
            f"and explain them that way):\n{earnings_block}\n\n"
            if earnings_block
            else ""
        )
        prompt = (
            f"{budget_line}\n\n"
            f"{ev_block}"
            f"{agreement_block_text}"
            f"{correlated_pairs_block}"
            f"{risk_parity_block_text}"
            f"{earnings_block_text}"
            f"Current holdings:\n{holdings_summary or '(none)'}\n\n"
            f"Picks (with bull theses):\n{picks_text}\n\n"
            f"Bear cases:\n{bear_case_text}"
        )
        logger.info("Sizing picks (%s)", self.provider)
        build_fallback = (
            (lambda: _build_agent(self.fallback[0], self.fallback[1], self.effort))
            if self.fallback and self.fallback[0] != self.provider
            else None
        )
        result = run_with_fallback(self.agent, build_fallback, prompt).content
        if result is None:
            raise RuntimeError("Sizer returned no content.")
        if isinstance(result, SizerOutput):
            return result
        if isinstance(result, str):
            return SizerOutput.model_validate_json(result)
        raise RuntimeError(f"Sizer returned unexpected type {type(result).__name__}.")


def enforce_earnings_blackout(
    output: SizerOutput,
    earnings_alerts: dict[str, dict],
    *,
    cash_budget: float | None = None,
    max_pct: float = 5.0,
) -> SizerOutput:
    """Cap any pick that reports earnings within the alert window at a
    starter position of `max_pct` of new capital. The trimmed amount is
    deliberately NOT redistributed (that could push other picks past their
    correlation caps) — it's held as cash to deploy after the print, and a
    warning says so. `earnings_alerts` is batch_earnings_flags' output."""
    by_ticker = {a.ticker: a for a in output.allocations}
    warnings: list[str] = []
    for ticker, alert in earnings_alerts.items():
        a = by_ticker.get(ticker)
        if a is None:
            continue
        if a.allocation_pct is not None:
            pct = a.allocation_pct
        elif a.allocation_usd is not None and cash_budget:
            pct = a.allocation_usd / cash_budget * 100
        else:
            continue
        if pct <= max_pct:
            continue
        update = (
            {"allocation_pct": max_pct}
            if a.allocation_pct is not None
            else {"allocation_usd": max_pct / 100 * cash_budget}
        )
        by_ticker[ticker] = a.model_copy(update=update)
        warnings.append(
            f"EARNINGS BLACKOUT: {ticker} reports {alert.get('earnings_date')} "
            f"(in {alert.get('days_until')}d) — capped at a {max_pct:.0f}% starter "
            f"position instead of {pct:.1f}%; hold the other {pct - max_pct:.1f}% as "
            f"cash and add after the print if the thesis holds"
        )
    if not warnings:
        return output
    return output.model_copy(
        update={
            "allocations": [by_ticker[a.ticker] for a in output.allocations],
            "concentration_warnings": [*output.concentration_warnings, *warnings],
        }
    )


def enforce_correlation_caps(
    output: SizerOutput,
    pairs: list[CorrelatedPair],
    *,
    cash_budget: float | None = None,
    max_combined_pct: float = 35.0,
) -> SizerOutput:
    """Deterministic post-LLM check: scale down any flagged correlated pair
    whose combined allocation exceeds `max_combined_pct`, proportionally,
    so their combined weight lands exactly at the cap.

    Same shape as cc_validation.py::validate_option_writes /
    reviewer.py::_repair_verdict_inconsistencies — mutates a frozen
    Pydantic model via model_copy and returns warnings alongside it.
    Pairs naming a ticker not in `output.allocations` are ignored (e.g.
    the pick was dropped or renamed downstream).
    """
    by_ticker = {a.ticker: a for a in output.allocations}
    warnings: list[str] = []

    def _effective_pct(a: Allocation) -> float | None:
        if a.allocation_pct is not None:
            return a.allocation_pct
        if a.allocation_usd is not None and cash_budget:
            return a.allocation_usd / cash_budget * 100
        return None

    for pair in pairs:
        a = by_ticker.get(pair.ticker_a)
        b = by_ticker.get(pair.ticker_b)
        if a is None or b is None:
            continue
        pct_a = _effective_pct(a)
        pct_b = _effective_pct(b)
        if pct_a is None or pct_b is None:
            continue
        combined = pct_a + pct_b
        if combined <= max_combined_pct:
            continue
        scale = max_combined_pct / combined
        new_pct_a = pct_a * scale
        new_pct_b = pct_b * scale
        update_a: dict[str, float] = {}
        update_b: dict[str, float] = {}
        if a.allocation_pct is not None:
            update_a["allocation_pct"] = new_pct_a
        else:
            update_a["allocation_usd"] = new_pct_a / 100 * cash_budget
        if b.allocation_pct is not None:
            update_b["allocation_pct"] = new_pct_b
        else:
            update_b["allocation_usd"] = new_pct_b / 100 * cash_budget
        by_ticker[pair.ticker_a] = a.model_copy(update=update_a)
        by_ticker[pair.ticker_b] = b.model_copy(update=update_b)
        warnings.append(
            f"CORRELATION CAP: {pair.ticker_a} + {pair.ticker_b} "
            f"({pair.shared_driver}) totaled {combined:.1f}% — scaled down to "
            f"{new_pct_a:.1f}% + {new_pct_b:.1f}% = {max_combined_pct:.0f}% combined"
        )

    if not warnings:
        return output

    new_allocations = [by_ticker[a.ticker] for a in output.allocations]
    return output.model_copy(
        update={
            "allocations": new_allocations,
            "concentration_warnings": [*output.concentration_warnings, *warnings],
        }
    )
