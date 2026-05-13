"""Pydantic schema for the rebalancer's structured output.

Replaces the regex-parsing layer that the rebalancer relied on previously
(parse_rebalance_status, parse_actions on a free-text plan — the source
of the `expected string or bytes-like object` crash we hit when the
LLM returned None content).

Now the rebalancer emits a `RebalancePlan` object validated against this
schema via agno's `output_schema=` parameter. The free-text `full_text`
field carries the human-readable plan for the PDF/email — that
preserves the existing report layout. Programmatic consumers (parsers,
dashboards, future analytics) read the structured fields directly.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CashMath(BaseModel):
    """Cash bridge for an ACTION plan."""

    model_config = ConfigDict(frozen=True)

    sell_proceeds: float = 0.0
    trim_proceeds: float = 0.0
    available_cash: float = 0.0
    total_budget: float = 0.0


class RebalanceAction(BaseModel):
    """One action line in an ACTION plan."""

    model_config = ConfigDict(frozen=True)

    order: int = Field(..., description="Execution order, 1-indexed.")
    action: Literal["SELL", "TRIM", "ADD", "BUY"]
    ticker: str
    reasoning: str = Field(..., description="1-2 sentence justification citing concrete data.")
    sizing: str = Field(
        ...,
        description=(
            "Plain-text sizing: 'full position', '25%', '~$3,400', "
            "'~10 shares'. Match the existing prose conventions."
        ),
    )
    lots_sold: list[str] = Field(
        default_factory=list,
        description=(
            "SELL/TRIM only. Format: '<YYYY-MM-DD>: <N> shares, "
            "<long-term|short-term>, realizes ~$<gain/loss>'."
        ),
    )
    wash_sale_notice: str | None = Field(
        default=None,
        description=(
            "Only set if any lot above is at a loss. Verbatim from the "
            "existing 'Wash-sale notice:' template."
        ),
    )


class TaxAgnosticAlternative(BaseModel):
    """One row of the mandatory 'Tax-agnostic alternative' section in NO_ACTION."""

    model_config = ConfigDict(frozen=True)

    source_ticker: str
    destination_ticker: str
    trim_pct: float = Field(..., description="What % of source we would have trimmed.")
    tax_cost_usd: float = Field(..., description="Realized tax cost if executed today.")
    tax_treatment: Literal["short_term", "long_term", "mixed"]
    forward_return_edge_pretax_pct: float
    net_edge_posttax_pct: float
    verdict: Literal["still_positive", "wiped_out_by_tax"]


class RebalancePlan(BaseModel):
    """Structured rebalance decision.

    The `full_text` field carries the prose plan exactly as we used to
    emit before — that lets the PDF/email layer stay unchanged. The
    structured fields are bonuses: parsers and downstream analytics read
    them without regex.

    Most fields are optional so the LLM only populates what's relevant
    to the chosen status (NO_ACTION fills the NO_ACTION block; ACTION
    fills the ACTION block).
    """

    status: Literal["NO_ACTION", "ACTION"] = Field(
        ...,
        description=(
            "Whether the plan recommends any portfolio action this run."
        ),
    )
    aggressiveness_applied: Literal["conservative", "balanced", "aggressive"] = Field(
        ...,
        description=(
            "Which mode you used. Echo back what the user message asked for."
        ),
    )

    # Always present — the prose rendering of the plan.
    full_text: str = Field(
        ...,
        description=(
            "Human-readable plan rendered exactly per the Format A / "
            "Format B templates in your instructions. This is what the "
            "user reads in the PDF/email."
        ),
    )

    # NO_ACTION fields
    add_first_walk: str | None = Field(
        default=None,
        description="One short paragraph: the STEP 2 / STEP 3 deployment-order audit.",
    )
    intra_portfolio_check: str | None = Field(
        default=None,
        description=(
            "One sentence listing every (source, destination) pair "
            "considered for intra-portfolio rebalance, with the "
            "confidence gap."
        ),
    )
    tax_agnostic_alternatives: list[TaxAgnosticAlternative] = Field(
        default_factory=list,
        description=(
            "MANDATORY in every NO_ACTION output. One row per pair "
            "rejected in the intra-portfolio check, plus any other "
            "trim->add pair that would be opportunistically attractive "
            "absent tax friction."
        ),
    )
    conclusion: str | None = Field(default=None, description="One-sentence wrap-up.")
    reasoning: str | None = Field(
        default=None,
        description="2-3 sentences citing specific reviewer verdicts.",
    )
    forward_outlook: str | None = Field(
        default=None,
        description="One paragraph: what's working, what to monitor, what would trigger a rebalance.",
    )
    opportunistic_note: str | None = Field(
        default=None,
        description="At most one sentence flagging a watchlist candidate.",
    )

    # ACTION fields
    summary: str | None = Field(
        default=None,
        description="2-3 sentences on the big shift this rebalance makes and why.",
    )
    cash_math: CashMath | None = Field(default=None)
    actions: list[RebalanceAction] = Field(
        default_factory=list,
        description=(
            "Ordered SELLs first, TRIMs second, ADDs/BUYs last. "
            "Sum of BUYs + ADDs must NOT exceed total_budget."
        ),
    )
    concentration_check: str | None = Field(
        default=None,
        description="Post-rebalance largest single position % and top-3 sector weights.",
    )
    risk_summary: str | None = Field(
        default=None,
        description="One paragraph: net change in portfolio risk profile.",
    )
    estimated_tax_impact: str | None = Field(
        default=None,
        description="One paragraph: aggregate realized long-term + short-term + losses.",
    )
    wash_sale_audit: str | None = Field(
        default=None,
        description="One paragraph confirming no wash-sale violations.",
    )


def status_from_plan(plan: RebalancePlan | None) -> str:
    """Lossy bridge for legacy parse_rebalance_status callers."""
    if plan is None:
        return "UNKNOWN"
    return plan.status


def actions_from_plan(plan: RebalancePlan | None) -> list[tuple[str, str]]:
    """Lossy bridge for legacy parse_actions callers — returns
    [(action_type, ticker), ...] in execution order."""
    if plan is None:
        return []
    return [(a.action, a.ticker) for a in plan.actions]


__all__ = [
    "CashMath",
    "RebalanceAction",
    "TaxAgnosticAlternative",
    "RebalancePlan",
    "status_from_plan",
    "actions_from_plan",
]
