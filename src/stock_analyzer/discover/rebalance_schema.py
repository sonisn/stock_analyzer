"""Pydantic schema for the rebalancer's structured output.

Kept deliberately minimal: Anthropic's tool/output schema has a hard
"Schema is too complex" complexity limit, and the original richer
schema (with nested CashMath / TaxAgnosticAlternative / RebalanceAction
carrying lots_sold + wash_sale_notice + verbose descriptions) blew past
it on Opus. The prose plan in `full_text` retains all the detail; the
structured fields cover only what the codebase actually reads
programmatically (parse_rebalance_status, parse_actions, dashboard
persistence).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OptionWrite(BaseModel):
    """Structured detail for one WRITE_CALL action.

    Joined to its corresponding RebalanceAction by ticker. Premium is
    quoted PER SHARE (the standard options convention); multiply by 100
    to get dollars per contract."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    strike: float
    expiry: str = Field(..., description="ISO date YYYY-MM-DD.")
    contracts: int = Field(..., gt=0, description="Number of contracts to write.")
    est_premium_per_share: float = Field(
        ..., ge=0,
        description="Mid of bid/ask in dollars per share. ×100 = per contract.",
    )
    delta: float = Field(..., ge=0.0, le=1.0)
    assignment_probability: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


class RebalanceAction(BaseModel):
    """One action line in an ACTION plan."""

    model_config = ConfigDict(frozen=True)

    action: Literal["SELL", "TRIM", "ADD", "BUY", "WRITE_CALL"]
    ticker: str
    sizing: str = Field(..., description="e.g. 'full position', '25%', '~$3,400'.")


class RebalancePlan(BaseModel):
    """Structured rebalance decision.

    Five fields total: status, aggressiveness_applied, actions, summary,
    full_text. Everything else (cash math, tax-agnostic alternatives,
    wash-sale audit, etc.) lives in `full_text` as prose."""

    model_config = ConfigDict(frozen=True)

    status: Literal["NO_ACTION", "ACTION"] = Field(
        ..., description="Whether the plan recommends any portfolio action."
    )
    aggressiveness_applied: Literal["conservative", "balanced", "aggressive"] = Field(
        ..., description="Mode you used (echo the user message)."
    )
    actions: list[RebalanceAction] = Field(
        default_factory=list,
        description="Ordered SELLs first, TRIMs second, ADDs/BUYs last. Empty when status=NO_ACTION.",
    )
    summary: str = Field(
        default="",
        description="One sentence: NO_ACTION rationale, or the big shift on ACTION.",
    )
    full_text: str = Field(
        ...,
        description=(
            "Complete human-readable plan rendered per the Format A / "
            "Format B templates in your instructions. This is what the "
            "user reads in the PDF/email — keep all prose detail here."
        ),
    )
    option_writes: list[OptionWrite] = Field(
        default_factory=list,
        description=(
            "Parallel to WRITE_CALL actions. Each entry MUST have a "
            "matching WRITE_CALL in `actions` with the same ticker. "
            "Empty list when no calls are recommended."
        ),
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
    "OptionWrite",
    "RebalanceAction",
    "RebalancePlan",
    "status_from_plan",
    "actions_from_plan",
]
