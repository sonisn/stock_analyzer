"""Core and satellite: an index-fund core sized by CORE_TARGET_PCT, stock
picks around it.

Why: across 2012-2026 nothing in the screen's fundamentals beat SPY
(`factor-study`), the model's picks have no graded edge yet (the
six-month scorecard), and the plan check found the stock mix's swings
(~37% a year against SPY's ~14%) cost more goal odds than any pick could
plausibly add. A core holds what the market gives; the satellite is where
the picks earn — or fail to earn — a larger share on the scorecard.

Off by default (CORE_TARGET_PCT=0): the target is the investor's call,
made with the plan check's odds-by-core-share table in hand.

When on, each rebalance:
  - measures the core (every holding in CORE_EQUIVALENTS — an S&P 500 or
    total-market fund is the same core whoever runs it) against the target;
  - asks for at most CORE_STEP_PCT of the portfolio per rebalance, so the
    move is gradual rather than one large sale;
  - tells the rebalancer to fund it from idle cash first, then trims of
    satellite stocks inside tax-advantaged accounts (no tax on a sale
    there), and never from a taxable sale just to fund the core;
  - checks the plan afterwards: a step it should have taken and did not
    is reported, not silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# S&P 500 and total-market index funds and ETFs: any of them counts toward
# the core. CORE_FUND (the one to buy) is added to this set.
CORE_EQUIVALENTS: frozenset[str] = frozenset(
    {
        "VOO", "IVV", "SPY", "SPLG", "VTI", "ITOT", "SCHB", "SCHX",
        "FXAIX", "FSKAX", "FZROX", "SWPPX", "SWTSX", "VFIAX", "VTSAX",
    }
)  # fmt: skip


@dataclass(frozen=True)
class CoreStatus:
    fund: str
    target_pct: float
    core_value: float
    total_value: float
    holdings: tuple[str, ...]  # the core funds held

    @property
    def core_pct(self) -> float:
        return 100 * self.core_value / self.total_value if self.total_value else 0.0

    @property
    def shortfall_usd(self) -> float:
        return max(0.0, self.target_pct / 100 * self.total_value - self.core_value)

    def step_usd(self, step_pct: float) -> float:
        """What this rebalance should move into the core: the shortfall,
        capped at `step_pct` of the portfolio."""
        return min(self.shortfall_usd, step_pct / 100 * self.total_value)


def core_status(
    positions: dict[str, dict[str, Any]],
    cash: float | None,
    *,
    fund: str,
    target_pct: float,
    equivalents: frozenset[str] = CORE_EQUIVALENTS,
) -> CoreStatus | None:
    """The core against its target, or None when the core is off."""
    if target_pct <= 0:
        return None
    core_names = {*equivalents, fund.upper()}
    held = {
        t: float(p.get("value") or 0)
        for t, p in positions.items()
        if t.upper() in core_names and (p.get("value") or 0) > 0
    }
    total = sum(float(p.get("value") or 0) for p in positions.values()) + (cash or 0.0)
    return CoreStatus(
        fund=fund.upper(),
        target_pct=target_pct,
        core_value=sum(held.values()),
        total_value=total,
        holdings=tuple(sorted(held)),
    )


def core_block(status: CoreStatus | None, *, step_pct: float) -> str:
    """The rebalancer's prompt block for the core ("" when off or met)."""
    if status is None:
        return ""
    head = (
        f"CORE INDEX FUND (deterministic; the investor's own target): {status.target_pct:.0f}% of "
        f"the portfolio in a broad index fund, {status.fund} to buy. Now "
        f"{status.core_pct:.1f}% (${status.core_value:,.0f} of ${status.total_value:,.0f}"
        + (f" in {', '.join(status.holdings)}" if status.holdings else "")
        + "). The core is a diversified index fund: the single-position cap does not "
        "apply to it, and it is never a candidate for a TRIM or SELL."
    )
    step = status.step_usd(step_pct)
    if step < 1:
        return head + " Target met: no core purchase this time."
    return (
        f"{head}\nThis rebalance moves ~${step:,.0f} into {status.fund} (the gap, capped at "
        f"{step_pct:.0f}% of the portfolio per rebalance). Include it as a BUY or ADD of "
        f"{status.fund}, before any satellite BUY or ADD, funded in this order:\n"
        "  1. idle cash;\n"
        "  2. TRIMs of satellite stocks held in a tax-advantaged account (IRA, 401(k), "
        "HSA), where a sale owes no tax — weakest reviews first, and never a position "
        "the instructions or the investor say to keep;\n"
        "  3. never a sale in a taxable account made only to fund the core.\n"
        "If 1 and 2 cannot cover it, buy what they can and say how much is left for "
        "the next rebalance."
    )


def check_core(plan: Any, status: CoreStatus | None, *, step_pct: float) -> list[str]:
    """Warnings when a due core step is missing from `plan`."""
    if status is None or status.step_usd(step_pct) < 1:
        return []
    actions = getattr(plan, "actions", None) or []
    bought = any(a.action in ("BUY", "ADD") and a.ticker.upper() == status.fund for a in actions)
    if bought:
        return []
    return [
        f"Core index fund: the plan does not buy {status.fund}, though the core is "
        f"{status.core_pct:.1f}% against a {status.target_pct:.0f}% target "
        f"(~${status.step_usd(step_pct):,.0f} due this rebalance)."
    ]


__all__ = ["CORE_EQUIVALENTS", "CoreStatus", "check_core", "core_block", "core_status"]
