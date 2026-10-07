"""No buy may take one stock past REBALANCE_MAX_POSITION_PCT of the book.

The cap was only in the rebalancer's prompt. That is advice; this is the
arithmetic, the same split the sale and put checks use: each BUY/ADD
that would leave its stock above the cap is cut to the room left (rounded
down to $100), and dropped when under $100 of room is left.

The book is every position at the broker's price plus cash. A sale of
the same stock earlier in the plan counts first, so a trim-then-add
inside the cap passes. A buy whose size can't be read is left alone and
named, never guessed.
"""

from __future__ import annotations

from typing import Any

from ..logging import get_logger
from ..models.rebalance import RebalancePlan
from .csp_validation import estimate_action_dollars
from .sale_validation import BUY_ACTIONS, SALE_ACTIONS, _account, note_adjustments

logger = get_logger(__name__)

_MIN_BUY_USD = 100.0


def cap_positions(
    plan: RebalancePlan,
    *,
    positions: dict[str, dict[str, Any]],
    cash: float | None,
    max_pct: float,
    prices: dict[str, float | None] | None = None,
    accounts: list[str] | None = None,
) -> tuple[RebalancePlan, list[str]]:
    """Cut buys that would break the single-stock cap. (plan, warnings)."""
    if max_pct >= 100 or not plan.actions:
        return plan, []
    total = sum(float(p.get("value") or 0.0) for p in positions.values()) + float(cash or 0.0)
    if total <= 0:
        return plan, []
    limit = total * max_pct / 100.0
    held = {t: float(p.get("value") or 0.0) for t, p in positions.items()}
    actions = list(plan.actions)
    removed: set[int] = set()
    warnings: list[str] = []
    for i, action in enumerate(actions):
        pos = positions.get(action.ticker) or {}
        units = float(pos.get("units") or 0.0)
        price = (float(pos.get("value") or 0.0) / units if units else None) or (
            (prices or {}).get(action.ticker)
        )
        dollars = estimate_action_dollars(action, units=units, price=price)
        if action.action in SALE_ACTIONS:
            if dollars:
                held[action.ticker] = max(0.0, held.get(action.ticker, 0.0) - dollars)
            continue
        if action.action not in BUY_ACTIONS:
            continue
        if dollars is None:
            warnings.append(
                f"{action.ticker}: {action.action} '{action.sizing}' could not be sized — "
                f"check by hand that it keeps {action.ticker} under the {max_pct:g}% cap "
                f"(~${limit:,.0f})."
            )
            continue
        now = held.get(action.ticker, 0.0)
        if now + dollars <= limit + 1.0:
            held[action.ticker] = now + dollars
            continue
        room = (max(0.0, limit - now)) // 100 * 100
        if room < _MIN_BUY_USD:
            removed.add(i)
            warnings.append(
                f"{action.ticker}: dropped {action.action} of ~${dollars:,.0f} — "
                f"{action.ticker} is already ~{now / total:.0%} of the book, at or over the "
                f"{max_pct:g}% cap."
            )
            continue
        acct = _account(action.sizing, accounts or [])
        where = f" in {acct}" if acct else ""
        actions[i] = action.model_copy(
            update={
                "sizing": f"~${room:,.0f}{where} (cut from: {action.sizing}) — keeps "
                f"{action.ticker} at the {max_pct:g}% cap"
            }
        )
        held[action.ticker] = now + room
        warnings.append(
            f"{action.ticker}: {action.action} cut from ~${dollars:,.0f} to ~${room:,.0f} — "
            f"more would take it past the {max_pct:g}% single-stock cap (~${limit:,.0f})."
        )
    for w in warnings:
        logger.warning("Position cap: %s", w)
    if not warnings:
        return plan, []
    kept = [a for i, a in enumerate(actions) if i not in removed]
    plan = plan.model_copy(update={"actions": kept})
    changed = [w for w in warnings if "could not be sized" not in w]
    return (note_adjustments(plan, changed) if changed else plan), warnings
