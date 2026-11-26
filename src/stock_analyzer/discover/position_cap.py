"""No buy may take one stock past REBALANCE_MAX_POSITION_PCT of the book.

The cap was only in the rebalancer's prompt. That is advice; this is the
arithmetic, the same split the sale and put checks use: each BUY/ADD
that would leave its stock above the cap is cut to the room left (rounded
down to $100), and dropped when under $100 of room is left.

The book is every position at the broker's price plus cash. A sale of
the same stock earlier in the plan counts first, so a trim-then-add
inside the cap passes. A buy whose size can't be read is left alone and
named, never guessed.

Two tighter limits from discover/concentration.py ride on the same pass:
`ticker_max_pct` (a volatile stock's own, lower cap) and `groups` (sets of
holdings that move together, held to `max_group_pct` of the book
combined). A buy is cut to the least room any of them leaves.
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
    ticker_max_pct: dict[str, float] | None = None,
    groups: list[list[str]] | None = None,
    max_group_pct: float = 100.0,
) -> tuple[RebalancePlan, list[str]]:
    """Cut buys that would break the single-stock, volatility or group
    caps. (plan, warnings)."""
    ticker_max_pct = {t.upper(): p for t, p in (ticker_max_pct or {}).items() if p < max_pct}
    groups = groups or []
    group_of = (
        {t.upper(): i for i, g in enumerate(groups) for t in g} if 0 < max_group_pct < 100 else {}
    )
    if (max_pct >= 100 and not ticker_max_pct and not group_of) or not plan.actions:
        return plan, []
    total = sum(float(p.get("value") or 0.0) for p in positions.values()) + float(cash or 0.0)
    if total <= 0:
        return plan, []
    held = {t: float(p.get("value") or 0.0) for t, p in positions.items()}
    group_held = [0.0] * len(groups)
    for t, v in held.items():
        if t.upper() in group_of:
            group_held[group_of[t.upper()]] += v
    group_limit = total * max_group_pct / 100.0
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
        gi = group_of.get(action.ticker.upper())
        if action.action in SALE_ACTIONS:
            if dollars:
                out = min(dollars, held.get(action.ticker, 0.0))
                held[action.ticker] = held.get(action.ticker, 0.0) - out
                if gi is not None:
                    group_held[gi] -= out
            continue
        if action.action not in BUY_ACTIONS:
            continue
        cap = min(max_pct, ticker_max_pct.get(action.ticker.upper(), max_pct))
        limit = total * cap / 100.0
        why = (
            f"the {cap:g}% cap for a stock this volatile"
            if cap < max_pct
            else f"the {max_pct:g}% single-stock cap"
        )
        if dollars is None:
            warnings.append(
                f"{action.ticker}: {action.action} '{action.sizing}' could not be sized — "
                f"check by hand that it keeps {action.ticker} under {why} "
                f"(~${limit:,.0f})"
                + (
                    f" and its correlated group under {max_group_pct:g}%."
                    if gi is not None
                    else "."
                )
            )
            continue
        now = held.get(action.ticker, 0.0)
        pos_room = limit - now
        grp_room = group_limit - group_held[gi] if gi is not None else float("inf")
        if gi is not None and grp_room < pos_room:
            members = ", ".join(sorted(groups[gi])[:6])
            why = (
                f"the {max_group_pct:g}% cap on holdings that move together ({members}: "
                f"~{group_held[gi] / total:.0%} now)"
            )
        if dollars <= min(pos_room, grp_room) + 1.0:
            held[action.ticker] = now + dollars
            if gi is not None:
                group_held[gi] += dollars
            continue
        room = max(0.0, min(pos_room, grp_room)) // 100 * 100
        if room < _MIN_BUY_USD:
            removed.add(i)
            warnings.append(
                f"{action.ticker}: dropped {action.action} of ~${dollars:,.0f} — "
                f"{action.ticker} is already ~{now / total:.0%} of the book; {why} leaves "
                "no room."
            )
            continue
        acct = _account(action.sizing, accounts or [])
        where = f" in {acct}" if acct else ""
        actions[i] = action.model_copy(
            update={
                "sizing": f"~${room:,.0f}{where} (cut from: {action.sizing}) — keeps "
                f"{action.ticker} within {why}"
            }
        )
        held[action.ticker] = now + room
        if gi is not None:
            group_held[gi] += room
        warnings.append(
            f"{action.ticker}: {action.action} cut from ~${dollars:,.0f} to ~${room:,.0f} — "
            f"more would break {why}."
        )
    for w in warnings:
        logger.warning("Position cap: %s", w)
    if not warnings:
        return plan, []
    kept = [a for i, a in enumerate(actions) if i not in removed]
    plan = plan.model_copy(update={"actions": kept})
    changed = [w for w in warnings if "could not be sized" not in w]
    return (note_adjustments(plan, changed) if changed else plan), warnings
