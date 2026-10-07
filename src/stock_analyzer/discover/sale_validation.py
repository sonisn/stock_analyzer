"""Shares promised to a written call are not shares you can sell.

`fetch_covered_call_obligations` was wired into the daily email and the
tax-loss harvester, both of which subtract promised shares before
proposing a sale. The rebalancer never got it, and on 2026-09-20 it
proposed trimming 50 TSLA shares when 200 of 200.37 were backing two
$400 calls — 0.37 free. The plan read as ordinary and could not be
executed; selling would have left the calls naked.

Two layers, because either alone is thin. `covered_call_block` puts the
obligations in the prompt so the model can plan around them, and
`validate_sales` re-checks the plan it produces against the same numbers
— the prompt is advice, the validator is arithmetic. That is the split
the covered-call and put paths already use.

A dropped sale can leave a buy without its money: on 2026-10-07 the
plan put "~$40,200" into NVDA counting on a TSLA trim the validator had
to drop, with $21,404 actually in the account. `resize_unfunded_buys`
shrinks such buys to the cash that is there and says so in the summary.
"""

from __future__ import annotations

import re
from typing import Any

from ..logging import get_logger
from ..models.rebalance import RebalanceAction, RebalancePlan

logger = get_logger(__name__)

SALE_ACTIONS = frozenset({"SELL", "TRIM"})
BUY_ACTIONS = frozenset({"ADD", "BUY"})
_MIN_BUY_USD = 100.0


def free_shares(
    ticker: str,
    positions: dict[str, dict[str, Any]],
    obligations: dict[str, dict[str, Any]],
) -> float | None:
    """Shares not backing a short call, or None when it can't be known."""
    held = (positions.get(ticker) or {}).get("units")
    if held is None:
        return None
    committed = float((obligations.get(ticker) or {}).get("shares_committed") or 0.0)
    return max(0.0, float(held) - committed)


def covered_call_block(
    positions: dict[str, dict[str, Any]],
    obligations: dict[str, dict[str, Any]],
) -> str:
    """The prompt block naming what is already promised, and until when."""
    rows = []
    for ticker in sorted(obligations):
        rec = obligations[ticker]
        committed = float(rec.get("shares_committed") or 0.0)
        if committed <= 0:
            continue
        held = float((positions.get(ticker) or {}).get("units") or 0.0)
        free = max(0.0, held - committed)
        strike = rec.get("lowest_strike")
        expiry = rec.get("next_expiry")
        rows.append(
            f"  {ticker}: {held:,.2f} held, {committed:,.0f} promised to "
            f"{rec.get('contracts', 0)} short call(s)"
            + (f" at ${float(strike):,.2f}" if strike else "")
            + (f" expiring {expiry}" if expiry else "")
            + f" -> {free:,.2f} share(s) are free to sell"
        )
    if not rows:
        return ""
    return (
        "SHARES ALREADY PROMISED (covered calls you have written)\n"
        "Selling a share that backs a short call turns that call naked. Only\n"
        "the free shares below can be sold; to go further you must buy the\n"
        "call back first, and the plan has to say so and account for its cost.\n" + "\n".join(rows)
    )


def _requested_shares(sizing: str, held: float) -> float | None:
    """Shares a sizing string asks for, when it can be read as a count."""
    text = str(sizing or "")
    if m := re.search(r"(\d[\d,]*\.?\d*)\s*(?:share|unit)", text, re.I):
        return float(m.group(1).replace(",", ""))
    if m := re.search(r"(\d+(?:\.\d+)?)\s*%", text):
        return held * float(m.group(1)) / 100.0
    if re.search(r"\bfull position\b|\ball\b", text, re.I):
        return held
    return None


def validate_sales(
    plan: RebalancePlan,
    *,
    positions: dict[str, dict[str, Any]],
    obligations: dict[str, dict[str, Any]],
    account_cash: dict[str, float] | None = None,
) -> tuple[RebalancePlan, list[str]]:
    """Drop sale actions that would sell promised shares, then shrink the
    buys they were paying for (`resize_unfunded_buys`, given `account_cash`).

    A sale is only dropped when the numbers say so outright: the ticker
    has an obligation, the sizing can be read as a share count, and that
    count exceeds what is free. Anything unreadable is left alone and
    reported — this must not quietly delete a sale it merely failed to
    parse.
    """
    if not obligations:
        return plan, []
    kept: list[RebalanceAction] = []
    warnings: list[str] = []
    for action in plan.actions:
        if action.action not in SALE_ACTIONS or action.ticker not in obligations:
            kept.append(action)
            continue
        held = float((positions.get(action.ticker) or {}).get("units") or 0.0)
        free = free_shares(action.ticker, positions, obligations)
        wanted = _requested_shares(action.sizing, held)
        if free is None or wanted is None:
            kept.append(action)
            if free is not None:
                warnings.append(
                    f"{action.ticker}: {action.action} '{action.sizing}' could not be read as a "
                    f"share count — only {free:,.2f} of {held:,.2f} shares are free to sell "
                    "(the rest back written calls). Check it by hand."
                )
            continue
        if wanted <= free + 1e-6:
            kept.append(action)
            continue
        rec = obligations[action.ticker]
        warnings.append(
            f"{action.ticker}: dropped {action.action} of {wanted:,.2f} share(s) — only "
            f"{free:,.2f} are free. {float(rec.get('shares_committed') or 0):,.0f} back "
            f"{rec.get('contracts', 0)} short call(s)"
            + (f" at ${float(rec['lowest_strike']):,.2f}" if rec.get("lowest_strike") else "")
            + (f" expiring {rec['next_expiry']}" if rec.get("next_expiry") else "")
            + ". Buy the call back first, or wait for it to expire."
        )
        logger.warning("Sale validation: %s", warnings[-1])
    if not warnings:
        return plan, []
    dropped = [a for a in plan.actions if a not in kept]
    plan = plan.model_copy(update={"actions": kept})
    plan, buy_warnings = resize_unfunded_buys(
        plan, dropped=dropped, positions=positions, account_cash=account_cash
    )
    return _note_adjustments(plan, [*warnings, *buy_warnings]), [*warnings, *buy_warnings]


def _dollars(sizing: str) -> float | None:
    """The first dollar amount in a sizing string ("~$40,200", "$12k")."""
    m = re.search(r"\$\s*(\d[\d,]*\.?\d*)\s*([kK]\b)?", str(sizing or ""))
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return value * 1000 if m.group(2) else value


def _account(sizing: str, accounts: list[str]) -> str | None:
    """The account a sizing string names (longest match), if any."""
    text = str(sizing or "").lower()
    named = [a for a in accounts if a and a.lower() in text]
    return max(named, key=len) if named else None


def _sale_proceeds(action: RebalanceAction, positions: dict[str, dict[str, Any]]) -> float:
    """Roughly what a kept sale brings in, at the broker's price."""
    pos = positions.get(action.ticker) or {}
    units = float(pos.get("units") or 0.0)
    if units <= 0:
        return 0.0
    shares = _requested_shares(action.sizing, units)
    if shares is not None:
        return min(shares, units) * float(pos.get("value") or 0.0) / units
    return _dollars(action.sizing) or 0.0


def resize_unfunded_buys(
    plan: RebalancePlan,
    *,
    dropped: list[RebalanceAction],
    positions: dict[str, dict[str, Any]],
    account_cash: dict[str, float] | None,
) -> tuple[RebalancePlan, list[str]]:
    """Shrink dollar-sized buys to the cash left once `dropped` sales are gone.

    Buys are grouped by the account their sizing names (unnamed ones
    share the total cash). A group's budget is its cash plus what its
    kept sales bring in; when its dollar buys exceed that, each is scaled
    down in proportion, rounded down to $100, and dropped below $100.
    Buys without a dollar figure are left alone.
    """
    if not dropped or account_cash is None:
        return plan, []
    accounts = list(account_cash)
    total_cash = sum(account_cash.values())
    budgets: dict[str | None, float] = {}
    buys: dict[str | None, list[tuple[int, float]]] = {}
    for i, action in enumerate(plan.actions):
        acct = _account(action.sizing, accounts)
        if action.action in SALE_ACTIONS:
            budgets[acct] = budgets.get(acct, 0.0) + _sale_proceeds(action, positions)
        elif action.action in BUY_ACTIONS and (usd := _dollars(action.sizing)):
            buys.setdefault(acct, []).append((i, usd))
    lost = ", ".join(f"{a.ticker} {a.action.lower()}" for a in dropped)
    actions = list(plan.actions)
    removed: set[int] = set()
    warnings: list[str] = []
    for acct, group in buys.items():
        cash = account_cash.get(acct, 0.0) if acct else total_cash
        budget = max(0.0, cash + budgets.get(acct, 0.0) + (budgets.get(None, 0.0) if acct else 0.0))
        wanted = sum(usd for _, usd in group)
        if wanted <= budget + 1.0:
            continue
        scale = budget / wanted
        where = acct or "your accounts"
        for i, usd in group:
            action = actions[i]
            new = (usd * scale) // 100 * 100
            if new < _MIN_BUY_USD:
                removed.add(i)
                warnings.append(
                    f"{action.ticker}: dropped {action.action} of ~${usd:,.0f} — it counted on "
                    f"the {lost} that was dropped, and {where} has no cash left for it."
                )
                continue
            actions[i] = action.model_copy(
                update={
                    "sizing": f"~${new:,.0f} in {where} (cut from: {action.sizing}) — "
                    f"the {lost} it counted on was dropped"
                }
            )
            warnings.append(
                f"{action.ticker}: {action.action} cut from ~${usd:,.0f} to ~${new:,.0f} — the "
                f"{lost} it counted on was dropped; {where} has ~${budget:,.0f} to spend."
            )
    for w in warnings:
        logger.warning("Sale validation: %s", w)
    if not warnings:
        return plan, []
    kept = [a for i, a in enumerate(actions) if i not in removed]
    return plan.model_copy(update={"actions": kept}), warnings


def _note_adjustments(plan: RebalancePlan, warnings: list[str]) -> RebalancePlan:
    """Say in the summary and the plan text what changed after the model
    wrote them, so neither describes a trade that is no longer there."""
    note = "Changed after planning: " + " ".join(warnings)
    full = "ADJUSTED AFTER PLANNING\n" + "\n".join(f"- {w}" for w in warnings)
    return plan.model_copy(
        update={
            "summary": f"{note} Original plan: {plan.summary}".strip(),
            "full_text": f"{full}\n\n{plan.full_text}",
        }
    )
