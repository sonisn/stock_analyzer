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
"""

from __future__ import annotations

import re
from typing import Any

from ..logging import get_logger
from ..models.rebalance import RebalanceAction, RebalancePlan

logger = get_logger(__name__)

SALE_ACTIONS = frozenset({"SELL", "TRIM"})


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
        "call back first, and the plan has to say so and account for its cost.\n"
        + "\n".join(rows)
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
) -> tuple[RebalancePlan, list[str]]:
    """Drop sale actions that would sell promised shares.

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
    return plan.model_copy(update={"actions": kept}), warnings
