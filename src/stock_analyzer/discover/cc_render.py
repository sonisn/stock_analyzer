"""Deterministic compute for the CC report sections.

The renderer (HTML/PDF) reads these dicts directly — it does NOT parse
Opus's prose, so the box values are always internally consistent. Opus's
narrative remains in `full_text` for human context.
"""
from __future__ import annotations

from typing import Any

from .cc_eligibility import RoundLotCoverage
from .rebalance_schema import RebalancePlan


def compute_premium_income(
    plan: RebalancePlan,
    *,
    slippage_buffer: float,
) -> dict[str, Any]:
    """Compute the Premium Income box content.

    Returns rows + totals (gross / buffer / deployable). Empty rows list
    is valid (when plan has no option_writes).
    """
    rows: list[dict[str, Any]] = []
    gross = 0.0
    for ow in plan.option_writes:
        premium = ow.contracts * ow.est_premium_per_share * 100.0
        gross += premium
        rows.append({
            "ticker": ow.ticker,
            "strike": ow.strike,
            "expiry": ow.expiry,
            "contracts": ow.contracts,
            "premium_usd": premium,
            "delta": ow.delta,
            "assignment_pct": int(round(ow.assignment_probability * 100)),
        })
    buffer = round(gross * slippage_buffer, 2)
    return {
        "rows": rows,
        "gross_premium_usd": round(gross, 2),
        "slippage_buffer_usd": buffer,
        "deployable_premium_usd": round(gross - buffer, 2),
    }


def compute_premium_deployment(
    plan: RebalancePlan,
    *,
    cash_balance: float | None,
    slippage_buffer: float,
    stub_consolidation_usd: float = 0.0,
) -> dict[str, Any]:
    """Compute the Premium → Deployment box content.

    `deployments` lists every ADD/BUY plus every TRIM whose sizing
    contains 'stub' (so the reader sees the consolidation pair in
    context).
    """
    inc = compute_premium_income(plan, slippage_buffer=slippage_buffer)
    deployable = inc["deployable_premium_usd"]
    cash = float(cash_balance or 0.0)
    total = deployable + cash + stub_consolidation_usd
    deployments: list[dict[str, str]] = []
    for a in plan.actions:
        if a.action in ("ADD", "BUY"):
            deployments.append({
                "ticker": a.ticker, "action": a.action, "sizing": a.sizing,
            })
        elif a.action == "TRIM" and "stub" in a.sizing.lower():
            deployments.append({
                "ticker": a.ticker, "action": "TRIM", "sizing": a.sizing,
            })
    return {
        "gross_premium_usd": inc["gross_premium_usd"],
        "slippage_buffer_usd": inc["slippage_buffer_usd"],
        "deployable_premium_usd": deployable,
        "existing_cash_usd": cash,
        "stub_consolidation_usd": stub_consolidation_usd,
        "total_dry_powder_usd": round(total, 2),
        "deployments": deployments,
    }


def compute_round_lot_summary(
    coverage: dict[str, RoundLotCoverage],
) -> dict[str, Any]:
    """Compute the Round-Lot Coverage table.

    Only holdings with a non-zero stub are rendered. Sorted by stub
    dollar value descending so the user's eye lands on the biggest
    consolidation candidates first.
    """
    rows: list[dict[str, Any]] = []
    pool = 0.0
    for ticker in coverage:
        rec = coverage[ticker]
        if rec.stub_shares == 0:
            continue
        rows.append({
            "ticker": rec.ticker,
            "shares": rec.shares,
            "round_lots": rec.round_lots,
            "round_lot_shares": rec.round_lots * 100,
            "stub_shares": rec.stub_shares,
            "stub_dollar_value": rec.stub_dollar_value,
            "to_next_lot_shares": rec.to_next_lot_shares,
            "to_next_lot_cost": rec.to_next_lot_cost,
        })
        pool += rec.stub_dollar_value
    rows.sort(key=lambda r: r["stub_dollar_value"], reverse=True)
    return {"rows": rows, "stub_pool_total_usd": round(pool, 2)}
