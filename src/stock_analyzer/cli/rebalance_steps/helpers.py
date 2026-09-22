"""History, position and subject helpers shared by the rebalance steps."""

from __future__ import annotations

from datetime import date
from typing import Any

from ...db.repository import fetch_recent_holdings_history
from ...db.session import get_session
from ...discover.report import (
    build_rebalance_sections,
)
from ...logging import get_logger

logger = get_logger("stock_analyzer.cli.rebalance")


_build_rebalance_sections = build_rebalance_sections


def _build_history_block(db_path: str, *, n_runs: int = 3) -> str:
    """Reach into the discover DB and produce a compact `Previous decisions`
    block for the rebalancer prompt. Per-holding format:

        AAPL: HOLD-8 (2026-05-05) -> HOLD-7 (2026-05-10) -> today

    Oldest first so the LLM sees chronological drift left-to-right. Returns
    an empty string if no history exists (first run, fresh DB, or every
    prior run was a discover-only run).
    """
    try:
        with get_session(db_path) as session:
            history = fetch_recent_holdings_history(session, n_runs=n_runs)
    except Exception as e:
        logger.warning("history fetch failed (%s) — proceeding without it", e)
        return ""
    if not history:
        return ""
    lines: list[str] = []
    for ticker in sorted(history.keys()):
        entries = history[ticker]
        parts: list[str] = []
        for e in entries:
            verdict = e.get("verdict") or "?"
            conf = e.get("confidence")
            run_at = (e.get("run_at") or "")[:10]
            label = f"{verdict}-{conf}" if conf is not None else verdict
            parts.append(f"{label} ({run_at})")
        parts.append("today")
        lines.append(f"{ticker}: {' -> '.join(parts)}")
    return "\n".join(lines)


def _aggregate_positions(
    holdings: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Collapse holdings-across-accounts into one position per ticker."""
    agg: dict[str, dict[str, float]] = {}
    for items in holdings.values():
        for h in items:
            ticker = h.get("ticker")
            units = h.get("units") or 0
            avg = h.get("average_purchase_price") or 0
            if not ticker or not units:
                continue
            cur = agg.setdefault(ticker, {"units": 0.0, "cost": 0.0})
            cur["units"] += float(units)
            cur["cost"] += float(units) * float(avg)
    out: dict[str, dict[str, float]] = {}
    for ticker, v in agg.items():
        if v["units"]:
            out[ticker] = {
                "units": v["units"],
                "avg_buy_price": v["cost"] / v["units"],
                "cost_basis": v["cost"],
            }
    return out


def build_email_subject(
    *, action_count: int, gross_premium_usd: float, plan_failed: bool = False
) -> str:
    """Subject line for the rebalance email. Annotates premium total
    only when WRITE_CALL / SELL_PUT actions produced a non-trivial credit.

    A run whose plan was lost says so in the subject: the body of a failed
    run and the body of a genuine "hold everything" run look the same from
    the inbox, and only one of them is safe to skim past."""
    today = date.today()
    base = f"Portfolio Rebalance — {today.strftime('%b-%d')}"
    if plan_failed:
        return f"{base} — PLAN INCOMPLETE, re-run needed"
    if gross_premium_usd >= 1.0:
        return f"{base} ({action_count} actions + ${gross_premium_usd:,.0f} premium)"
    return base


def _build_position_splits(
    holdings: dict[str, list[dict[str, Any]]],
    account_meta: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Preserve per-account splits AND aggregate totals per ticker.

    Returns dict keyed by ticker with both views — the aggregated total
    (so existing code that reads holdings_positions still works) PLUS a
    list of per-account splits so the reviewer + rebalancer can reason
    about which slice of a position is in a tax-advantaged account
    (zero-tax trim) vs taxable (real tax cost):

      {
        "AAPL": {
          "total_units": 100, "total_cost": 12000, "avg_buy_price": 120,
          "splits": [
            {"account": "Fidelity Brokerage", "tax_status": "taxable",
             "units": 60, "avg_buy_price": 116.67, "cost_basis": 7000},
            {"account": "Fidelity IRA", "tax_status": "tax_advantaged",
             "units": 40, "avg_buy_price": 125.00, "cost_basis": 5000},
          ],
          "has_tax_advantaged": True,
          "has_taxable": True,
          "tax_advantaged_units": 40, "taxable_units": 60,
        },
      }
    """
    raw: dict[str, list[dict[str, Any]]] = {}
    for account_name, items in holdings.items():
        meta = account_meta.get(account_name) or {}
        tax_status = meta.get("tax_status") or "taxable"
        for h in items:
            ticker = h.get("ticker")
            units = h.get("units") or 0
            avg = h.get("average_purchase_price") or 0
            if not ticker or not units:
                continue
            raw.setdefault(ticker, []).append(
                {
                    "account": account_name,
                    "tax_status": tax_status,
                    "units": float(units),
                    "avg_buy_price": float(avg),
                    "cost_basis": float(units) * float(avg),
                }
            )

    out: dict[str, dict[str, Any]] = {}
    for ticker, splits in raw.items():
        total_units = sum(s["units"] for s in splits)
        total_cost = sum(s["cost_basis"] for s in splits)
        if not total_units:
            continue
        ta_units = sum(s["units"] for s in splits if s["tax_status"] == "tax_advantaged")
        tx_units = sum(s["units"] for s in splits if s["tax_status"] == "taxable")
        out[ticker] = {
            "total_units": total_units,
            "total_cost": total_cost,
            "avg_buy_price": total_cost / total_units if total_units else 0,
            "splits": splits,
            "has_tax_advantaged": ta_units > 0,
            "has_taxable": tx_units > 0,
            "tax_advantaged_units": ta_units,
            "taxable_units": tx_units,
        }
    return out
