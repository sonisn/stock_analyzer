"""Holdings review payload assembly for the rebalance pipeline."""
from __future__ import annotations

from typing import Any

from .tax_lot_helper import enrich_tax_lots_with_impact


def _trim(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def build_holding_review_payloads(
    *,
    positions: dict[str, dict[str, Any]],
    fund: dict[str, dict[str, Any]],
    tech: dict[str, dict[str, Any]],
    rfs: dict[str, dict[str, Any]],
    insider_selling: dict[str, int],
    finnhub_signals: dict[str, Any],
    eps_revisions: dict[str, Any],
    position_splits: dict[str, dict[str, Any]],
    account_meta: dict[str, dict[str, Any]],
    tax_lots_raw: dict[str, Any],
    share_trades: dict[str, Any],
    holdings_quarterly_mda: dict[str, dict[str, Any]],
    holdings_peers: dict[str, Any],
    holdings_transcripts: dict[str, dict[str, Any]],
    news: dict[str, list[dict[str, Any]]],
    risk_factors_chars: int,
    quarterly_mda_chars: int,
    transcript_chars: int,
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for ticker, pos in positions.items():
        t = tech.get(ticker) or {}
        current = t.get("price")
        avg = pos["avg_buy_price"]
        units = pos["units"]
        pnl = None
        pnl_pct = None
        if current and avg:
            pnl = (current - avg) * units
            pnl_pct = (current - avg) / avg * 100
        fh = finnhub_signals.get(ticker) or {}
        insider_activity: Any = fh.get("insider_activity") or {
            "mention_count": insider_selling.get(ticker, 0),
        }
        splits_info = position_splits.get(ticker) or {}
        payloads[ticker] = {
            "position": {
                "units": units,
                "avg_buy_price": avg,
                "current_price": current,
                "cost_basis": pos["cost_basis"],
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": pnl_pct,
                "account_splits": splits_info.get("splits") or [],
                "tax_advantaged_units": splits_info.get("tax_advantaged_units", 0),
                "taxable_units": splits_info.get("taxable_units", 0),
                "has_tax_advantaged": splits_info.get("has_tax_advantaged", False),
                "has_taxable": splits_info.get("has_taxable", True),
            },
            "fundamentals": fund.get(ticker) or {},
            "technicals": t,
            "insider_activity": insider_activity,
            "earnings_surprise_history": fh.get("earnings_surprise") or [],
            "recommendation_trend": fh.get("recommendation_trend") or [],
            "analyst_price_targets": fh.get("price_targets") or {},
            "eps_revisions": eps_revisions.get(ticker) or {},
            "share_trades": share_trades.get(ticker),
            "risk_factors_10k": _trim(
                (rfs.get(ticker) or {}).get("risk_factors"),
                risk_factors_chars,
            ),
            "quarterly_mda": _trim(
                (holdings_quarterly_mda.get(ticker) or {}).get("mda"),
                quarterly_mda_chars,
            ),
            "peers": holdings_peers.get(ticker),
            "earnings_transcript": _trim(
                (holdings_transcripts.get(ticker) or {}).get("snippet"),
                transcript_chars,
            ),
            "news": news.get(ticker, []),
            "tax_lots": enrich_tax_lots_with_impact(
                tax_lots_raw.get(ticker) or {},
                current or 0.0,
                account_meta,
            ),
        }
    return payloads
