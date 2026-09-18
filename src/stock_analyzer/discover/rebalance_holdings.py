"""Holdings review payload assembly for the rebalance pipeline."""

from __future__ import annotations

from typing import Any

from ..logging import get_logger
from ..models.llm import HoldingReview
from .tax_lot_helper import enrich_tax_lots_with_impact

logger = get_logger(__name__)


def _trim(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _compute_unrealized_pnl_pct(current: float | None, avg: float | None) -> float | None:
    if not current or not avg:
        return None
    return (current - avg) / avg * 100


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
    recent_news: dict[str, list[dict[str, Any]]] | None = None,
    thesis_checks: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    recent_news = recent_news or {}
    thesis_by_ticker = {c["ticker"]: c for c in thesis_checks or []}
    payloads: dict[str, dict[str, Any]] = {}
    for ticker, pos in positions.items():
        t = tech.get(ticker) or {}
        current = t.get("price")
        avg = pos["avg_buy_price"]
        units = pos["units"]
        pnl_pct = _compute_unrealized_pnl_pct(current, avg)
        pnl = (current - avg) * units if pnl_pct is not None else None
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
            "recent_news": recent_news.get(ticker, []),
            "news": news.get(ticker, []),
            # Only for holdings that were discover picks: how the original
            # thesis is holding up against its own targets and catalysts.
            "original_pick_thesis_check": thesis_by_ticker.get(ticker),
            "tax_lots": enrich_tax_lots_with_impact(
                tax_lots_raw.get(ticker) or {},
                current or 0.0,
                account_meta,
            ),
        }
    return payloads


def flag_drawdown_reviews(
    reviews: dict[str, HoldingReview],
    positions: dict[str, dict[str, Any]],
    technicals: dict[str, dict[str, Any]],
    *,
    review_pct: float = -20.0,
) -> list[str]:
    """One line per holding down `review_pct` or worse from cost, stating
    the verdict the Reviewer reached on it.

    Holdings are long-term (3-5 year) investments, so a drawdown asks for
    the thesis to be re-underwritten (the Reviewer's DRAWDOWN REVIEW rule),
    not for a mechanical sale — verdicts are never changed here. The lines
    surface in the report so a HOLD on a deep loser is a visible, reasoned
    choice rather than a silent one."""
    out: list[str] = []
    for ticker, review in reviews.items():
        pos = positions.get(ticker)
        current = (technicals.get(ticker) or {}).get("price")
        pnl_pct = _compute_unrealized_pnl_pct(current, pos.get("avg_buy_price") if pos else None)
        if pnl_pct is None or pnl_pct > review_pct:
            continue
        line = (
            f"{ticker}: down {pnl_pct:.0f}% from cost — thesis re-checked, "
            f"reviewer says {review.verdict} (confidence {review.confidence}/10)"
        )
        out.append(line)
        logger.info("Drawdown review %s", line)
    return out
