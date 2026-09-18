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


def apply_stop_loss_overrides(
    reviews: dict[str, HoldingReview],
    positions: dict[str, dict[str, Any]],
    technicals: dict[str, dict[str, Any]],
    *,
    hard_stop_pct: float = -20.0,
) -> tuple[dict[str, HoldingReview], list[str]]:
    """Deterministic backstop for the Reviewer's own soft DOWNTREND
    OVERRIDE prompt rule (reviewer.py): a HOLD verdict on a position down
    `hard_stop_pct` or worse from cost basis is mechanically escalated to
    TRIM 25%, regardless of what the LLM's reasoning argued.

    This is a backstop for exactly the case the soft prompt rule already
    flags as serious (`unrealized_pnl_pct <= -20%`) but where the LLM
    chose to stay HOLD anyway — not a redundant second trigger. TRIM/SELL
    verdicts the LLM already chose are left untouched.

    Same compute-then-force-correct shape as cc_validation.py::
    validate_option_writes and reviewer.py::_repair_verdict_inconsistencies
    — mutates frozen Pydantic output via model_copy.
    """
    updated: dict[str, HoldingReview] = {}
    warnings: list[str] = []
    for ticker, review in reviews.items():
        if review.verdict != "HOLD":
            updated[ticker] = review
            continue
        pos = positions.get(ticker)
        tech = technicals.get(ticker) or {}
        current = tech.get("price")
        avg = pos.get("avg_buy_price") if pos else None
        pnl_pct = _compute_unrealized_pnl_pct(current, avg)
        if pnl_pct is None or pnl_pct > hard_stop_pct:
            updated[ticker] = review
            continue
        note = (
            f"MECHANICAL STOP-LOSS: down {pnl_pct:.0f}% from cost basis — auto-escalated from HOLD"
        )
        warnings.append(f"{ticker}: {note}")
        logger.warning("Stop-loss override %s: %s", ticker, note)
        updated[ticker] = review.model_copy(
            update={
                "verdict": "TRIM",
                "trim_pct": 25.0,
                "reasoning": f"{review.reasoning} {note}",
            }
        )
    return updated, warnings
