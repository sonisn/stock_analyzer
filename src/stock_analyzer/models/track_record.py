"""Pydantic models for the track-record measurement pipeline.

Captures one scored decision (``PickReturn``) — buy or sell — plus the
per-direction stats (``DirectionStats``) and the top-level aggregate
(``TrackRecord``) used by the report header and the ranker prompt.

Sign convention for ``alpha_pct``: positive always means "the call was
right" — for BUY picks ``alpha = stock_ret - spy_ret``; for SELL/TRIM
calls the raw difference is flipped so a stock that underperformed
SPY after a SELL still scores positive alpha.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Direction = Literal["buy", "sell"]


class Quote(BaseModel):
    """One yfinance price snapshot: pick-date close and measurement-date
    close (renamed from ``_Quote`` now that it's part of the public
    model surface)."""

    model_config = ConfigDict(frozen=True)

    pick_price: float | None
    measured_price: float | None


class PickReturn(BaseModel):
    """One scored decision — its realized return and how it compared to SPY.

    ``direction`` distinguishes buy picks (from discover) from sell/trim calls
    (from rebalance). ``alpha_pct`` is sign-flipped for sells so positive
    alpha always means "the call was right".
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    pick_date: str          # ISO yyyy-mm-dd
    age_days: int
    direction: Direction = "buy"
    pick_price: float | None
    measured_price: float | None
    pick_return_pct: float | None
    spy_return_pct: float | None
    alpha_pct: float | None  # direction-aware: positive = right call
    is_mature: bool          # >= _MIN_AGE_DAYS old


class DirectionStats(BaseModel):
    """Aggregate stats for one direction (buy or sell)."""

    model_config = ConfigDict(frozen=True)

    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    winners: int
    losers: int
    flats: int


class TrackRecord(BaseModel):
    """Aggregate summary of mature decisions over the lookback window.

    Top-level ``mean_*`` / ``winners`` / ``losers`` / ``flats`` cover ALL
    mature decisions (buys + sells) so existing consumers keep working.
    ``buy_stats`` / ``sell_stats`` break it down per direction.
    """

    model_config = ConfigDict(frozen=True)

    n_picks_total: int
    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    winners: int       # mature decisions where alpha > 0
    losers: int        # mature decisions where alpha < 0
    flats: int         # mature decisions where alpha ≈ 0
    buy_stats: DirectionStats
    sell_stats: DirectionStats
    picks: list[PickReturn]
    pending: list[PickReturn]


__all__ = [
    "Direction",
    "Quote",
    "PickReturn",
    "DirectionStats",
    "TrackRecord",
]
