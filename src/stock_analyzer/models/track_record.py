"""Pydantic models for the track-record measurement pipeline.

Captures one scored decision (``PickReturn``) — buy / hold / trim / sell —
plus per-direction stats (``DirectionStats``), per-Opus-model breakdown
(``ModelStats``), and the top-level aggregate (``TrackRecord``) used by
the report header and the ranker prompt.

Sign convention for ``alpha_pct``: positive always means "the call was
right". BUY and HOLD use ``alpha = stock_ret - spy_ret`` (the holding
direction — vindicated when the stock outperforms SPY). TRIM and SELL
flip the sign — vindicated when the stock underperforms SPY after the
verdict.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Direction = Literal["buy", "hold", "trim", "sell"]


class Quote(BaseModel):
    """One yfinance price snapshot: pick-date close and measurement-date
    close (renamed from ``_Quote`` now that it's part of the public
    model surface)."""

    model_config = ConfigDict(frozen=True)

    pick_price: float | None
    measured_price: float | None


class PickReturn(BaseModel):
    """One scored decision — its realized return and how it compared to SPY.

    ``direction`` is one of buy / hold / trim / sell. ``alpha_pct`` is
    sign-adjusted so positive always means "the call was right".
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
    alpha_pct: float | None
    is_mature: bool


class DirectionStats(BaseModel):
    """Aggregate stats for one direction (buy / hold / trim / sell)."""

    model_config = ConfigDict(frozen=True)

    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    winners: int
    losers: int
    flats: int
    sharpe: float | None  # None when n_mature < 5 or stdev <= 0.001


class ModelStats(BaseModel):
    """Per-Opus-model performance for BUY decisions only."""

    model_config = ConfigDict(frozen=True)

    opus_model: str
    n_mature: int
    mean_alpha_pct: float | None
    sharpe: float | None


class TrackRecord(BaseModel):
    """Aggregate summary of mature decisions over the lookback window.

    Top-level ``mean_*`` / ``winners`` / ``losers`` / ``flats`` cover ALL
    mature decisions across every direction. ``buy_stats`` / ``hold_stats`` /
    ``trim_stats`` / ``sell_stats`` break it down. ``sell_stats`` is
    SELL-only (TRIM moved to its own field); ``model_breakdown`` carries
    BUY-only per-Opus-model rows for models with n_mature >= 3.
    """

    model_config = ConfigDict(frozen=True)

    n_picks_total: int
    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    winners: int
    losers: int
    flats: int
    overall_sharpe: float | None

    buy_stats: DirectionStats
    hold_stats: DirectionStats
    trim_stats: DirectionStats
    sell_stats: DirectionStats          # SELL-only (was SELL+TRIM bundled).

    model_breakdown: list[ModelStats]

    picks: list[PickReturn]
    pending: list[PickReturn]


__all__ = [
    "Direction",
    "Quote",
    "PickReturn",
    "DirectionStats",
    "ModelStats",
    "TrackRecord",
]
