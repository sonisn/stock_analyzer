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

Every scored row carries the ``horizon_days`` it was measured over, and
stats are only ever aggregated WITHIN one horizon (``HorizonStats``).
Averaging a 15-day outcome against a 90-day one produced a number that
tracked how recently the pipeline had run rather than how good the calls
were, so the horizon is now part of the data rather than an assumption.

``beta_adjusted_alpha_pct`` exists because the screen selects for
high-beta momentum leaders by design (above 200DMA, positive RS, near
52-week highs). Raw excess return over SPY therefore credits market
exposure as stock-selection skill in a rising tape, and reverses in a
drawdown. Beta is estimated on daily returns in the window BEFORE the
decision date, so it never peeks at the outcome it adjusts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Direction = Literal["buy", "hold", "trim", "sell"]

# Why a decision could not be scored. "too_young" is the benign case (not
# enough elapsed time yet); "no_price_data" means the forward price lookup
# came back empty where SPY had data — a delisting, or a bad symbol from
# the news-regex universe. Dropping the latter silently biased the mean
# upward by removing exactly the worst outcomes, so it is now counted and
# surfaced instead.
UnmeasurableReason = Literal["too_young", "no_price_data"]


class PickReturn(BaseModel):
    """One scored decision — its realized return and how it compared to SPY.

    ``direction`` is one of buy / hold / trim / sell. ``alpha_pct`` is
    sign-adjusted so positive always means "the call was right".
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    pick_date: str  # ISO yyyy-mm-dd
    age_days: int
    direction: Direction = "buy"
    # The completed window this row was measured over. 0 marks a "pending"
    # row, whose return is a live mark rather than a finished measurement.
    horizon_days: int = 0
    measured_date: str | None = None
    pick_price: float | None
    measured_price: float | None
    pick_return_pct: float | None
    spy_return_pct: float | None
    alpha_pct: float | None
    # Trailing beta vs SPY estimated on the pre-decision window, and the
    # alpha that remains once the market exposure it implies is removed.
    beta: float | None = None
    beta_adjusted_alpha_pct: float | None = None
    is_mature: bool


class DirectionStats(BaseModel):
    """Aggregate stats for one direction (buy / hold / trim / sell)."""

    model_config = ConfigDict(frozen=True)

    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    # Mean of ret - beta*spy_ret (direction-adjusted). None when no row in
    # the sample had an estimable beta.
    mean_beta_adjusted_alpha_pct: float | None = None
    n_beta_adjusted: int = 0
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


class ProviderStats(BaseModel):
    """Per-provider (claude/gemini/openai) performance for BUY decisions.

    A pick can count toward more than one provider's bucket when multiple
    ranker consensus rounds (different providers) agreed on it — see
    `Pick.voting_providers` and `discover/track_record.py::
    _compute_provider_breakdown`. Only populated for picks made after
    multi-provider consensus rounds existed; legacy single-provider picks
    have no `voting_providers` and are excluded, not bucketed as
    'unknown' the way `ModelStats` handles a missing opus_model.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    n_mature: int
    mean_alpha_pct: float | None
    sharpe: float | None


class UnmeasurableDecision(BaseModel):
    """A decision that could not be scored, and why.

    Reported rather than dropped: a ``no_price_data`` row is usually a
    delisting (the worst possible BUY outcome) or a bad symbol, and
    removing those from the sample silently inflates measured alpha.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    pick_date: str
    direction: Direction
    age_days: int
    reason: UnmeasurableReason


class HorizonStats(BaseModel):
    """Every stat for ONE measurement horizon. Never mixed across horizons.

    ``overall`` deduplicates across directions by ticker (keeping the
    earliest decision) so a name that is both a BUY pick and a HOLD
    verdict is not counted twice in the headline number; the per-direction
    fields keep every decision.
    """

    model_config = ConfigDict(frozen=True)

    horizon_days: int
    overall: DirectionStats
    buy_stats: DirectionStats
    hold_stats: DirectionStats
    trim_stats: DirectionStats
    sell_stats: DirectionStats
    model_breakdown: list[ModelStats]
    provider_breakdown: list[ProviderStats] = []
    decisions: list[PickReturn]


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
    # The horizon the top-level fields and `picks` describe: the primary
    # horizon when it has data, else the longest horizon that does. Always
    # rendered explicitly so a reader never has to guess the window.
    reported_horizon_days: int = 0
    horizons: list[HorizonStats] = []
    n_unmeasurable: int = 0
    unmeasurable: list[UnmeasurableDecision] = []
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
    sell_stats: DirectionStats  # SELL-only (was SELL+TRIM bundled).

    model_breakdown: list[ModelStats]
    provider_breakdown: list[ProviderStats] = []

    picks: list[PickReturn]
    pending: list[PickReturn]


__all__ = [
    "Direction",
    "UnmeasurableReason",
    "PickReturn",
    "DirectionStats",
    "ModelStats",
    "ProviderStats",
    "UnmeasurableDecision",
    "HorizonStats",
    "TrackRecord",
]
