"""Pydantic models for forecast-calibration scoring.

Calibration asks whether the ranker's *stated* numbers — expected return,
conviction, and the bull/base/bear probabilities — bear any relationship to
what actually happened. See `discover/calibration.py` for the measurement,
including why EV error and conviction ordering are measured over different
horizons.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ScenarioLabel = Literal["bull", "base", "bear"]


class EVError(BaseModel):
    """One pick's expected return against what it actually returned."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    pick_date: str
    conviction: int | None
    ev_pct: float
    realized_pct: float
    # realized - EV. Negative means the forecast was too optimistic.
    error_pct: float


class ConvictionBucket(BaseModel):
    """Mean realized alpha for picks in one conviction band.

    The point of the bucket is the ordering between buckets, not the level:
    if high conviction does not out-earn low conviction, the conviction
    score carries no information and should stop being weighted.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    n: int
    mean_alpha_pct: float


class ScenarioReliability(BaseModel):
    """Stated probability vs observed frequency for one scenario label."""

    model_config = ConfigDict(frozen=True)

    label: ScenarioLabel
    n: int
    mean_stated_probability: float
    # Share of scored picks where this scenario's target was closest to the
    # realized return. None when nothing has been attributed yet.
    observed_frequency: float | None
    n_landed: int


class CalibrationRecord(BaseModel):
    """Everything the calibration pass measured."""

    model_config = ConfigDict(frozen=True)

    ev_horizon_days: int = 0
    conviction_horizon_days: int = 0
    n_scored: int = 0
    n_pending: int = 0
    # Picks persisted before the forecast columns existed. Counted so the
    # sample size is never silently smaller than it looks.
    n_without_forecast: int = 0
    mean_ev_error_pct: float | None = None
    median_ev_error_pct: float | None = None
    ev_errors: list[EVError] = []
    conviction_buckets: list[ConvictionBucket] = []
    scenario_reliability: list[ScenarioReliability] = []

    @property
    def is_conviction_monotone(self) -> bool:
        """True when mean alpha rises with conviction across every bucket.

        Buckets arrive in ascending conviction order. Fewer than two
        buckets is reported as monotone — there is no ordering to violate
        yet, and flagging it would cry wolf on a young database.
        """
        alphas = [b.mean_alpha_pct for b in self.conviction_buckets]
        if len(alphas) < 2:
            return True
        return all(b >= a for a, b in zip(alphas, alphas[1:], strict=False))


__all__ = [
    "ScenarioLabel",
    "EVError",
    "ConvictionBucket",
    "ScenarioReliability",
    "CalibrationRecord",
]
