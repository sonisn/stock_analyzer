"""Sanity-check the Ranker's stated scenarios against hard data.

Catches hallucination/miscalibration cheaply — pure arithmetic against
data the pipeline already fetched (current price, realized volatility),
no extra LLM call. This does not reject picks; it surfaces warnings
alongside them so a human (or a future automated gate) can weigh in.
"""

from __future__ import annotations

import re

from ..models.llm import RankerPick
from ..models.market import RealizedVolatility

# How many "horizon-scaled standard deviations" a bull target may imply
# before being flagged. 4 sigma over the horizon is already an extreme
# outcome; beyond that it's worth a second look rather than automatic trust.
_MAX_SIGMA_MULTIPLE = 4.0
_DEFAULT_HORIZON_MONTHS = 9.0  # midpoint of the old "6-12 months" horizon
_IMPLAUSIBLE_RETURN_PCT = 200.0  # catches unit-confusion (e.g. 250 vs 2.5)
_IMPLAUSIBLE_ANNUAL_PCT = 100.0  # the same, for per-year targets


def _is_annualized(time_horizon: str) -> bool:
    """Multi-year horizons ("3-5 years") state targets as %/yr; the old
    "6-12 months" ones state a total return over the whole horizon."""
    return "year" in (time_horizon or "").lower()


def _horizon_months(time_horizon: str) -> float:
    """Best-effort parse of '6-12 months' / '3-5 years' style strings.
    Falls back to the old 6-12 month default when unparseable."""
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", time_horizon or "")]
    if not numbers:
        return _DEFAULT_HORIZON_MONTHS
    mid = sum(numbers) / len(numbers)
    return mid * 12 if _is_annualized(time_horizon) else mid


def validate_pick_scenarios(
    pick: RankerPick,
    price: float | None,
    hv: RealizedVolatility | None,
) -> list[str]:
    """Warnings when a pick's stated scenarios look inconsistent with the
    hard data the pipeline already has (current price, realized vol).

    Pure arithmetic — no LLM call. Returns [] when nothing looks off, or
    when there isn't enough data (missing price/vol) to check at all.
    """
    warnings: list[str] = []

    annualized = _is_annualized(pick.time_horizon)
    per = "/yr" if annualized else ""
    limit = _IMPLAUSIBLE_ANNUAL_PCT if annualized else _IMPLAUSIBLE_RETURN_PCT
    if price is not None and price > 0:
        for scenario in pick.scenarios:
            if abs(scenario.target_return_pct) > limit:
                warnings.append(
                    f"{pick.ticker}: {scenario.label} target_return_pct="
                    f"{scenario.target_return_pct:+.0f}%{per} is implausibly large for a "
                    f"{pick.time_horizon} single-stock scenario — check for a unit error."
                )

    if hv is None or hv.hv_annualized is None or hv.hv_annualized <= 0:
        return warnings

    horizon_years = _horizon_months(pick.time_horizon) / 12.0
    # A total return's spread grows with sqrt(T); an annualized (CAGR)
    # return's shrinks with it — T years of returns averaged per year.
    horizon_sigma_pct = (
        hv.hv_annualized / (horizon_years**0.5) * 100
        if annualized
        else hv.hv_annualized * (horizon_years**0.5) * 100
    )
    if horizon_sigma_pct <= 0:
        return warnings

    for scenario in pick.scenarios:
        if scenario.label != "bull":
            continue
        multiple = abs(scenario.target_return_pct) / horizon_sigma_pct
        if multiple > _MAX_SIGMA_MULTIPLE:
            warnings.append(
                f"{pick.ticker}: bull target {scenario.target_return_pct:+.0f}%{per} over "
                f"{pick.time_horizon} implies ~{multiple:.1f}x the ticker's own "
                f"realized-volatility-implied move for that horizon "
                f"({horizon_sigma_pct:.0f}%, from {hv.hv_annualized:.0%} annualized HV) "
                f"— unusually aggressive; verify the bull_thesis actually justifies it."
            )

    return warnings
