"""Named style-factor tilts for reporting — remaps *existing* score_candidate()
leaf sub-scores (screen.py) into 5 named buckets. No new scoring, no weight
changes; this is a reporting-only lens on data already computed.

Timing leaves (entry_zone, weekly_rsi, volume_trend) and attention leaves
(mentions, source_diversity) are deliberately excluded — not style-factor
exposure. Named "style factor tilt" throughout to avoid confusion with
calibration.py's unrelated flattened score_breakdown similarity vector,
which also uses "factor" in its naming.
"""

from __future__ import annotations

from typing import Any

from ..models.market import RealizedVolatility

# Low-vol bucket has no fixed scoring-formula max (unlike the other four,
# each bounded by score_candidate()'s own point budget), so it's normalized
# against a fixed realized-volatility band instead: <=10% annualized scores
# 100, >=50% scores 0.
_LOW_VOL_FLOOR = 0.10
_LOW_VOL_CEIL = 0.50


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_factor_tilt(
    score_breakdown: dict[str, Any] | None,
    hv: RealizedVolatility | None,
) -> dict[str, float]:
    """Map score_candidate()'s fundamentals/trend leaf sub-scores into 5
    named style-factor buckets, each normalized 0-100 against its own
    point budget in the scoring formula. Missing groups/leaves default to
    0. `low_vol` is omitted entirely when `hv` is unavailable."""
    if not score_breakdown:
        return {}
    fund = score_breakdown.get("fundamentals") or {}
    trend = score_breakdown.get("trend") or {}

    tilt: dict[str, float] = {}
    tilt["growth"] = round(_clamp((fund.get("revenue_growth") or 0) / 17 * 100, 0, 100), 1)
    tilt["value"] = round(_clamp((fund.get("fcf_yield") or 0) / 11 * 100, 0, 100), 1)
    quality_raw = (fund.get("operating_margin") or 0) + (fund.get("debt_health") or 0)
    tilt["quality"] = round(_clamp(quality_raw / 17 * 100, 0, 100), 1)
    momentum_raw = max(0.0, (trend.get("rs_6mo") or 0) + (trend.get("eps_revisions") or 0))
    tilt["momentum"] = round(_clamp(momentum_raw / 25 * 100, 0, 100), 1)

    if hv is not None and hv.hv_annualized:
        low_vol = (_LOW_VOL_CEIL - hv.hv_annualized) / (_LOW_VOL_CEIL - _LOW_VOL_FLOOR) * 100
        tilt["low_vol"] = round(_clamp(low_vol, 0, 100), 1)

    return tilt


def average_factor_tilts(tilts: list[dict[str, float]]) -> dict[str, float]:
    """Portfolio-level tilt: mean of each bucket across picks that have it.
    A bucket present on only some picks (e.g. low_vol, when HV was
    unavailable for others) is averaged over just the picks that have it."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for tilt in tilts:
        for bucket, value in tilt.items():
            sums[bucket] = sums.get(bucket, 0.0) + value
            counts[bucket] = counts.get(bucket, 0) + 1
    return {bucket: round(sums[bucket] / counts[bucket], 1) for bucket in sums}
