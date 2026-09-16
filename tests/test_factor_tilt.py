"""Named style-factor tilt: reporting-only remap of score_candidate()'s
existing fundamentals/trend leaf sub-scores into 5 named buckets. No
rescoring — pure function over already-computed score_breakdown dicts."""

from __future__ import annotations

from stock_analyzer.discover.factor_tilt import average_factor_tilts, compute_factor_tilt
from stock_analyzer.models.market import RealizedVolatility


def _breakdown(**overrides) -> dict:
    base = {
        "fundamentals": {
            "revenue_growth": 0.0,
            "fcf_yield": 0.0,
            "operating_margin": 0.0,
            "debt_health": 0.0,
        },
        "trend": {
            "rs_6mo": 0.0,
            "entry_zone": 0.0,
            "volume_trend": 0.0,
            "weekly_rsi": 0.0,
            "eps_revisions": 0.0,
        },
        "conviction": {"mentions": 0.0, "source_diversity": 0.0},
    }
    base.update(overrides)
    return base


def _hv(hv_annualized: float) -> RealizedVolatility:
    return RealizedVolatility(ticker="X", hv_annualized=hv_annualized, sample_size=252)


def test_known_input_maps_to_expected_buckets():
    breakdown = _breakdown(
        fundamentals={
            "revenue_growth": 17.0,  # own max -> 100
            "fcf_yield": 5.5,  # half of max 11 -> 50
            "operating_margin": 11.0,
            "debt_health": 6.0,  # quality max 17 -> 100
        },
        trend={
            "rs_6mo": 17.0,
            "entry_zone": 10.0,  # excluded (timing)
            "volume_trend": 5.0,  # excluded (timing)
            "weekly_rsi": 5.0,  # excluded (timing)
            "eps_revisions": 8.0,  # momentum max 25 -> 100
        },
        conviction={"mentions": 6.0, "source_diversity": 4.0},  # excluded entirely
    )
    tilt = compute_factor_tilt(breakdown, None)
    assert tilt["growth"] == 100.0
    assert tilt["value"] == 50.0
    assert tilt["quality"] == 100.0
    assert tilt["momentum"] == 100.0
    assert "low_vol" not in tilt


def test_missing_hv_omits_low_vol_cleanly():
    tilt = compute_factor_tilt(_breakdown(), None)
    assert "low_vol" not in tilt
    assert set(tilt) == {"growth", "value", "quality", "momentum"}


def test_zero_hv_annualized_omits_low_vol():
    tilt = compute_factor_tilt(_breakdown(), _hv(0.0))
    assert "low_vol" not in tilt


def test_low_vol_present_when_hv_available():
    tilt = compute_factor_tilt(_breakdown(), _hv(0.10))
    assert tilt["low_vol"] == 100.0
    tilt_high = compute_factor_tilt(_breakdown(), _hv(0.50))
    assert tilt_high["low_vol"] == 0.0


def test_negative_momentum_floored_at_zero():
    breakdown = _breakdown(trend={"rs_6mo": 0.0, "eps_revisions": -3.0})
    tilt = compute_factor_tilt(breakdown, None)
    assert tilt["momentum"] == 0.0


def test_empty_score_breakdown_returns_empty_dict():
    assert compute_factor_tilt(None, None) == {}
    assert compute_factor_tilt({}, None) == {}


def test_average_factor_tilts_across_picks():
    tilts = [
        {"growth": 100.0, "value": 0.0, "quality": 50.0, "momentum": 20.0},
        {"growth": 0.0, "value": 100.0, "quality": 50.0, "momentum": 40.0},
    ]
    avg = average_factor_tilts(tilts)
    assert avg["growth"] == 50.0
    assert avg["value"] == 50.0
    assert avg["quality"] == 50.0
    assert avg["momentum"] == 30.0


def test_average_factor_tilts_handles_partial_low_vol():
    tilts = [
        {"growth": 100.0, "low_vol": 80.0},
        {"growth": 0.0},  # no low_vol for this pick
    ]
    avg = average_factor_tilts(tilts)
    assert avg["growth"] == 50.0
    assert avg["low_vol"] == 80.0  # averaged over the one pick that has it


def test_average_factor_tilts_empty_input():
    assert average_factor_tilts([]) == {}
