"""Unit tests for discover.screen — pure-math filter + scoring logic.

Only the pure functions are tested here; I/O paths (yfinance, SEC EDGAR,
Tavily) are exercised in the smoke test, not unit tests.
"""
from __future__ import annotations

from stock_analyzer.discover.screen import (
    MAX_DEBT_TO_EQUITY,
    MAX_DRAWDOWN_FROM_52W_HIGH,
    MIN_MARKET_CAP,
    MIN_REVENUE_GROWTH,
    passes_hard_filter,
    score_candidate,
)

# --- helpers ----------------------------------------------------------------


def _good_fundamentals(**overrides):
    base = {
        "market_cap": 50e9,
        "revenue_growth_yoy": 0.15,
        "operating_cash_flow": 5e9,
        "debt_to_equity": 0.5,
        "fcf_yield": 0.04,
        "operating_margin": 0.25,
        "sector": "Technology",
    }
    base.update(overrides)
    return base


def _good_technicals(**overrides):
    base = {
        "price": 100.0,
        "sma_50": 95.0,
        "sma_200": 85.0,
        "above_200dma": True,
        "ma_alignment_50_200": True,
        "rs_3mo": 0.05,
        "rs_6mo": 0.08,
        "dist_from_52w_high": -0.10,
        "volume_trend_20_60": 0.10,
        "weekly_rsi": 55.0,
    }
    base.update(overrides)
    return base


def _universe_entry(**overrides):
    base = {"sources": ["insider", "watchlist"], "conviction": 6}
    base.update(overrides)
    return base


# --- hard filter tests ------------------------------------------------------


def test_passes_with_good_inputs():
    passes, reasons = passes_hard_filter(_good_fundamentals(), _good_technicals())
    assert passes is True
    assert reasons == []


def test_fails_when_missing_data():
    passes, reasons = passes_hard_filter(None, _good_technicals())
    assert passes is False
    assert any("fundamentals" in r for r in reasons)


def test_fails_below_market_cap_threshold():
    passes, reasons = passes_hard_filter(
        _good_fundamentals(market_cap=MIN_MARKET_CAP - 1), _good_technicals()
    )
    assert passes is False
    assert any("market_cap" in r for r in reasons)


def test_fails_below_revenue_growth_threshold():
    passes, reasons = passes_hard_filter(
        _good_fundamentals(revenue_growth_yoy=MIN_REVENUE_GROWTH - 0.01),
        _good_technicals(),
    )
    assert passes is False
    assert any("revenue_growth" in r for r in reasons)


def test_fails_negative_operating_cash_flow():
    passes, reasons = passes_hard_filter(
        _good_fundamentals(operating_cash_flow=-1e9), _good_technicals()
    )
    assert passes is False
    assert any("operating_cash_flow" in r for r in reasons)


def test_fails_high_debt_to_equity():
    passes, reasons = passes_hard_filter(
        _good_fundamentals(debt_to_equity=MAX_DEBT_TO_EQUITY + 0.5),
        _good_technicals(),
    )
    assert passes is False
    assert any("debt_to_equity" in r for r in reasons)


def test_passes_with_unknown_debt_to_equity():
    """Missing D/E shouldn't auto-fail — many tickers genuinely lack the field."""
    passes, _ = passes_hard_filter(
        _good_fundamentals(debt_to_equity=None), _good_technicals()
    )
    assert passes is True


def test_fails_below_200_dma():
    passes, reasons = passes_hard_filter(
        _good_fundamentals(), _good_technicals(above_200dma=False)
    )
    assert passes is False
    assert any("200DMA" in r for r in reasons)


def test_fails_when_50_dma_below_200_dma():
    passes, reasons = passes_hard_filter(
        _good_fundamentals(), _good_technicals(ma_alignment_50_200=False)
    )
    assert passes is False
    assert any("50DMA" in r for r in reasons)


def test_fails_negative_rs_6mo():
    passes, reasons = passes_hard_filter(
        _good_fundamentals(), _good_technicals(rs_6mo=-0.02)
    )
    assert passes is False
    assert any("rs_6mo" in r for r in reasons)


def test_fails_deep_drawdown_from_52w_high():
    passes, reasons = passes_hard_filter(
        _good_fundamentals(),
        _good_technicals(dist_from_52w_high=MAX_DRAWDOWN_FROM_52W_HIGH - 0.01),
    )
    assert passes is False
    assert any("52w drawdown" in r for r in reasons)


# --- score tests ------------------------------------------------------------


def test_score_bounds_total_le_100():
    """No combination of inputs should ever exceed the documented 100-pt cap."""
    perfect_f = _good_fundamentals(
        revenue_growth_yoy=0.50,
        fcf_yield=0.10,
        operating_margin=0.50,
        debt_to_equity=0.0,
    )
    perfect_t = _good_technicals(
        rs_6mo=0.50,
        dist_from_52w_high=-0.10,
        volume_trend_20_60=0.50,
        weekly_rsi=55,
    )
    perfect_u = _universe_entry(
        sources=["insider", "billionaire", "watchlist"], conviction=100
    )
    scored = score_candidate(perfect_f, perfect_t, perfect_u)
    assert 0 <= scored["score"] <= 100


def test_score_zero_on_terrible_inputs():
    """Failing on every soft criterion still produces a valid score >= 0."""
    bad_f = {
        "market_cap": 1e9,
        "revenue_growth_yoy": 0.0,
        "operating_cash_flow": 0,
        "debt_to_equity": 5.0,
        "fcf_yield": -0.05,
        "operating_margin": -0.1,
    }
    bad_t = {
        "rs_6mo": -0.10,
        "dist_from_52w_high": -0.50,
        "volume_trend_20_60": -0.30,
        "weekly_rsi": 90,
    }
    bad_u = {"sources": [], "conviction": 0}
    scored = score_candidate(bad_f, bad_t, bad_u)
    assert scored["score"] >= 0


def test_score_components_sum_to_total():
    """Each component is rounded to 1 decimal; total = sum of components, rounded."""
    scored = score_candidate(_good_fundamentals(), _good_technicals(), _universe_entry())
    comp = scored["components"]
    # Rounding may introduce ±0.1 drift; allow a small tolerance.
    assert abs(
        scored["score"] - (comp["fundamentals"] + comp["trend"] + comp["conviction"])
    ) < 0.3


def test_higher_growth_scores_higher_fundamentals():
    low = score_candidate(
        _good_fundamentals(revenue_growth_yoy=0.10),
        _good_technicals(),
        _universe_entry(),
    )
    high = score_candidate(
        _good_fundamentals(revenue_growth_yoy=0.30),
        _good_technicals(),
        _universe_entry(),
    )
    assert high["components"]["fundamentals"] > low["components"]["fundamentals"]


def test_entry_zone_peaks_around_10pct_pullback():
    """Score is highest in the ideal entry zone (around -10% from 52w high)."""
    extended = score_candidate(
        _good_fundamentals(),
        _good_technicals(dist_from_52w_high=0.0),
        _universe_entry(),
    )
    sweet_spot = score_candidate(
        _good_fundamentals(),
        _good_technicals(dist_from_52w_high=-0.10),
        _universe_entry(),
    )
    deep_pullback = score_candidate(
        _good_fundamentals(),
        _good_technicals(dist_from_52w_high=-0.25),
        _universe_entry(),
    )
    assert sweet_spot["components"]["trend"] > extended["components"]["trend"]
    assert sweet_spot["components"]["trend"] > deep_pullback["components"]["trend"]


def test_more_sources_means_higher_conviction_score():
    one = score_candidate(
        _good_fundamentals(),
        _good_technicals(),
        _universe_entry(sources=["insider"], conviction=2),
    )
    three = score_candidate(
        _good_fundamentals(),
        _good_technicals(),
        _universe_entry(
            sources=["insider", "billionaire", "watchlist"], conviction=2
        ),
    )
    assert three["components"]["conviction"] > one["components"]["conviction"]
