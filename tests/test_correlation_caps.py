"""Sizer correlation-cap enforcement: deterministic post-LLM check that
combined allocation for a flagged correlated pair never exceeds the cap."""

from __future__ import annotations

from stock_analyzer.discover.sizer import enforce_correlation_caps, enforce_earnings_blackout
from stock_analyzer.models.llm import Allocation, CorrelatedPair, SizerOutput


def _alloc(ticker: str, *, pct: float | None = None, usd: float | None = None) -> Allocation:
    return Allocation.model_construct(
        ticker=ticker,
        allocation_pct=pct,
        allocation_usd=usd,
        rationale="x",
    )


def _output(allocations: list[Allocation], warnings: list[str] | None = None) -> SizerOutput:
    return SizerOutput.model_construct(
        allocations=allocations,
        concentration_warnings=warnings or [],
        full_text="original text",
    )


def test_pair_under_cap_is_untouched():
    output = _output([_alloc("NVDA", pct=15.0), _alloc("AMD", pct=15.0)])
    pairs = [CorrelatedPair(ticker_a="NVDA", ticker_b="AMD", shared_driver="AI capex")]
    result = enforce_correlation_caps(output, pairs)
    assert result is output
    assert result.concentration_warnings == []


def test_pair_over_cap_scaled_down_proportionally_pct():
    output = _output([_alloc("NVDA", pct=25.0), _alloc("AMD", pct=25.0)])
    pairs = [CorrelatedPair(ticker_a="NVDA", ticker_b="AMD", shared_driver="AI capex")]
    result = enforce_correlation_caps(output, pairs, max_combined_pct=35.0)
    by_ticker = {a.ticker: a for a in result.allocations}
    assert by_ticker["NVDA"].allocation_pct == by_ticker["AMD"].allocation_pct
    assert abs(by_ticker["NVDA"].allocation_pct + by_ticker["AMD"].allocation_pct - 35.0) < 1e-6
    assert len(result.concentration_warnings) == 1
    assert "CORRELATION CAP" in result.concentration_warnings[0]


def test_pair_over_cap_scaled_down_proportionally_usd_budget():
    output = _output([_alloc("NVDA", usd=6000.0), _alloc("AMD", usd=4000.0)])
    pairs = [CorrelatedPair(ticker_a="NVDA", ticker_b="AMD", shared_driver="AI capex")]
    # budget 10_000 -> 60% + 40% = 100%, cap at 35% combined
    result = enforce_correlation_caps(output, pairs, cash_budget=10_000.0, max_combined_pct=35.0)
    by_ticker = {a.ticker: a for a in result.allocations}
    total_usd = by_ticker["NVDA"].allocation_usd + by_ticker["AMD"].allocation_usd
    assert abs(total_usd - 3500.0) < 1e-6
    # ratio preserved: NVDA:AMD was 60:40
    assert abs(by_ticker["NVDA"].allocation_usd / total_usd - 0.6) < 1e-6


def test_unknown_ticker_pair_ignored():
    output = _output([_alloc("NVDA", pct=25.0)])
    pairs = [CorrelatedPair(ticker_a="NVDA", ticker_b="MSFT", shared_driver="cloud capex")]
    result = enforce_correlation_caps(output, pairs)
    assert result is output


def test_usd_allocation_without_cash_budget_skipped():
    output = _output([_alloc("NVDA", usd=6000.0), _alloc("AMD", usd=4000.0)])
    pairs = [CorrelatedPair(ticker_a="NVDA", ticker_b="AMD", shared_driver="AI capex")]
    result = enforce_correlation_caps(output, pairs, cash_budget=None)
    assert result is output


def test_existing_warnings_preserved_and_appended():
    output = _output(
        [_alloc("NVDA", pct=25.0), _alloc("AMD", pct=25.0)],
        warnings=["Semiconductors sector at 40% combined"],
    )
    pairs = [CorrelatedPair(ticker_a="NVDA", ticker_b="AMD", shared_driver="AI capex")]
    result = enforce_correlation_caps(output, pairs)
    assert result.concentration_warnings[0] == "Semiconductors sector at 40% combined"
    assert len(result.concentration_warnings) == 2


# --- earnings blackout ------------------------------------------------------

_ALERT = {"NVDA": {"ticker": "NVDA", "earnings_date": "2026-09-20", "days_until": 3}}


def test_pick_reporting_soon_is_capped_to_starter_position():
    output = _output([_alloc("NVDA", pct=30.0), _alloc("AMD", pct=20.0)])
    result = enforce_earnings_blackout(output, _ALERT, max_pct=5.0)
    by_ticker = {a.ticker: a for a in result.allocations}
    assert by_ticker["NVDA"].allocation_pct == 5.0
    assert by_ticker["AMD"].allocation_pct == 20.0  # freed capital not redistributed
    assert "EARNINGS BLACKOUT" in result.concentration_warnings[0]
    assert "2026-09-20" in result.concentration_warnings[0]


def test_blackout_leaves_small_positions_and_unflagged_tickers_alone():
    output = _output([_alloc("NVDA", pct=4.0), _alloc("AMD", pct=40.0)])
    assert enforce_earnings_blackout(output, _ALERT, max_pct=5.0) is output


def test_blackout_caps_usd_allocations_against_budget():
    output = _output([_alloc("NVDA", usd=3000.0)])
    result = enforce_earnings_blackout(output, _ALERT, cash_budget=10_000.0, max_pct=5.0)
    assert result.allocations[0].allocation_usd == 500.0


def test_blackout_skips_usd_allocation_without_budget():
    output = _output([_alloc("NVDA", usd=3000.0)])
    assert enforce_earnings_blackout(output, _ALERT) is output
