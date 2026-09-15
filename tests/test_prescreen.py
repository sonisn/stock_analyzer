"""The prescreen gate: what gets fetched, and what the screen says about
names eliminated before their fundamentals were ever fetched."""

from __future__ import annotations

from unittest.mock import MagicMock

from stock_analyzer.cli.discover import DiscoverPipeline
from stock_analyzer.config import Settings
from stock_analyzer.discover.screen import passes_hard_filter, passes_trend_gate

UPTREND = {
    "price": 100.0,
    "sma_50": 95.0,
    "sma_200": 85.0,
    "above_200dma": True,
    "ma_alignment_50_200": True,
    "rs_6mo": 0.08,
    "dist_from_52w_high": -0.10,
}

DOWNTREND = {
    **UPTREND,
    "above_200dma": False,
    "ma_alignment_50_200": False,
    "rs_6mo": -0.20,
    "dist_from_52w_high": -0.55,
}

GOOD_FUNDAMENTALS = {
    "market_cap": 50e9,
    "revenue_growth_yoy": 0.15,
    "operating_cash_flow": 5e9,
    "debt_to_equity": 0.5,
    "sector": "Technology",
}


def _pipeline(tickers, technicals, universe=None, **settings_kwargs):
    pipe = DiscoverPipeline(Settings(**settings_kwargs))  # type: ignore[call-arg]
    pipe.state["tickers"] = list(tickers)
    pipe.state["technicals"] = technicals
    pipe.state["universe"] = universe or {
        t: {"sources": ["index"], "conviction": 0, "in_base_universe": True} for t in tickers
    }
    return pipe


def test_trend_gate_matches_the_hard_filter_on_trend_rules():
    """The gate exists to pre-decide part of the hard filter, so anything
    it rejects must also be rejected by the full filter with good
    fundamentals — otherwise the prescreen would drop real candidates."""
    assert passes_trend_gate(UPTREND)[0]
    assert not passes_trend_gate(DOWNTREND)[0]
    assert passes_hard_filter(GOOD_FUNDAMENTALS, UPTREND)[0]
    assert not passes_hard_filter(GOOD_FUNDAMENTALS, DOWNTREND)[0]


def test_trend_gate_rejects_missing_technicals():
    ok, reasons = passes_trend_gate(None)
    assert not ok
    assert reasons == ["no technicals data"]


def test_prescreen_keeps_only_uptrending_names():
    pipe = _pipeline(["AAA", "BBB"], {"AAA": UPTREND, "BBB": DOWNTREND})
    pipe.step_prescreen(MagicMock())
    assert pipe.state["screen_tickers"] == ["AAA"]
    assert "BBB" in pipe.state["prescreen_reasons"]


def test_prescreen_keeps_holdings_and_watchlist_regardless_of_trend():
    """A holding that broke down is exactly the name worth an opinion."""
    universe = {
        "AAA": {"sources": ["index"], "conviction": 0, "in_base_universe": True},
        "OWNED": {"sources": ["index", "holding"], "conviction": 0, "in_base_universe": True},
        "WATCHED": {"sources": ["watchlist"], "conviction": 0, "in_base_universe": True},
    }
    pipe = _pipeline(
        ["AAA", "OWNED", "WATCHED"],
        {"AAA": DOWNTREND, "OWNED": DOWNTREND, "WATCHED": DOWNTREND},
        universe,
    )
    pipe.step_prescreen(MagicMock())
    assert set(pipe.state["screen_tickers"]) == {"OWNED", "WATCHED"}


def test_prescreen_caps_survivors_by_relative_strength():
    tickers = ["W", "X", "Y", "Z"]
    technicals = {
        t: {**UPTREND, "rs_6mo": rs}
        for t, rs in zip(tickers, [0.01, 0.40, 0.30, 0.02], strict=True)
    }
    pipe = _pipeline(tickers, technicals, discover_max_screen_candidates=2)
    pipe.step_prescreen(MagicMock())
    assert set(pipe.state["screen_tickers"]) == {"X", "Y"}
    assert pipe.state["prescreen_reasons"]["W"] == [
        "below the relative-strength cap for deep analysis"
    ]


def test_screen_reports_the_real_reason_for_prescreened_names():
    """Without this, every gated name shows up as 'no fundamentals data',
    which reads as a fetch failure rather than a filter decision."""
    pipe = _pipeline(["AAA", "BBB"], {"AAA": UPTREND, "BBB": DOWNTREND})
    pipe.step_prescreen(MagicMock())
    pipe.state["fundamentals"] = {"AAA": GOOD_FUNDAMENTALS}
    pipe.state["eps_revisions"] = {}
    pipe.state["sector_rotation"] = {}
    pipe.step_screen(MagicMock())

    by_ticker = {c["ticker"]: c for c in pipe.state["candidates"]}
    assert by_ticker["AAA"]["passed_filter"]
    assert not by_ticker["BBB"]["passed_filter"]
    assert "no fundamentals data" not in by_ticker["BBB"]["fail_reasons"]
    assert any("200DMA" in r for r in by_ticker["BBB"]["fail_reasons"])
