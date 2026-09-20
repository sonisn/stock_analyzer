"""The exchanges that trade before New York opens.

Context for a 3-5 year holder, not a trading signal: the trailing windows
lead and the overnight move is reported last.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_analyzer.data.world_markets import (
    MARKETS,
    Market,
    fetch_world_markets,
    world_markets_text,
    world_signals,
)


def _row(name, region, *, d1=None, m1=None, m6=None, y1=None, bears_on=(), note=""):
    return {
        "symbol": name,
        "name": name,
        "region": region,
        "bears_on": list(bears_on),
        "note": note,
        "1d": d1,
        "1mo": m1,
        "6mo": m6,
        "1y": y1,
    }


def test_a_big_overnight_move_is_named():
    rows = [_row("KOSPI", "South Korea", d1=2.7), _row("FTSE 100", "UK", d1=0.4)]
    assert world_signals(rows).notable == ["KOSPI +2.7% (South Korea)"]


def test_a_year_long_decline_is_a_regime_break():
    rows = [_row("Hang Seng", "Hong Kong", y1=-14.0), _row("DAX", "Germany", y1=7.0)]
    breaks = world_signals(rows).regime_breaks
    assert breaks == ["Hang Seng -14% over a year"]


def test_context_is_limited_to_what_is_actually_held():
    rows = [
        _row(
            "Taiwan Weighted",
            "Taiwan",
            m6=40.7,
            bears_on=("TSM", "NVDA", "AVGO"),
            note="foundry capacity",
        ),
        _row("Euro Stoxx 50", "Europe", m6=11.1, bears_on=("ASML",)),
    ]
    context = world_signals(rows, held={"NVDA", "TSM"}).holdings_context
    assert len(context) == 1  # nothing owned reads off Euro Stoxx
    assert "TSM, NVDA" in context[0] or "NVDA, TSM" in context[0]
    assert "foundry capacity" in context[0]


def test_no_holdings_means_no_cross_reads():
    rows = [_row("Taiwan Weighted", "Taiwan", m6=40.7, bears_on=("TSM",))]
    assert world_signals(rows).holdings_context == []


def test_the_prompt_block_names_every_market_and_its_windows():
    rows = [_row("Nikkei 225", "Japan", d1=1.4, m1=-1.8, m6=21.1, y1=51.1)]
    text = world_markets_text(rows)
    assert "Nikkei 225 (Japan)" in text
    assert "6mo +21.1%" in text and "1y +51.1%" in text


def test_the_block_says_so_when_there_is_nothing():
    assert world_markets_text([]) == "World markets: data unavailable."


def test_a_market_that_fails_is_left_out_not_faked(monkeypatch):
    from stock_analyzer.data import world_markets as wm

    frame = pd.DataFrame(
        {"Close": [100.0, 102.0]},
        index=pd.to_datetime(["2026-09-17", "2026-09-18"]),
    )

    def fake_map(fn, symbols, **kwargs):
        for s in symbols:
            yield s, (None if s == "^HSI" else fn(s))

    monkeypatch.setattr(wm.yf_gateway, "map_symbols", fake_map)
    monkeypatch.setattr(wm.yf_gateway, "ticker_call", lambda *a, **k: frame)
    rows = fetch_world_markets((MARKETS[2], MARKETS[3]))  # Taiwan, Hang Seng
    assert [r["name"] for r in rows] == ["Taiwan Weighted"]
    assert rows[0]["1d"] == pytest.approx(2.0)


def test_rows_keep_the_east_to_west_order(monkeypatch):
    # The pool finishes in whatever order the fetches return; the table
    # still has to read the way the trading day runs.
    from stock_analyzer.data import world_markets as wm

    frame = pd.DataFrame({"Close": [1.0, 1.0]}, index=pd.to_datetime(["2026-09-17", "2026-09-18"]))

    def out_of_order(fn, symbols, **kwargs):
        for s in reversed(list(symbols)):
            yield s, fn(s)

    monkeypatch.setattr(wm.yf_gateway, "map_symbols", out_of_order)
    monkeypatch.setattr(wm.yf_gateway, "ticker_call", lambda *a, **k: frame)
    rows = fetch_world_markets(MARKETS[:3])
    assert [r["name"] for r in rows] == ["Nikkei 225", "KOSPI", "Taiwan Weighted"]


def test_every_mapped_ticker_is_a_plausible_symbol():
    # A typo in bears_on silently stops a cross-read from ever firing.
    for market in MARKETS:
        assert isinstance(market, Market)
        for ticker in market.bears_on:
            assert ticker.isupper() and 1 <= len(ticker) <= 6


# --- the email block --------------------------------------------------------------


def test_the_email_block_leads_with_trailing_windows():
    from stock_analyzer.reporting.health import build_portfolio_health, render_world_markets_html

    health = build_portfolio_health(
        {"IRA": [{"ticker": "NVDA", "units": 10, "price": 222.0}]},
        world_markets=[
            _row(
                "Taiwan Weighted",
                "Taiwan",
                d1=0.0,
                m1=4.3,
                m6=40.7,
                y1=92.6,
                bears_on=("TSM", "NVDA"),
                note="foundry capacity",
            ),
        ],
    )
    html = render_world_markets_html(health)
    assert "World markets" in html
    assert html.index("6mo") < html.index("1d")  # regime first, session last
    assert "+40.7%" in html and "foundry capacity" in html
    assert "NVDA" in html  # held, so the cross-read fires


def test_no_world_data_renders_nothing():
    from stock_analyzer.reporting.health import build_portfolio_health, render_world_markets_html

    assert render_world_markets_html(build_portfolio_health({})) == ""
