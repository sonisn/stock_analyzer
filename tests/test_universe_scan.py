"""The >= $2B US universe and the gate for companies burning cash."""

from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl

from stock_analyzer.data import universe_base as ub
from stock_analyzer.data import universe_scan as us
from stock_analyzer.discover.screen import passes_hard_filter

TODAY = date(2026, 9, 27)


def _q(
    symbol,
    name="Real Co Inc",
    *,
    cap=5e9,
    price=50.0,
    vol=1e6,
    exch="NMS",
    kind="EQUITY",
    listed="2020-01-02",
):
    return {
        "symbol": symbol,
        "longName": name,
        "exchange": exch,
        "quoteType": kind,
        "marketCap": cap,
        "regularMarketPrice": price,
        "averageDailyVolume3Month": vol,
        "firstTradeDateMilliseconds": int(pd.Timestamp(listed).timestamp() * 1000),
    }


QUOTES = [
    _q("GOOD"),
    _q("OTC", exch="PNK"),  # not a listed exchange
    _q("PENNY", price=3.0),  # small and under $5
    _q("STLA", "Stellantis N.V.", cap=17e9, price=4.5, vol=1e7),  # big: the price rule skips it
    _q("THIN", vol=1e5),  # $5M a day
    _q("NEWIPO", listed="2026-06-01"),
    _q("SPAC", "Rocket Acquisition Corp"),
    _q("CEF", "PIMCO Dynamic Income Fund"),
    _q("GS-PD", "The Goldman Sachs Group, Inc."),  # a preferred
    _q("BRK.B", "Berkshire Hathaway Inc."),  # a share class, kept
    _q("REIT", "Federal Realty Investment Trust"),  # "Trust" is a real business
    _q("ETF", kind="ETF"),
    _q("GOOGL", "Alphabet Inc.", cap=3e12, vol=3e7),
    _q("GOOG", "Alphabet Inc.", cap=3e12, vol=2e7),  # same company, trades less
    _q("NONAME1", ""),  # no name: never merged with another
    _q("NONAME2", ""),
]


def _fake_screen(offset):
    return {"quotes": QUOTES[offset : offset + us.PAGE], "total": len(QUOTES)}


def test_only_tradable_common_stock_survives():
    rows = us.scan(screen=_fake_screen, pause=0)
    kept = us.symbols(us.investable(rows, today=TODAY))
    assert sorted(kept) == ["BRK-B", "GOOD", "GOOGL", "NONAME1", "NONAME2", "REIT", "STLA"]


def test_refresh_defaults_to_the_local_copy_outside_the_repo(tmp_path, monkeypatch):
    local = tmp_path / "home" / "us_2b_universe.txt"
    monkeypatch.setattr(ub, "LOCAL_US_2B", local)
    rows = us.scan(screen=_fake_screen, pause=0)
    assert ub.refresh_us_2b(today=TODAY, rows=rows) == 7 and local.exists()
    assert ub._us_2b_file() == local
    ub.load_base_universe.cache_clear()
    try:
        assert set(ub.load_base_universe()) == {
            "STLA",
            "GOOD",
            "REIT",
            "BRK-B",
            "GOOGL",
            "NONAME1",
            "NONAME2",
        }
    finally:
        ub.load_base_universe.cache_clear()


def test_refresh_writes_the_snapshot_and_the_frame_reads_it(tmp_path, monkeypatch):
    rows = us.scan(screen=_fake_screen, pause=0)
    path = tmp_path / "us_2b.txt"
    assert ub.refresh_us_2b(path, today=TODAY, rows=rows) == 7
    text = path.read_text()
    assert text.startswith("# Every US-listed stock") and "STLA" in text
    monkeypatch.setattr(ub, "_us_2b_file", lambda: path)
    ub.load_base_universe.cache_clear()
    try:
        assert set(ub.load_base_universe()) == {
            "STLA",
            "GOOD",
            "REIT",
            "BRK-B",
            "GOOGL",
            "NONAME1",
            "NONAME2",
        }
        assert "AAPL" in ub.sp500()  # the model's universe is untouched
        monkeypatch.setenv("DISCOVER_UNIVERSE", "sp500")
        ub.load_base_universe.cache_clear()
        assert "AAPL" in ub.load_base_universe()
    finally:
        ub.load_base_universe.cache_clear()


def _passes(f):
    tech = {
        "above_200dma": True,
        "ma_alignment_50_200": True,
        "rs_6mo": 0.1,
        "dist_from_52w_high": -0.05,
    }
    base = {
        "market_cap": 7e9,
        "revenue_growth_yoy": 0.5,
        "debt_to_equity": 0.1,
        "analyst_count": 8,
        "operating_cash_flow": 5e8,
        "free_cash_flow": 3e8,
        "return_on_equity": 0.18,
    }
    return passes_hard_filter({**base, **f}, tech, trend_gate="off")


def test_a_name_from_outside_the_universe_meets_the_same_quality_bar():
    assert _passes({}) == (True, [])
    ok, why = _passes({"operating_cash_flow": -1e8, "free_cash_flow": -2e8})
    assert not ok and why == [
        "operating_cash_flow=-100000000.0 not positive",
        "free_cash_flow=-200000000.0 not positive",
    ]
    assert _passes({"return_on_equity": 0.06})[1] == ["return_on_equity=0.06 < 10%"]
    assert _passes({"return_on_equity": None})[0] is False  # negative equity or no data
    assert _passes({"free_cash_flow": None})[0] is False


def test_symbols_use_the_package_spelling():
    rows = pl.DataFrame({"symbol": ["BRK.B", "nvda"]})
    assert us.symbols(rows) == ["BRK-B", "NVDA"]


def test_the_quality_rules_go_to_yahoo_in_the_query(monkeypatch):
    import yfinance as yf

    sent: list[str] = []

    def fake_screen(query, **kw):
        sent.append(str(query.to_dict()))
        return {"quotes": QUOTES[:1], "total": 1}

    monkeypatch.setattr(yf, "screen", fake_screen)
    assert us.symbols(us.scan(pause=0)) == ["GOOD"]
    for _, field, _ in us.QUALITY_RULES:
        assert field in sent[0]
    us.scan(quality=False, pause=0)
    assert "returnonequity" not in sent[1]
