"""`analyze-portfolio --snapshot-only`: closes in, snapshot out, no models."""

from __future__ import annotations

from datetime import date

import polars as pl

from stock_analyzer.cli import portfolio
from stock_analyzer.config import Settings


def test_snapshot_only_values_holdings_at_the_close(monkeypatch):
    holdings = {"Brokerage": [{"ticker": "AAPL", "units": 10, "price": 150.0}]}
    monkeypatch.setattr(portfolio, "fetch_portfolio_holdings", lambda: holdings)
    closes = pl.DataFrame({"date": [date(2026, 9, 24), date(2026, 9, 25)], "Close": [199.0, 200.0]})
    monkeypatch.setattr(portfolio.yf_gateway, "daily_closes", lambda *a, **k: {"AAPL": closes})
    seen = {}
    monkeypatch.setattr(
        portfolio,
        "record_portfolio_snapshot",
        lambda settings, h, *, prices: seen.update(holdings=h, prices=prices),
    )
    monkeypatch.setattr(
        portfolio, "_build_agent", lambda *_: (_ for _ in ()).throw(AssertionError("no models"))
    )
    portfolio.snapshot_only(Settings(), today=date(2026, 9, 25))
    assert seen["holdings"] is holdings and seen["prices"]["AAPL"] == 200.0
