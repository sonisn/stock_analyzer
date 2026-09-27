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
    alerts = []
    monkeypatch.setattr(portfolio, "holding_alerts", lambda *a, **k: alerts.append(a))
    portfolio.snapshot_only(Settings(), today=date(2026, 9, 25))
    assert seen["holdings"] is holdings and seen["prices"]["AAPL"] == 200.0
    assert alerts and alerts[0][1] == ["AAPL"]


def test_a_failing_alert_check_never_costs_the_snapshot(monkeypatch):
    holdings = {"Brokerage": [{"ticker": "AAPL", "units": 1, "price": 1.0}]}
    monkeypatch.setattr(portfolio, "fetch_portfolio_holdings", lambda: holdings)
    monkeypatch.setattr(portfolio.yf_gateway, "daily_closes", lambda *a, **k: {})
    recorded = []
    monkeypatch.setattr(
        portfolio, "record_portfolio_snapshot", lambda *a, **k: recorded.append(True)
    )

    def boom(*a, **k):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(portfolio, "holding_alerts", boom)
    portfolio.snapshot_only(Settings(), today=date(2026, 9, 25))
    assert recorded == [True]
