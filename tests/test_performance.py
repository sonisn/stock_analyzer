"""Portfolio vs SPY (time-weighted) and the brokerage cash-activity feed."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stock_analyzer.data import transactions
from stock_analyzer.db.repository import fetch_snapshots, record_snapshot
from stock_analyzer.db.session import get_session
from stock_analyzer.reporting.performance import (
    performance_vs_spy,
    render_performance_html,
    time_weighted_return,
)

D = date


def test_deposits_are_not_performance():
    # 100 -> 110 (+10%), then a 50 deposit and flat market: 110 -> 160.
    snaps = [(D(2026, 1, 2), 100.0), (D(2026, 1, 3), 110.0), (D(2026, 1, 4), 160.0)]
    flows = [(D(2026, 1, 4), 50.0)]
    assert time_weighted_return(snaps, flows) == pytest.approx(10.0)
    # A withdrawal the same way: 110 -> 60 after taking 50 out is flat.
    assert time_weighted_return(snaps[:2] + [(D(2026, 1, 4), 60.0)], [(D(2026, 1, 4), -50.0)]) == (
        pytest.approx(10.0)
    )
    assert time_weighted_return(snaps[:1], []) is None


def test_vs_spy_windows_and_render():
    idx = pd.bdate_range("2025-12-20", "2026-04-10")
    spy = pd.DataFrame({"Close": [100 + i * 0.1 for i in range(len(idx))]}, index=idx)
    snaps = [(D(2026, 1, 2), 100.0), (D(2026, 3, 31), 112.0), (D(2026, 4, 1), 113.0)]
    rows = performance_vs_spy(
        snaps,
        [],
        windows={
            "Last quarter": D(2026, 1, 1),
            "Since tracking began": D(2026, 1, 2),
            "Too new": D(2026, 4, 1),
        },
        fetch=lambda t, s, e: spy,
    )
    assert [r["label"] for r in rows] == ["Last quarter", "Since tracking began"]
    assert rows[0]["portfolio_pct"] == pytest.approx(13.0)
    assert rows[0]["diff_pts"] == pytest.approx(13.0 - rows[0]["spy_pct"])
    body = render_performance_html(rows, first_day=D(2026, 1, 2))
    assert "Your portfolio vs SPY" in body and "+13.0%" in body
    assert "Not enough history yet" in render_performance_html([], first_day=D(2026, 9, 19))


def test_snapshot_upsert(tmp_path: Path):
    db = str(tmp_path / "s.db")
    with get_session(db) as s:
        record_snapshot(s, day="2026-09-18", holdings_value=100.0, cash=5.0)
        record_snapshot(s, day="2026-09-18", holdings_value=110.0, cash=5.0)
        record_snapshot(s, day="2026-09-19", holdings_value=120.0, cash=0.0)
        s.commit()
        snaps = [(x.day, x.total) for x in fetch_snapshots(s)]
    assert snaps == [("2026-09-18", 115.0), ("2026-09-19", 120.0)]


def _client_with(activities):
    client = MagicMock()
    client.account_information.list_user_accounts.return_value = [{"id": "a", "name": "IRA"}]
    client.account_information.get_account_activities.return_value = activities
    return client


def test_cash_activity_flows_and_dividends(monkeypatch):
    acts = [
        {"type": "CONTRIBUTION", "amount": 4000, "trade_date": "2026-08-27T04:00:00Z"},
        {"type": "WITHDRAWAL", "amount": -4000, "trade_date": "2026-08-31T04:00:00Z"},
        {
            "type": "TRANSFER",
            "amount": 72997.57,
            "trade_date": "2026-04-27T04:00:00Z",
            "symbol": {"symbol": "NVDA"},
        },
        {
            "type": "DIVIDEND",
            "amount": 44.03,
            "trade_date": "2026-09-14T04:00:00Z",
            "symbol": {"symbol": "GOOGL"},
        },
        {
            "type": "REI",
            "amount": -44.03,
            "units": 0.13,
            "price": 339.4,
            "trade_date": "2026-09-14T04:00:00Z",
            "symbol": {"symbol": "GOOGL"},
        },
        {
            "type": "DIVIDEND",
            "amount": 3.69,
            "trade_date": "2026-08-20T00:00:00Z",
            "symbol": {"symbol": "POWL"},
        },
        {
            "type": "BUY",
            "amount": -500,
            "units": 5,
            "price": 100,
            "trade_date": "2026-08-01T00:00:00Z",
            "symbol": {"symbol": "X"},
        },
    ]
    monkeypatch.setattr(transactions, "_credentials", lambda: ("u", "s"))
    monkeypatch.setattr(transactions, "_client", lambda: _client_with(acts))
    out = transactions.fetch_cash_activity(days_back=400)
    assert [(f["type"], f["amount"]) for f in out["flows"]] == [
        ("CONTRIBUTION", 4000.0),
        ("WITHDRAWAL", -4000.0),
        ("TRANSFER", 72997.57),
    ]
    assert [(d["ticker"], d["amount"], d["reinvested"]) for d in out["dividends"]] == [
        ("GOOGL", 44.03, True),
        ("POWL", 3.69, False),
    ]


def test_dividend_reinvestments_are_lots(monkeypatch):
    acts = [
        {
            "type": "REI",
            "units": 0.13,
            "price": 339.4,
            "trade_date": "2026-09-14T04:00:00Z",
            "symbol": {"symbol": "GOOGL"},
            "account": {"id": "a"},
        },
        {
            "type": "BUY",
            "units": 2,
            "price": 300,
            "trade_date": "2025-01-10T04:00:00Z",
            "symbol": {"symbol": "GOOGL"},
            "account": {"id": "a"},
        },
    ]
    monkeypatch.setattr(transactions, "_credentials", lambda: ("u", "s"))
    monkeypatch.setattr(transactions, "_client", lambda: _client_with(acts))
    summary = transactions.fetch_transaction_history(years_back=3)["GOOGL"]
    assert sorted(lot.units for lot in summary.lots) == [0.13, 2.0]
    assert {lot.account for lot in summary.lots} == {"IRA"}
