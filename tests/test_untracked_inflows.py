"""Money that arrives without a deposit row must not read as return.

Two ways it does, both live in this portfolio: a 401(k) whose payroll
contributions show up only as purchases, and a Schwab reconnect that
revealed four accounts SnapTrade had never returned before.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from stock_analyzer.cli.portfolio import snapshot_account_values
from stock_analyzer.data import transactions
from stock_analyzer.db.repository import fetch_snapshots, record_snapshot, snapshot_accounts
from stock_analyzer.db.session import get_session
from stock_analyzer.reporting.performance import account_change_flows, time_weighted_return

D = date


def _buy(day: str, amount: float, symbol: str = "FGCCPS") -> dict:
    return {"type": "BUY", "amount": amount, "trade_date": day, "symbol": {"symbol": symbol}}


def _plan_ledger() -> dict[str, list[dict]]:
    """The real shape of the 401(k) feed: two BUYs a payday, $730 in all,
    and no CONTRIBUTION row anywhere."""
    return {
        "Broadcom U.S. 401(k) Plan": [
            _buy("2026-08-31T00:00:00Z", -336.92),
            _buy("2026-08-31T00:00:00Z", -393.08),
            _buy("2026-09-14T00:00:00Z", -336.93),
            _buy("2026-09-14T00:00:00Z", -393.08),
        ]
    }


def test_plan_purchases_are_contributions(monkeypatch):
    monkeypatch.setattr(transactions, "activities_by_account", lambda **_: _plan_ledger())
    monkeypatch.setattr(transactions, "date", _FrozenDate)
    flows = transactions.implied_plan_flows(days_back=400)
    assert [(f["date"], f["amount"]) for f in flows] == [
        (D(2026, 8, 31), 730.0),
        (D(2026, 9, 14), 730.01),
    ]
    assert {f["account"] for f in flows} == {"Broadcom U.S. 401(k) Plan"}
    assert {f["type"] for f in flows} == {"PLAN_CONTRIBUTION"}


def test_accounts_that_report_their_own_flows_are_left_alone(monkeypatch):
    ledger = _plan_ledger()
    ledger["Robinhood Individual"] = [
        {"type": "CONTRIBUTION", "amount": 1000.0, "trade_date": "2026-09-11T00:00:00Z"},
        _buy("2026-09-11T00:00:00Z", -1000.0, "NVDA"),
    ]
    # An account with no trades at all has nothing to imply either.
    ledger["Equity Awards"] = [
        {"type": "DIVIDEND", "amount": 5.0, "trade_date": "2026-09-11T00:00:00Z"}
    ]
    monkeypatch.setattr(transactions, "activities_by_account", lambda **_: ledger)
    monkeypatch.setattr(transactions, "date", _FrozenDate)
    flows = transactions.implied_plan_flows(days_back=400)
    assert {f["account"] for f in flows} == {"Broadcom U.S. 401(k) Plan"}


def test_in_plan_rebalance_is_not_a_contribution(monkeypatch):
    ledger = {
        "Broadcom U.S. 401(k) Plan": [
            _buy("2026-09-14T00:00:00Z", -5000.0, "FUND_B"),
            {
                "type": "SELL",
                "amount": 5000.0,
                "trade_date": "2026-09-14T00:00:00Z",
                "symbol": {"symbol": "FUND_A"},
            },
        ]
    }
    monkeypatch.setattr(transactions, "activities_by_account", lambda **_: ledger)
    monkeypatch.setattr(transactions, "date", _FrozenDate)
    assert transactions.implied_plan_flows(days_back=400) == []


def test_an_account_holding_cash_settles_its_own_trades(monkeypatch):
    monkeypatch.setattr(transactions, "activities_by_account", lambda **_: _plan_ledger())
    monkeypatch.setattr(transactions, "date", _FrozenDate)
    cash = {"Broadcom U.S. 401(k) Plan": 20_000.0}
    assert transactions.implied_plan_flows(days_back=400, cash=cash) == []
    # Sweep residue is still "no cash".
    residue = {"Broadcom U.S. 401(k) Plan": 0.12}
    assert transactions.implied_plan_flows(days_back=400, cash=residue)


def test_a_connected_account_is_an_inflow_not_a_gain():
    before = {
        "Robinhood Individual": {"value": 100_000.0, "cash": 1_000.0},
        "Traditional IRA": {"value": 380_000.0, "cash": 19_878.0},
    }
    after = dict(before) | {"Broadcom U.S. 401(k) Plan": {"value": 4_495.65, "cash": 0.0}}
    assert account_change_flows([(D(2026, 9, 18), before), (D(2026, 9, 20), after)]) == [
        (D(2026, 9, 20), 4495.65)
    ]
    # And a broker that drops off the feed did not lose the money.
    assert account_change_flows([(D(2026, 9, 20), after), (D(2026, 9, 21), before)]) == [
        (D(2026, 9, 21), -4495.65)
    ]
    # Same accounts, more money: that is performance, not a flow.
    grown = {k: {"value": v["value"] * 1.1, "cash": v["cash"]} for k, v in before.items()}
    assert account_change_flows([(D(2026, 9, 18), before), (D(2026, 9, 20), grown)]) == []


def test_an_unknown_breakdown_never_invents_a_flow():
    known = {"Robinhood Individual": {"value": 100.0, "cash": 0.0}}
    assert account_change_flows([(D(2026, 9, 18), None), (D(2026, 9, 20), known)]) == []
    assert account_change_flows([(D(2026, 9, 18), known), (D(2026, 9, 20), None)]) == []
    assert account_change_flows([(D(2026, 9, 18), known)]) == []


def test_the_401k_does_not_beat_the_market():
    """A flat portfolio that gains a 401(k) and two paydays is flat."""
    snaps = [
        (D(2026, 9, 18), 500_000.0),
        (D(2026, 9, 20), 504_495.65),  # the plan joins the feed
        (D(2026, 9, 28), 505_225.65),  # one payday, nothing else
    ]
    breakdown = [
        (D(2026, 9, 18), {"IRA": {"value": 500_000.0, "cash": 0.0}}),
        (
            D(2026, 9, 20),
            {"IRA": {"value": 500_000.0, "cash": 0.0}, "401k": {"value": 4_495.65, "cash": 0.0}},
        ),
        (
            D(2026, 9, 28),
            {"IRA": {"value": 500_000.0, "cash": 0.0}, "401k": {"value": 5_225.65, "cash": 0.0}},
        ),
    ]
    flows = account_change_flows(breakdown) + [(D(2026, 9, 25), 730.0)]
    assert time_weighted_return(snaps, flows) == pytest.approx(0.0, abs=1e-9)
    # Without the correction the same two events read as a +1.05% quarter.
    assert time_weighted_return(snaps, []) == pytest.approx(1.045, abs=0.01)


def test_snapshot_keeps_its_account_breakdown(tmp_path: Path):
    db = str(tmp_path / "s.db")
    accounts = {"Traditional IRA": {"value": 100.0, "cash": 5.0}}
    with get_session(db) as s:
        record_snapshot(s, day="2026-09-20", holdings_value=100.0, cash=5.0, accounts=accounts)
        # A later write that could not read the breakdown must not erase it.
        record_snapshot(s, day="2026-09-20", holdings_value=101.0, cash=5.0)
        s.commit()
        rows = fetch_snapshots(s)
        assert snapshot_accounts(rows[0]) == accounts
        assert rows[0].total == 106.0


def test_empty_accounts_are_still_recorded():
    holdings = {
        "Traditional IRA": [{"ticker": "NVDA", "units": 2, "price": 100.0}],
        "Equity Awards": [],
    }
    values = snapshot_account_values(holdings, {"Traditional IRA": 50.0, "Individual ...652": 0.0})
    assert values == {
        "Traditional IRA": {"value": 200.0, "cash": 50.0},
        "Equity Awards": {"value": 0.0, "cash": 0.0},
        "Individual ...652": {"value": 0.0, "cash": 0.0},
    }


class _FrozenDate(date):
    """`date.today()` pinned, so the lookback window is stable."""

    @classmethod
    def today(cls) -> date:
        return D(2026, 9, 20)
