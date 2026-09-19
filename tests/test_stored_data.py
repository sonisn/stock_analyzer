"""Stored brokerage history, the per-stock reference cache, and the
database size guard."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlmodel import select

from stock_analyzer.data import activity_ledger, reference, transactions
from stock_analyzer.db.retention import (
    RetentionPolicy,
    compact_if_worth_it,
    database_size,
    prune_database,
    size_line,
)
from stock_analyzer.db.session import get_session
from stock_analyzer.db.tables import BrokerageActivity, TickerReference

TODAY = date(2026, 9, 18)


def _act(i, kind, day, ticker="GOOGL", **kw):
    return {
        "id": f"a{i}",
        "type": kind,
        "trade_date": f"{day}T04:00:00Z",
        "symbol": {"symbol": ticker, "description": "long nested payload" * 20},
        **kw,
    }


@pytest.fixture
def db(tmp_path: Path) -> str:
    activity_ledger._synced.clear()
    return str(tmp_path / "s.db")


def test_sync_fetches_only_new_activity_per_account(db, monkeypatch):
    calls = []
    feeds = {
        1: {
            "IRA": [
                _act(1, "BUY", "2016-03-01", units=2, price=100),
                _act(2, "DIVIDEND", "2026-09-01", amount=5),
            ]
        },
        2: {
            "IRA": [
                _act(2, "DIVIDEND", "2026-09-01", amount=5),  # overlap: skipped
                _act(3, "REI", "2026-09-01", units=0.01, price=500, amount=-5),
            ],
            "New Acct": [_act(4, "CONTRIBUTION", "2020-01-02", ticker=None, amount=1000)],
        },
    }

    def live(start, end, *, since=None):
        calls.append((start, dict(since or {})))
        return feeds[len(calls)]

    monkeypatch.setattr(transactions, "_live_activities_by_account", live)
    assert activity_ledger.sync_activities(db, today=TODAY) == 2
    assert activity_ledger.sync_activities(db, today=TODAY) == 2
    first_start, first_since = calls[0]
    assert first_start == TODAY - timedelta(days=10 * 365) and first_since == {}
    # Second sync: the known account resumes 10 days before its latest
    # activity; the new one gets the full horizon.
    assert calls[1][1] == {"IRA": date(2026, 8, 22)}
    with get_session(db) as s:
        sizes = [len(r.data) for r in s.exec(select(BrokerageActivity)).all()]
    assert len(sizes) == 4 and max(sizes) < 400  # compact: nested payloads dropped


def test_readers_use_full_stored_history(db, monkeypatch):
    monkeypatch.setattr(
        transactions,
        "_live_activities_by_account",
        lambda start, end, since=None: {
            "IRA": [
                _act(1, "BUY", "2016-03-01", units=2, price=100),  # older than 3 years
                _act(2, "DIVIDEND", "2026-09-01", amount=5),
                _act(3, "REI", "2026-09-01", units=0.01, price=500, amount=-5),
                _act(4, "CONTRIBUTION", "2026-08-01", ticker=None, amount=1000),
            ]
        },
    )
    lots = transactions.fetch_transaction_history(db_path=db)["GOOGL"].lots
    assert sorted(lot.date for lot in lots) == ["2016-03-01", "2026-09-01"]
    assert {lot.account for lot in lots} == {"IRA"}
    cash = transactions.fetch_cash_activity(days_back=400, db_path=db)
    assert [f["amount"] for f in cash["flows"]] == [1000.0]
    assert [(d["ticker"], d["reinvested"]) for d in cash["dividends"]] == [("GOOGL", True)]


def test_stored_history_serves_when_brokerage_is_down(db, monkeypatch):
    activity_ledger.store_activities(db, {"IRA": [_act(1, "BUY", "2024-01-02", units=1, price=10)]})

    def boom(*a, **k):
        raise RuntimeError("SnapTrade down")

    monkeypatch.setattr(transactions, "_live_activities_by_account", boom)
    assert list(activity_ledger.ledger_activities(db)) == ["IRA"]


def test_reference_profiles_cached_and_refreshed(db):
    calls = []

    def fetch(ts):
        calls.append(list(ts))
        return {t: {"name": t.lower(), "sector": "Tech", "industry": "Chips"} for t in ts}

    assert reference.profiles(["A", "B"], db, fetch=fetch, today=TODAY)["A"]["sector"] == "Tech"
    reference.profiles(["A", "B"], db, fetch=fetch, today=TODAY + timedelta(days=10))
    reference.profiles(["A", "C"], db, fetch=fetch, today=TODAY + timedelta(days=40))
    assert calls == [["A", "B"], ["A", "C"]]  # day 10 all cached; day 40 A is stale


def test_reference_earnings_refetched_once_the_date_passes(db):
    calls = []
    dates = {"A": date(2026, 9, 20)}

    def fetch(t):
        calls.append(t)
        return dates.get(t)

    got = reference.next_earnings_dates(["A", "N"], db, fetch=fetch, today=TODAY)
    assert got == {"A": date(2026, 9, 20), "N": None}
    reference.next_earnings_dates(["A", "N"], db, fetch=fetch, today=TODAY + timedelta(days=1))
    assert calls == ["A", "N"]  # cached
    dates["A"] = date(2026, 12, 1)
    got = reference.next_earnings_dates(["A"], db, fetch=fetch, today=date(2026, 9, 21))
    assert got["A"] == date(2026, 12, 1)  # reported on the 20th: fetched again


def test_stale_reference_rows_pruned_and_size_guard(db):
    with get_session(db) as s:
        s.add(TickerReference(ticker="OLD", profile_updated="2024-01-01"))
        s.add(TickerReference(ticker="NEW", profile_updated=TODAY.isoformat()))
    out = prune_database(db, RetentionPolicy(), today=TODAY)
    assert out["stale_reference_rows"] == 1
    size = database_size(db, warn_mb=0.000001)
    assert size["over"] and "OVER the" in size_line(size)
    assert "Database:" in size_line(database_size(db, warn_mb=50))


def test_vacuum_only_when_enough_is_free(tmp_path: Path):
    path = tmp_path / "v.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (x TEXT)")
    con.executemany("INSERT INTO t VALUES (?)", [("x" * 1000,)] * 2000)
    con.commit()
    assert compact_if_worth_it(str(path), min_free_pct=20) == 0  # nothing free
    con.execute("DELETE FROM t")
    con.commit()
    con.close()
    assert compact_if_worth_it(str(path), min_free_pct=20) > 1_000_000
