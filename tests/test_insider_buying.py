"""Insider-buying clusters: storage, detection, the daily email and the ledger."""

from __future__ import annotations

from datetime import date

from stock_analyzer.data import insider_buying as ib
from stock_analyzer.reporting.health import (
    build_portfolio_health,
    render_health_html,
    suggestion_rows,
)

TODAY = date(2026, 9, 28)


def _buy(ticker, name, filed, shares=1000.0, price=30.0, fid=None):
    return {
        "ticker": ticker,
        "filing_id": fid or f"{ticker}-{name}-{filed}",
        "name": name,
        "filed": filed,
        "traded": filed,
        "shares": shares,
        "price": price,
    }


def test_a_cluster_needs_two_different_insiders_inside_ninety_days(tmp_path):
    db = str(tmp_path / "i.db")
    buys = {
        "KMI": [_buy("KMI", "Smith", "2026-08-01"), _buy("KMI", "Jones", "2026-09-20")],
        "ONE": [  # the same insider twice is not a cluster
            _buy("ONE", "Lee", "2026-09-01"),
            _buy("ONE", "Lee", "2026-09-15", fid="x2"),
        ],
        "OLD": [_buy("OLD", "A", "2026-05-01"), _buy("OLD", "B", "2026-05-15")],  # too old
    }
    summary = ib.watch(db, today=TODAY, tickers=list(buys), fetch=lambda t, today: buys[t])
    assert summary["added"] == 6
    (kmi,) = summary["clusters"]
    assert (kmi["ticker"], kmi["buyers"], kmi["formed_on"]) == ("KMI", 2, "2026-09-20")
    assert kmi["value_usd"] == 60_000.0 and kmi["names"] == ["Smith", "Jones"]
    again = ib.watch(db, today=TODAY, tickers=list(buys), fetch=lambda t, today: buys[t])
    assert again["added"] == 0


def test_fetch_keeps_only_open_market_purchases(monkeypatch):
    rows = [
        {
            "name": "SMITH WILLIAM A",
            "change": 3000,
            "filingDate": "2026-02-03",
            "transactionDate": "2026-02-02",
            "transactionCode": "P",
            "transactionPrice": 29.75,
            "id": "0001033442-26-000001",
            "isDerivative": False,
        },
        {
            "name": "SELLER",
            "change": -500,
            "filingDate": "2026-02-03",
            "transactionCode": "S",
            "transactionPrice": 30.0,
            "id": "a",
        },
        {
            "name": "GRANTEE",
            "change": 900,
            "filingDate": "2026-02-03",
            "transactionCode": "A",
            "id": "b",
        },
    ]

    class Client:
        def stock_insider_transactions(self, *args):
            return {"data": rows}

    monkeypatch.setattr(ib.finnhub_data, "_client", lambda: Client())
    monkeypatch.setattr(ib.finnhub_data, "_safe_call", lambda label, t, fn, *a: fn(*a))
    (buy,) = ib.fetch_insider_buys("KMI", today=TODAY)
    assert buy["name"] == "Smith William A" and buy["shares"] == 3000 and buy["price"] == 29.75


def test_new_clusters_show_in_the_email_and_the_ledger():
    cluster = {
        "ticker": "KMI",
        "buyers": 3,
        "value_usd": 250_000.0,
        "names": ["Smith", "Jones", "Lee"],
        "formed_on": "2026-09-26",
    }
    held = {"Brokerage": [{"ticker": "AVGO", "units": 1, "price": 350.0}]}
    mine = {**cluster, "ticker": "AVGO"}
    health = build_portfolio_health(
        held, prices={"AVGO": 350.0}, insider_clusters=lambda: [cluster, mine]
    )
    body = render_health_html(health)
    assert "Insider buying" in body and "Smith, Jones, Lee" in body and "$250,000" in body
    assert "AVGO (held)" in body
    (row,) = [
        r for r in suggestion_rows(health, today="2026-09-28") if r["action"] == "INSIDER_BUYS"
    ]
    assert row["ticker"] == "KMI" and "3 insiders bought $250,000" in row["detail"]


def test_small_plan_and_entity_buys_do_not_make_a_cluster(tmp_path):
    db = str(tmp_path / "f.db")
    rows = [
        # Two small buyers: under the $10,000 floor each (SPG's reinvestment).
        _buy("SPG", "Cicco", "2026-09-01", shares=13, price=223.0),
        _buy("SPG", "Jones", "2026-09-01", shares=2, price=224.0),
        # Five staff at one price on one day: a plan, whatever the total.
        *[_buy("TSM", f"Staff {i}", "2026-09-09", shares=500, price=76.2) for i in range(5)],
        # Two real buyers: counted once each is over the floor.
        _buy("BSX", "Ludwig", "2026-08-01", shares=200, price=40.0),  # $8,000 ...
        _buy("BSX", "Ludwig", "2026-08-20", shares=100, price=41.0, fid="l2"),  # ... +$4,100
        _buy("BSX", "Mahoney", "2026-09-02", shares=1000, price=42.0),
    ]
    ib.record_buys(db, rows)
    (bsx,) = ib.clusters(db, today=TODAY)
    assert bsx["ticker"] == "BSX" and bsx["names"] == ["Ludwig", "Mahoney"]
    assert bsx["formed_on"] == "2026-09-02"


def test_entities_and_foreign_rows_are_not_insiders(monkeypatch):
    assert not ib.is_person("Horizon Kinetics Asset Management Llc")
    assert not ib.is_person("Smith Family Trust")
    assert ib.is_person("Khosrowshahi Dara") and ib.is_person("Fundaro Jane")
    rows = [
        {
            "name": "HOME EXCHANGE ROW",
            "change": 100,
            "filingDate": "2026-09-01",
            "transactionCode": "P",
            "transactionPrice": 10.0,
            "id": "t",
            "source": "twse",
        },
        {
            "name": "ICAHN CAPITAL LP",
            "change": 100,
            "filingDate": "2026-09-01",
            "transactionCode": "P",
            "transactionPrice": 10.0,
            "id": "i",
            "source": "sec",
        },
    ]

    class Client:
        def stock_insider_transactions(self, *args):
            return {"data": rows}

    monkeypatch.setattr(ib.finnhub_data, "_client", lambda: Client())
    monkeypatch.setattr(ib.finnhub_data, "_safe_call", lambda label, t, fn, *a: fn(*a))
    assert ib.fetch_insider_buys("X", today=TODAY) == []
