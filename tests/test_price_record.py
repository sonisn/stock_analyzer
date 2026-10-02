"""A permanent record of what things were worth.

Every price this system fetched was used once and thrown away, so the
question grading needs — "what was it worth on the day we advised it?" —
had no answer, and re-deriving it later meant a network call against a
history that may since have been restated.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from stock_analyzer.data.price_record import (
    coverage,
    missing_today,
    record_prices,
    stored_history,
)
from stock_analyzer.db.session import get_session
from stock_analyzer.reporting.quarterly import VERDICT_MIN_AGE_DAYS, grade_suggestions

TODAY = date(2026, 9, 21)


def _db(tmp_path: Path) -> str:
    db = str(tmp_path / "p.db")
    with get_session(db) as s:
        s.commit()
    return db


def test_a_second_run_the_same_day_asks_the_network_nothing(tmp_path: Path):
    db = _db(tmp_path)
    calls: list[list[str]] = []

    def fetch(symbols):
        calls.append(sorted(symbols))
        return {s: 100.0 for s in symbols}

    record_prices(db, ["NVDA", "SPY"], today=TODAY, fetch=fetch)
    record_prices(db, ["NVDA", "SPY"], today=TODAY, fetch=fetch)
    assert calls == [["NVDA", "SPY"]], "the second call must be served from the record"
    assert missing_today(db, ["NVDA", "SPY"], today=TODAY) == []
    # A ticker never seen before is still fetched.
    record_prices(db, ["NVDA", "LLY"], today=TODAY, fetch=fetch)
    assert calls[-1] == ["LLY"]


def test_a_failed_fetch_leaves_a_gap_not_an_exception(tmp_path: Path):
    db = _db(tmp_path)

    def boom(symbols):
        raise RuntimeError("yfinance is having a day")

    assert record_prices(db, ["NVDA"], today=TODAY, fetch=boom) == {}
    assert coverage(db)["rows"] == 0


def test_the_record_serves_the_graders_in_the_shape_they_expect(tmp_path: Path):
    db = _db(tmp_path)
    for day, nvda, spy in (("2026-08-01", 100.0, 500.0), ("2026-09-21", 130.0, 550.0)):
        record_prices(
            db,
            ["NVDA", "SPY"],
            today=date.fromisoformat(day),
            fetch=lambda s, n=nvda, p=spy: {"NVDA": n, "SPY": p},
        )
    fetch = stored_history(db)
    frame = fetch("NVDA", date(2026, 8, 1), TODAY)
    assert list(frame["Close"]) == [100.0, 130.0]
    assert fetch("NOPE", date(2026, 8, 1), TODAY) is None


def test_a_sell_is_graded_against_what_replaced_it(tmp_path: Path):
    """The whole point: selling X to buy Y is good only if Y beat X."""
    db = _db(tmp_path)
    prices = {
        "2026-08-01": {"AVGO": 100.0, "LLY": 100.0, "SPY": 100.0},
        "2026-09-21": {"AVGO": 110.0, "LLY": 130.0, "SPY": 105.0},
    }
    for day, px in prices.items():
        record_prices(db, list(px), today=date.fromisoformat(day), fetch=lambda s, p=px: p)
    item = dict(
        suggested_on="2026-08-01",
        ticker="AVGO",
        action="SELL",
        source="daily",
        reinvest_into="LLY",
        units_held=100.0,
        price=100.0,
    )
    row = grade_suggestions([item], today=TODAY, units_now={"AVGO": 0.0}, fetch=stored_history(db))[
        0
    ]
    assert round(row["return_pct"]) == 10  # the stock you sold rose 10%
    assert round(row["reinvest_pct"]) == 30  # what you bought rose 30%
    assert round(row["edge_pct"]) == 20  # the switch gained 20 points
    assert row["verdict"] == "good call"
    assert row["acted"] == "yes"


def test_a_switch_into_something_worse_is_marked_missed(tmp_path: Path):
    db = _db(tmp_path)
    prices = {
        "2026-08-01": {"AVGO": 100.0, "LLY": 100.0, "SPY": 100.0},
        "2026-09-21": {"AVGO": 150.0, "LLY": 105.0, "SPY": 105.0},
    }
    for day, px in prices.items():
        record_prices(db, list(px), today=date.fromisoformat(day), fetch=lambda s, p=px: p)
    item = dict(
        suggested_on="2026-08-01",
        ticker="AVGO",
        action="SELL",
        source="daily",
        reinvest_into="LLY",
        units_held=100.0,
        price=100.0,
    )
    row = grade_suggestions(
        [item], today=TODAY, units_now={"AVGO": 100.0}, fetch=stored_history(db)
    )[0]
    assert round(row["edge_pct"]) == -45
    assert row["verdict"] == "missed"
    assert row["acted"] == "no", "units unchanged means the advice was not taken"


def test_advice_too_young_to_have_an_outcome_says_so(tmp_path: Path):
    """A one-day-old suggestion scored 'good call' on a 0.0% edge reads as
    a result and is only the absence of one."""
    db = _db(tmp_path)
    px = {"AVGO": 100.0, "SPY": 100.0}
    record_prices(db, list(px), today=TODAY, fetch=lambda s: px)
    item = dict(
        suggested_on=TODAY.isoformat(),
        ticker="AVGO",
        action="SELL",
        source="daily",
        reinvest_into=None,
        units_held=1.0,
        price=100.0,
    )
    row = grade_suggestions([item], today=TODAY, units_now={}, fetch=stored_history(db))[0]
    assert row["verdict"] == "too early"
    assert VERDICT_MIN_AGE_DAYS >= 21


def test_nothing_is_recorded_before_the_days_close_is_final(monkeypatch, tmp_path):
    """At 6 AM (or on a Sunday) the latest close is an earlier session's:
    stored under today it would block the real close after the bell."""
    from datetime import UTC, datetime

    from stock_analyzer.data import price_record, yf_gateway

    monkeypatch.setattr(
        yf_gateway, "download", lambda *a, **k: pytest.fail("no fetch before the close")
    )
    db = str(tmp_path / "p.db")
    monday = date(2026, 9, 28)
    six_am = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)  # 6:00 AM New York
    assert price_record.record_prices(db, ["NVDA"], today=monday, now=six_am) == {}
    sunday = datetime(2026, 9, 27, 18, 0, tzinfo=UTC)
    assert price_record.record_prices(db, ["NVDA"], today=date(2026, 9, 27), now=sunday) == {}


def test_the_record_covers_holdings_suggestions_reinvestments_and_spy(tmp_path: Path):
    from stock_analyzer.data.price_record import record_tickers
    from stock_analyzer.db.tables import Suggestion

    db = _db(tmp_path)
    with get_session(db) as s:
        s.add(
            Suggestion(
                suggested_on="2026-09-20",
                source="daily",
                action="SELL",
                ticker="TSLA",
                reinvest_into="AVGO",
            )
        )
        s.add(Suggestion(suggested_on="2026-09-20", source="daily", action="BUY", ticker="LLY"))
        s.commit()
    assert record_tickers(db, ["NVDA"]) == ["AVGO", "LLY", "NVDA", "SPY", "TSLA"]


def test_a_backfill_fills_missed_days_and_never_overwrites(tmp_path: Path, monkeypatch):
    import polars as pl

    from stock_analyzer.data import bar_store

    db = _db(tmp_path)
    record_prices(db, ["NVDA"], today=date(2026, 9, 28), fetch=lambda s: {"NVDA": 100.0})
    days = [date(2026, 9, 28), date(2026, 9, 29)]
    stored = pl.DataFrame({"date": days, "Close": [999.0, 101.0]})

    class Stored:
        frame = stored

    monkeypatch.setattr(bar_store, "load", lambda t: Stored() if t == "NVDA" else None)
    monkeypatch.setattr(
        "stock_analyzer.data.price_record.date",
        type("D", (date,), {"today": staticmethod(lambda: date(2026, 9, 30))}),
    )
    from stock_analyzer.data.price_record import backfill_from_panel

    assert backfill_from_panel(db, ["NVDA"]) == 1
    frame = stored_history(db)("NVDA", days[0], days[1])
    assert list(frame["Close"]) == [100.0, 101.0], "the recorded close stays; the gap fills"
