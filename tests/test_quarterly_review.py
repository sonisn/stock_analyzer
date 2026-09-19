"""Suggestion ledger + quarterly review (reporting/quarterly.py)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from stock_analyzer.db.repository import insert_pick, insert_run, record_suggestions
from stock_analyzer.db.session import get_session
from stock_analyzer.reporting.health import build_portfolio_health, suggestion_rows
from stock_analyzer.reporting.quarterly import (
    collect_suggestions,
    first_trading_day,
    grade_suggestions,
    headline,
    is_first_trading_day_of_quarter,
    previous_quarter,
    render_quarterly_html,
    summarize,
)

TODAY = date(2026, 10, 1)


def test_first_trading_day_of_each_quarter():
    assert [first_trading_day(2026, q) for q in (1, 2, 3, 4)] == [
        date(2026, 1, 2),  # New Year's Day
        date(2026, 4, 1),
        date(2026, 7, 1),
        date(2026, 10, 1),
    ]
    assert first_trading_day(2023, 1) == date(2023, 1, 3)  # Jan 1 Sunday → observed Monday
    assert first_trading_day(2022, 1) == date(2022, 1, 3)  # Saturday: not observed; weekend
    assert first_trading_day(1994, 2) == date(1994, 4, 4)  # Good Friday on April 1
    assert first_trading_day(2027, 1) == date(2027, 1, 4)
    assert is_first_trading_day_of_quarter(date(2026, 10, 1))
    assert not is_first_trading_day_of_quarter(date(2026, 10, 2))
    assert not is_first_trading_day_of_quarter(date(2026, 1, 1))


def test_previous_quarter():
    assert previous_quarter(date(2026, 10, 1)) == ("Q3 2026", date(2026, 7, 1), date(2026, 9, 30))
    assert previous_quarter(date(2027, 1, 4)) == ("Q4 2026", date(2026, 10, 1), date(2026, 12, 31))
    assert previous_quarter(date(2026, 4, 1)) == ("Q1 2026", date(2026, 1, 1), date(2026, 3, 31))


def _prices(path: dict[str, tuple[float, float]]):
    """fetch() stub: each ticker moves linearly from a to b over Q3."""
    idx = pd.bdate_range(date(2026, 6, 1), TODAY)

    def fetch(ticker, start, end):
        if ticker not in path:
            return None
        a, b = path[ticker]
        n = len(idx) - 1
        return pd.DataFrame({"Close": [a + (b - a) * i / n for i in range(len(idx))]}, index=idx)

    return fetch


def _seed(db: str) -> None:
    rows = [
        # A repeated daily line is graded once, from its first day.
        dict(
            suggested_on="2026-07-01",
            source="daily",
            action="SELL",
            ticker="BAD",
            detail="thesis broken",
            price=100.0,
            units_held=10.0,
            reinvest_into="NEW",
        ),
        dict(
            suggested_on="2026-07-02",
            source="daily",
            action="SELL",
            ticker="BAD",
            detail="thesis broken",
            price=99.0,
            units_held=10.0,
            reinvest_into="NEW",
        ),
        dict(
            suggested_on="2026-07-01",
            source="daily",
            action="REVIEW",
            ticker="DIP",
            detail="re-check",
            price=80.0,
            units_held=5.0,
        ),
        dict(
            suggested_on="2026-08-03",
            source="rebalance",
            action="ADD",
            ticker="WIN",
            detail="$1,000",
            price=50.0,
            units_held=10.0,
            run_id=7,
        ),
        dict(
            suggested_on="2026-08-03",
            source="rebalance",
            action="SELL_PUT",
            ticker="PUT",
            detail="1 contract $90.00P 2026-09-18",
            units_held=0.0,
            run_id=7,
        ),
        dict(
            suggested_on="2026-06-30",
            source="daily",
            action="SELL",
            ticker="OLD",
            detail="before the quarter",
        ),
    ]
    with get_session(db) as s:
        record_suggestions(s, rows)
        run_id = insert_run(
            s,
            universe_size=1,
            survivors=1,
            picks=1,
            opus_model="o",
            sonnet_model="s",
            cash_budget=None,
        )
        s.exec(
            text("UPDATE runs SET run_at='2026-08-10T10:00:00' WHERE id=:i"),
            params={"i": run_id},
        )
        insert_pick(s, run_id, rank=1, ticker="PICK", entry_price=20.0)
        s.commit()


def test_collect_grade_and_render(tmp_path: Path):
    db = str(tmp_path / "q.db")
    _seed(db)
    label, start, end = previous_quarter(TODAY)
    items = collect_suggestions(db, start, end)
    assert [(i["source"], i["action"], i["ticker"]) for i in items] == [
        ("daily", "SELL", "BAD"),
        ("daily", "REVIEW", "DIP"),
        ("rebalance", "SELL_PUT", "PUT"),
        ("rebalance", "ADD", "WIN"),
        ("discover", "BUY", "PICK"),
    ]
    fetch = _prices(
        {
            "SPY": (100, 105),
            "BAD": (100, 80),  # fell after the sell call
            "NEW": (100, 110),  # the suggested replacement rose
            "DIP": (100, 95),
            "WIN": (100, 130),
            "PUT": (100, 101),
            "PICK": (100, 102),
        }
    )
    graded = {
        g["ticker"]: g
        for g in grade_suggestions(
            items, today=TODAY, units_now={"BAD": 0.0, "DIP": 5.0, "WIN": 10.0}, fetch=fetch
        )
    }
    bad = graded["BAD"]
    assert bad["verdict"] == "good call" and bad["acted"] == "yes"
    assert round(bad["edge_pct"]) == round(bad["reinvest_pct"] - bad["return_pct"])
    assert bad["edge_pct"] > 20
    assert graded["DIP"]["verdict"] == "missed" and graded["DIP"]["acted"] == "—"
    assert graded["WIN"]["verdict"] == "good call" and graded["WIN"]["acted"] == "no"
    assert graded["PUT"]["verdict"] == "—" and graded["PUT"]["edge_pct"] is None
    assert graded["PICK"]["acted"] == "no"

    summary = summarize(list(graded.values()))
    lines = headline(summary)
    assert lines[0].startswith("Sell / trim advice: 1 suggestion(s), 1 of 1 worked out")
    assert "you acted on 1 of 1" in lines[0]
    body = render_quarterly_html(
        label=label,
        start=start,
        end=end,
        graded=list(graded.values()),
        summary=summary,
        health_html="<h2>Where the portfolio stands today</h2>",
    )
    assert "Quarterly review — Q3 2026" in body
    assert "NEW +" in body and "GOOD CALL" in body and "Where the portfolio stands today" in body


def test_empty_quarter_says_so():
    body = render_quarterly_html(
        label="Q3 2026",
        start=date(2026, 7, 1),
        end=date(2026, 9, 30),
        graded=[],
        summary=summarize([]),
    )
    assert "No suggestions were recorded last quarter" in body


def test_daily_decisions_become_ledger_rows():
    h = build_portfolio_health(
        {"B": [{"ticker": "DOWN", "units": 10, "average_purchase_price": 100.0, "price": 70.0}]},
        earnings=lambda t: {
            "DOWN": {"ticker": "DOWN", "earnings_date": "2026-09-20", "days_until": 1}
        },
        reinvest=lambda held, over, n: [
            {"ticker": "NEW", "rank": 1, "pick_date": "2026-09-17", "sector": "Tech"}
        ],
    )
    rows = suggestion_rows(h, today="2026-09-18")
    # Earnings is information, not advice — only the drawdown re-check is kept.
    assert [(r["action"], r["ticker"], r["reinvest_into"]) for r in rows] == [
        ("REVIEW", "DOWN", "NEW")
    ]
    assert rows[0]["units_held"] == 10 and rows[0]["price"] == 70.0
