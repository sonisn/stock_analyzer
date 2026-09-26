"""The nightly point-in-time forecast snapshot."""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text

from stock_analyzer.data import forecast_snapshots as fs
from stock_analyzer.db.session import get_session
from stock_analyzer.db.tables import EarningsEvent, StockView
from tests.test_pick_scorecard import _run

TODAY = date(2026, 9, 28)


def _forecast(ticker):
    if ticker == "SPAXX":
        return None  # a money-market fund has no forecast
    return {"price": 100.0, "eps_next_year": 5.0, "target_mean": 120.0, "analysts": 20}


def test_tracks_holdings_picks_survivors_and_standouts(tmp_path):
    db = str(tmp_path / "t.db")
    with get_session(db) as session:
        session.add(StockView(ticker="AVGO", written_on="2026-09-20", view="v"))
        session.add(StockView(ticker="OLD", written_on="2026-08-01", view="v"))  # not held now
        _run(session, "2026-09-20T10:00:00", ["ANET"], others=("MU",))
        session.add(
            EarningsEvent(
                ticker="GOOD", report_date="2026-09-10", status="standout", decided_on="2026-09-18"
            )
        )
    assert fs.tracked_tickers(db, today=TODAY) == ["AVGO", "ANET", "MU", "GOOD"]


def test_stores_one_row_per_ticker_per_day(tmp_path):
    db = str(tmp_path / "s.db")
    tickers = ["AVGO", "ANET", "SPAXX"]
    first = fs.record_snapshots(db, today=TODAY, tickers=tickers, fetch=_forecast)
    assert first == {"stored": 2, "no_forecast": 1, "already": 0}
    again = fs.record_snapshots(db, today=TODAY, tickers=tickers, fetch=_forecast)
    assert again == {"stored": 0, "no_forecast": 1, "already": 2}
    with get_session(db) as session:
        rows = session.exec(
            text("SELECT ticker, eps_next_year, target_mean FROM forecast_snapshots ORDER BY 1")
        ).all()
    assert [tuple(r) for r in rows] == [("ANET", 5.0, 120.0), ("AVGO", 5.0, 120.0)]


def test_fetch_reads_yahoo_shapes(monkeypatch):
    trend = pd.DataFrame({"current": [4.9, 5.1, 22.8, 24.9]}, index=["0q", "+1q", "0y", "+1y"])
    revenue = pd.DataFrame({"avg": [3.3e9, 3.6e9, 12.7e9, 16.2e9]}, index=trend.index)
    info = {"currentPrice": 199.4, "targetMeanPrice": 241.0, "numberOfAnalystOpinions": 28}
    answers = {"info": info, "eps_trend": trend, "revenue_estimate": revenue}
    monkeypatch.setattr(
        fs.yf_gateway, "ticker_call", lambda t, what, fn, default=None: answers[what]
    )
    row = fs.fetch_forecast("ANET")
    assert row["eps_current_year"] == 22.8 and row["eps_next_year"] == 24.9
    assert row["revenue_next_year"] == 16.2e9 and row["analysts"] == 28
    assert row["target_mean"] == 241.0 and row["recommendation_mean"] is None

    answers.update(info={"currentPrice": 1.0}, eps_trend=None, revenue_estimate=None)
    assert fs.fetch_forecast("SPAXX") is None
