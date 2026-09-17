"""Catalyst grading: stored catalyst calls are scored against the stock's
move vs SPY around the event date, and summarized for the Ranker. Uses a
temp SQLite DB through the real repository path and fake price history."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from stock_analyzer.db.repository import insert_pick_catalysts, insert_run
from stock_analyzer.db.session import get_session
from stock_analyzer.discover.catalyst_grading import (
    CatalystReport,
    GradedCatalyst,
    format_catalyst_grading_block,
    grade_catalysts,
)

TODAY = date(2026, 9, 17)
EVENT = date(2026, 8, 20)


def _frame(pre: float, post: float, event: date = EVENT) -> pd.DataFrame:
    """Flat at `pre` through the day before `event`, then `post` after."""
    idx = pd.date_range(event - timedelta(days=30), event + timedelta(days=30), freq="D")
    closes = [pre if ts.date() < event else post for ts in idx]
    return pd.DataFrame({"Close": closes}, index=idx)


def _fetcher(frames: dict[str, pd.DataFrame]):
    calls: list[str] = []

    def fetch(ticker, start, end):
        calls.append(ticker)
        return frames.get(ticker)

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def _seed(db_path, ticker, catalysts, n_runs=1):
    with get_session(db_path) as session:
        for _ in range(n_runs):
            run_id = insert_run(
                session,
                universe_size=1,
                survivors=1,
                picks=1,
                opus_model="o",
                sonnet_model="s",
                cash_budget=None,
            )
            insert_pick_catalysts(session, run_id, ticker, catalysts)


def _cat(direction="positive", expected_date=EVENT.isoformat(), impact="high"):
    return {
        "event": "Q2 earnings",
        "expected_date": expected_date,
        "direction": direction,
        "impact": impact,
        "source": "news:N1",
    }


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "t.db")


def test_positive_call_graded_on_excess_move_vs_spy(db):
    _seed(db, "NVDA", [_cat("positive")])
    fetch = _fetcher({"SPY": _frame(100, 102), "NVDA": _frame(100, 110)})
    report = grade_catalysts(db, today=TODAY, fetch=fetch)
    (g,) = report.graded
    assert g.excess_move_pct == pytest.approx(8.0)
    assert g.hit


def test_negative_call_misses_when_stock_rises(db):
    _seed(db, "NVDA", [_cat("negative")])
    fetch = _fetcher({"SPY": _frame(100, 100), "NVDA": _frame(100, 105)})
    (g,) = grade_catalysts(db, today=TODAY, fetch=fetch).graded
    assert not g.hit


def test_uncertain_call_hits_only_on_a_real_move(db):
    _seed(db, "AAA", [_cat("uncertain")])
    _seed(db, "BBB", [_cat("uncertain")])
    fetch = _fetcher({"SPY": _frame(100, 100), "AAA": _frame(100, 94), "BBB": _frame(100, 101)})
    graded = {g.ticker: g for g in grade_catalysts(db, today=TODAY, fetch=fetch).graded}
    assert graded["AAA"].hit
    assert not graded["BBB"].hit


def test_undated_future_and_too_recent_catalysts_are_not_graded(db):
    _seed(db, "NVDA", [_cat(expected_date=None)])
    _seed(db, "AMD", [_cat(expected_date=(TODAY + timedelta(days=10)).isoformat())])
    _seed(db, "ARM", [_cat(expected_date=(TODAY - timedelta(days=2)).isoformat())])
    fetch = _fetcher({})
    assert grade_catalysts(db, today=TODAY, fetch=fetch).graded == []
    assert fetch.calls == []  # nothing due -> no price fetches at all


def test_same_event_named_on_several_runs_is_graded_once(db):
    _seed(db, "NVDA", [_cat("positive")], n_runs=3)
    fetch = _fetcher({"SPY": _frame(100, 100), "NVDA": _frame(100, 110)})
    report = grade_catalysts(db, today=TODAY, fetch=fetch)
    assert len(report.graded) == 1
    assert fetch.calls.count("NVDA") == 1


def test_missing_price_history_skips_that_ticker(db):
    _seed(db, "GONE", [_cat()])
    fetch = _fetcher({"SPY": _frame(100, 100)})
    assert grade_catalysts(db, today=TODAY, fetch=fetch).graded == []


def _g(direction, move, impact="high"):
    return GradedCatalyst("X", "2026-08-01", direction, impact, move)


def test_block_flags_too_few_samples():
    block = format_catalyst_grading_block(CatalystReport([_g("positive", 5)]))
    assert "too few" in block


def test_block_warns_when_direction_calls_are_a_coin_flip():
    report = CatalystReport(
        [
            _g("positive", 5),
            _g("positive", -5),
            _g("negative", 5),
            _g("negative", -5),
            _g("positive", -1),
        ]
    )
    block = format_catalyst_grading_block(report)
    assert "40% right on 5" in block
    assert "NOT beaten a coin flip" in block


def test_block_flags_impact_labels_that_dont_predict_size():
    report = CatalystReport(
        [
            _g("positive", 1, "high"),
            _g("positive", 1, "high"),
            _g("positive", 8, "low"),
            _g("positive", 8, "low"),
            _g("uncertain", 9, "medium"),
        ]
    )
    block = format_catalyst_grading_block(report)
    assert "'high impact' labels have not moved stocks more" in block
    assert "'Uncertain' binary events that actually moved the stock >=3%: 100%" in block
