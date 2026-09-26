"""The nightly earnings-standout check and its place in the daily email."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import text

from stock_analyzer.db.session import get_session
from stock_analyzer.discover import earnings_standouts as es
from stock_analyzer.reporting.health import build_portfolio_health, render_health_html

REPORT = date(2026, 10, 20)  # a Tuesday


def _row(ticker, eps=1.2, eps_est=1.0, rev=1.05e9, rev_est=1.0e9, day=REPORT, hour="amc"):
    return {
        "ticker": ticker,
        "date": day.isoformat(),
        "hour": hour,
        "eps_estimate": eps_est,
        "eps_actual": eps,
        "revenue_estimate": rev_est,
        "revenue_actual": rev,
    }


def _closes(moves: dict[str, float]):
    """Bars Oct 1 - Nov 30; each ticker jumps by `moves[t]` (fraction) on
    the first session after the report, SPY by 1%."""
    idx = pd.bdate_range("2026-10-01", "2026-11-30")
    after = idx > pd.Timestamp(REPORT)

    def series(jump: float) -> pd.Series:
        return pd.Series([100.0 * (1 + jump) if a else 100.0 for a in after], index=idx)

    def fetch(symbols, start):
        return {s: series(0.01 if s == "SPY" else moves.get(s, 0.0)) for s in symbols}

    return fetch


def test_clear_beat_needs_eps_revenue_and_size():
    assert es.is_clear_beat(_row("A"))
    assert not es.is_clear_beat(_row("A", eps=1.02))  # 2% EPS beat
    assert not es.is_clear_beat(_row("A", rev=1.0e9))  # revenue in line
    assert not es.is_clear_beat(_row("A", rev=105e6, rev_est=100e6))  # too small
    assert es.is_clear_beat(_row("A", eps=-0.10, eps_est=-0.20))  # smaller loss
    assert not es.is_clear_beat(_row("A", eps=None))


def test_only_beats_and_past_picks_are_stored_once(tmp_path):
    db = str(tmp_path / "e.db")
    rows = [
        _row("BEAT"),
        _row("MEH", eps=1.0),
        _row("PICK", eps=0.5),  # a past pick that missed: kept for its history
        _row("OTC"),  # beat, but not an SEC-listed ticker
        {**_row("LATER"), "eps_actual": None},  # hasn't reported yet
    ]
    kwargs = dict(picks={"PICK"}, listed={"BEAT", "MEH", "PICK", "LATER"})
    assert es.record_reports(db, rows, **kwargs) == 2
    assert es.record_reports(db, rows, **kwargs) == 0
    with get_session(db) as session:
        stored = {t for (t,) in session.exec(text("SELECT ticker FROM earnings_events")).all()}
    assert stored == {"BEAT", "PICK"}


def test_reaction_spans_the_report_and_waits_for_a_final_close():
    fetch = _closes({"X": 0.08})
    bars = fetch(["X", "SPY"], None)
    moved = es.reaction(bars["X"], bars["SPY"], REPORT, last_final=date(2026, 10, 21))
    assert moved == pytest.approx(7.0)
    assert es.reaction(bars["X"], bars["SPY"], REPORT, last_final=REPORT) is None


def test_a_night_by_night_run_confirms_only_the_real_standout(tmp_path):
    db = str(tmp_path / "w.db")
    calendar = lambda start, end: [  # noqa: E731
        _row("GOOD"),
        _row("FLAT"),  # beat, market shrugged
        _row("NOREV"),  # beat and rewarded, analysts didn't raise
        _row("FLUKE"),  # all of that, but missed the quarter before
        _row("SHRINK"),  # all of that, but revenue down on a year ago
    ]
    revisions = {"GOOD": 0.06, "NOREV": 0.01, "FLUKE": 0.05, "SHRINK": 0.05}
    records = {
        "GOOD": {"prior_beats": 1, "prior_quarters": 1, "revenue_yoy_pct": 12.0},
        "FLUKE": {"prior_beats": 0, "prior_quarters": 1, "revenue_yoy_pct": 5.0},
        "SHRINK": {"prior_beats": 1, "prior_quarters": 1, "revenue_yoy_pct": -3.0},
    }
    asked: list[str] = []

    def track_record(ticker, report_day, revenue):
        assert report_day == REPORT and revenue == 1.05e9
        asked.append(ticker)
        return records[ticker]

    common = dict(
        calendar=calendar,
        closes=_closes({"GOOD": 0.09, "FLAT": 0.02, "NOREV": 0.10, "FLUKE": 0.1, "SHRINK": 0.1}),
        estimate_change=revisions.get,
        track_record=track_record,
        picks=set(),
        listed={"GOOD", "FLAT", "NOREV", "FLUKE", "SHRINK"},
    )
    night1 = es.watch(db, today=REPORT, last_final=REPORT, **common)
    assert night1["recorded"] == 5 and night1["reactions"] == 0

    night2 = es.watch(db, today=date(2026, 10, 21), last_final=date(2026, 10, 21), **common)
    assert night2["reactions"] == 5 and night2["standouts"] == []

    week = es.watch(db, today=date(2026, 10, 27), last_final=date(2026, 10, 27), **common)
    assert week["standouts"] == ["GOOD"]
    # The history is only fetched for names that passed every other gate.
    assert sorted(asked) == ["FLUKE", "GOOD", "SHRINK"]

    with get_session(db) as session:
        status = dict(session.exec(text("SELECT ticker, status FROM earnings_events")).all())
    assert status == {
        "GOOD": "standout",
        "FLAT": "no",
        "NOREV": "no",
        "FLUKE": "no",
        "SHRINK": "no",
    }

    (s,) = es.recent_standouts(db, days=5, today=date(2026, 10, 28))
    assert s["ticker"] == "GOOD"
    assert round(s["eps_surprise_pct"]) == 20 and round(s["revenue_surprise_pct"]) == 5
    assert round(s["reaction_pct"]) == 8 and round(s["revision_pct"]) == 6
    assert (s["prior_beats"], s["prior_quarters"], s["revenue_yoy_pct"]) == (1, 1, 12.0)
    assert es.recent_standouts(db, days=5, today=date(2026, 11, 10)) == []


def test_standouts_render_with_details_and_view():
    standout = {
        "ticker": "GOOD",
        "report_date": "2026-10-20",
        "eps_surprise_pct": 20.0,
        "revenue_surprise_pct": 5.0,
        "reaction_pct": 8.0,
        "revision_pct": 6.0,
        "prior_beats": 1,
        "prior_quarters": 1,
        "revenue_yoy_pct": 38.0,
        "details": {"name": "Good Co", "price": "$50", "view": "Durable demand."},
    }
    body = render_health_html(build_portfolio_health({}, standouts=lambda: [standout]))
    assert "Earnings standouts" in body and "GOOD — Good Co" in body
    assert "+20.0%" in body and "<b>+6.0%</b>" in body
    assert "Beat the quarter before" in body and "<td>yes</td>" in body and "+38.0%" in body
    assert "Long-term view:</b> Durable demand." in body
    assert "Earnings standouts" not in render_health_html(build_portfolio_health({}))


def test_track_record_reads_the_year_before_the_report():
    from stock_analyzer.data.earnings_history import prior_beats, year_ago_revenue

    # Yahoo's shape: newest first, the report itself and a future date included.
    dates = pd.DataFrame(
        {
            "EPS Estimate": [6.02, 4.70, 4.14, 3.31, 3.73, 3.05, 2.72],
            "Reported EPS": [None, 5.68, 4.85, 4.73, 3.60, 3.58, 2.99],
        },
        index=pd.to_datetime(
            [
                "2027-01-07",
                "2026-09-24",
                "2026-06-25",
                "2026-03-31",
                "2026-01-08",
                "2025-09-25",
                "2025-06-24",
            ]
        ),
    )
    # Only the quarter before counts: June beat; January missed.
    assert prior_beats(dates, date(2026, 9, 24)) == (1, 1)
    assert prior_beats(dates, date(2026, 3, 31)) == (0, 1)
    assert prior_beats(None, date(2026, 9, 24)) == (0, 0)

    stmt = pd.DataFrame(
        [[1.96e10, 1.72e10, 1.74e10, 1.57e10, None]],
        index=["Total Revenue"],
        columns=pd.to_datetime(
            ["2026-05-31", "2026-02-28", "2025-11-30", "2025-08-31", "2025-05-31"]
        ),
    )
    assert year_ago_revenue(stmt, date(2026, 9, 24)) == 1.57e10
    assert year_ago_revenue(stmt, date(2028, 1, 1)) is None
    assert es.passes_track_record({"prior_beats": 1, "revenue_yoy_pct": None})
    assert not es.passes_track_record({"prior_beats": 0, "revenue_yoy_pct": 20.0})
    assert not es.passes_track_record({"prior_beats": 1, "revenue_yoy_pct": -1.0})


def test_a_shown_standout_is_recorded_and_graded_six_months_on(tmp_path):
    from stock_analyzer.db.repository import record_suggestions
    from stock_analyzer.discover.pick_scorecard import standout_scorecard
    from stock_analyzer.reporting.health import suggestion_rows

    held = {"Brokerage": [{"ticker": "HELD", "units": 1, "price": 10.0}]}
    shown = [
        {"ticker": "GOOD", "eps_surprise_pct": 20.0, "revision_pct": 6.0},
        {"ticker": "HELD", "eps_surprise_pct": 9.0, "revision_pct": 4.0},
    ]
    health = build_portfolio_health(held, prices={"HELD": 10.0}, standouts=lambda: shown)
    rows = suggestion_rows(health, today="2026-01-05")
    (row,) = [r for r in rows if r["action"] == "STANDOUT"]  # not the held one
    assert row["ticker"] == "GOOD" and "EPS +20.0%" in row["detail"]

    db = str(tmp_path / "g.db")
    with get_session(db) as session:
        record_suggestions(session, [row, {**row, "suggested_on": "2026-01-06"}])
        record_suggestions(session, [{**row, "ticker": "NEW", "suggested_on": "2026-09-01"}])

    idx = pd.bdate_range("2025-12-01", "2026-09-25")

    def closes(symbols, start):
        up = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)
        flat = pd.Series(100.0, index=idx)
        return {s: (flat if s == "SPY" else up) for s in symbols if s != "NEW"}

    card = standout_scorecard(db, closes, today=date(2026, 9, 26))
    (jan,) = card["cohorts"]
    assert (jan["cohort"], jan["picks"]) == ("Jan 2026", 1)  # two days, one decision
    entry = 100 + idx.get_loc(pd.Timestamp("2026-01-06"))  # first close after the email
    assert jan["spy_pct"] == 0 and jan["excess_pct"] == pytest.approx(126 / entry * 100)
    assert card["maturing"] == 1  # NEW, shown this month


def _actions(ticker):
    return [
        {  # after the report
            "graded_at": "2026-10-21 13:03:18",
            "firm": "Barclays",
            "action": "main",
            "to_grade": "Overweight",
            "from_grade": "Overweight",
            "target_action": "Raises",
            "target": 289.0,
            "prior_target": 195.0,
        },
        {
            "graded_at": "2026-10-21 09:00:00",
            "firm": "UBS",
            "action": "up",
            "to_grade": "Buy",
            "from_grade": "Neutral",
            "target_action": "Raises",
            "target": 260.0,
            "prior_target": 200.0,
        },
        {  # before the report: context, stored but not "since the report"
            "graded_at": "2026-09-01 17:43:37",
            "firm": "Deutsche Bank",
            "action": "init",
            "to_grade": "Buy",
            "from_grade": "",
            "target_action": "Announces",
            "target": 220.0,
            "prior_target": None,
        },
        {  # years old: not stored
            "graded_at": "2019-01-01 10:00:00",
            "firm": "Old Firm",
            "action": "down",
            "to_grade": "Sell",
            "from_grade": "Hold",
            "target_action": "Lowers",
            "target": 10.0,
            "prior_target": 20.0,
        },
    ]


def test_analyst_actions_are_stored_for_followed_reports_and_summarized(tmp_path):
    from stock_analyzer.data.analyst_actions import describe, summarize

    db = str(tmp_path / "a.db")
    es.record_reports(db, [_row("GOOD")], picks=set(), listed={"GOOD"})
    asked = []

    def fetch(t):
        asked.append(t)
        return _actions(t)

    assert es.track_analysts(db, fetch, today=date(2026, 10, 27)) == 3
    assert es.track_analysts(db, fetch, today=date(2026, 10, 28)) == 0  # nothing new
    assert asked == ["GOOD", "GOOD"]

    since = summarize(db, "GOOD", since=REPORT)
    assert (since["count"], since["raised"], since["upgrades"], since["initiated"]) == (2, 2, 1, 0)
    assert since["avg_target_change_pct"] == pytest.approx((289 / 195 + 260 / 200 - 2) / 2 * 100)
    assert describe(since["actions"][0]) == "Oct 21 Barclays: target $195 → $289 (Overweight)"
    assert describe(since["actions"][1]) == "Oct 21 UBS: upgrade, target $200 → $260 (Buy)"
    assert summarize(db, "GOOD", since=date(2026, 8, 1))["initiated"] == 1


def test_standout_email_lists_analyst_work_or_says_none():
    analysts = {
        "count": 2,
        "raised": 2,
        "lowered": 0,
        "upgrades": 1,
        "downgrades": 0,
        "initiated": 0,
        "avg_target_change_pct": 39.1,
        "actions": [
            {
                "day": "2026-10-21",
                "firm": "Barclays",
                "action": "main",
                "to_grade": "Overweight",
                "target_action": "Raises",
                "target": 289.0,
                "prior_target": 195.0,
            }
        ],
    }
    base = {"ticker": "GOOD", "report_date": "2026-10-20"}
    body = render_health_html(
        build_portfolio_health({}, standouts=lambda: [{**base, "analysts": analysts}])
    )
    assert "Analysts since report" in body and "2 raised, 1 upgrade, targets +39% avg" in body
    assert "Barclays: target $195 → $289 (Overweight)" in body
    quiet = render_health_html(build_portfolio_health({}, standouts=lambda: [base]))
    assert "<td>none</td>" in quiet and "analyst actions since the report" not in quiet


def test_fetch_normalizes_yahoo_rows(monkeypatch):
    from stock_analyzer.data import analyst_actions as aa

    df = pd.DataFrame(
        {
            "Firm": ["UBS", ""],
            "ToGrade": ["Buy", "x"],
            "FromGrade": ["Buy", "x"],
            "Action": ["main", "main"],
            "priceTargetAction": ["Raises", ""],
            "currentPriceTarget": [259.0, 0.0],
            "priorPriceTarget": [187.0, 0.0],
        },
        index=pd.to_datetime(["2026-08-05 21:09:59", "2026-08-01 00:00:00"]),
    )
    monkeypatch.setattr(aa.yf_gateway, "ticker_call", lambda t, what, fn: df)
    (row,) = aa.fetch_analyst_actions("ANET")  # the firm-less row is dropped
    assert row["graded_at"] == "2026-08-05 21:09:59" and row["target"] == 259.0
