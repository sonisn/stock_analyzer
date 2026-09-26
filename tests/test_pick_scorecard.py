"""The daily email's six-month pick scorecard."""

from __future__ import annotations

from datetime import date

from sqlalchemy import text

from stock_analyzer.db.repository import insert_candidate, insert_pick, insert_run
from stock_analyzer.db.session import get_session
from stock_analyzer.db.tables import CandidateOutcome
from stock_analyzer.discover.pick_scorecard import pick_scorecard
from stock_analyzer.model.labels import label_candidates
from stock_analyzer.reporting.health import build_portfolio_health, render_health_html
from tests.test_model_training import _panel


def _run(session, run_at: str, picks: list[str], others: tuple[str, ...] = ()) -> int:
    run_id = insert_run(
        session,
        universe_size=1,
        survivors=1,
        picks=len(picks),
        opus_model="o",
        sonnet_model="s",
        cash_budget=None,
    )
    session.exec(
        text("UPDATE runs SET run_at = :r WHERE id = :i"), params={"r": run_at, "i": run_id}
    )
    for ticker in [*picks, *others]:
        insert_candidate(
            session,
            run_id,
            ticker,
            passed_filter=True,
            fail_reasons=[],
            score=None,
            score_components=None,
            score_breakdown=None,
            sources=[],
            conviction=0,
            sector=None,
            price=None,
        )
    for rank, ticker in enumerate(picks, 1):
        insert_pick(session, run_id, rank=rank, ticker=ticker)
    return run_id


def _outcome(session, run_id: int, ticker: str, ret: float, spy: float) -> None:
    session.add(
        CandidateOutcome(
            run_id=run_id,
            ticker=ticker,
            horizon_days=126,
            entry_date="x",
            exit_date="y",
            return_pct=ret,
            spy_return_pct=spy,
            excess_pct=ret - spy,
        )
    )


def test_cohorts_by_month_one_decision_per_ticker(tmp_path):
    db = str(tmp_path / "s.db")
    with get_session(db) as session:
        jan1 = _run(session, "2026-01-12T10:00:00", ["AAA", "BBB"])
        jan2 = _run(session, "2026-01-13T10:00:00", ["AAA"])  # same decision again
        feb = _run(session, "2026-02-10T10:00:00", ["CCC", "GONE"])
        _run(session, "2026-09-01T10:00:00", ["NEW"])
        _outcome(session, jan1, "AAA", 20.0, 5.0)
        _outcome(session, jan1, "BBB", -5.0, 5.0)
        _outcome(session, jan2, "AAA", 99.0, 5.0)
        _outcome(session, feb, "CCC", 8.0, 4.0)

    sc = pick_scorecard(db, today=date(2026, 9, 26))

    jan, feb_row = sc["cohorts"]
    assert (jan["cohort"], jan["picks"]) == ("Jan 2026", 2)
    assert jan["excess_pct"] == 2.5 and jan["beat_spy"] == 0.5
    assert (feb_row["cohort"], feb_row["picks"], feb_row["excess_pct"]) == ("Feb 2026", 1, 4.0)
    assert sc["overall"]["picks"] == 3
    # Window closed with no price: counted, not dropped or left maturing.
    assert sc["unmeasured"] == ["GONE"]
    assert sc["maturing"] == 1 and date(2027, 2, 20) < sc["next_due"] < date(2027, 3, 10)


def test_labels_only_the_picks_including_six_months(tmp_path):
    db = str(tmp_path / "l.db")
    p = _panel(3, 300, signal=0)
    with get_session(db) as session:
        _run(session, p.close.index[100].isoformat(), ["T0"], others=("T1",))

    assert label_candidates(db, fetch_panel=lambda t: p, only_picks=True) == 3
    with get_session(db) as session:
        rows = session.exec(
            text("SELECT ticker, horizon_days FROM candidate_outcomes ORDER BY horizon_days")
        ).all()
    assert [tuple(r) for r in rows] == [("T0", 21), ("T0", 63), ("T0", 126)]


def test_scorecard_renders_in_the_health_block():
    sc = {
        "horizon": 126,
        "cohorts": [
            {
                "cohort": "May 2026",
                "picks": 22,
                "return_pct": 1.0,
                "spy_pct": 9.0,
                "excess_pct": -8.0,
                "beat_spy": 0.2,
            }
        ],
        "overall": None,
        "maturing": 18,
        "next_due": date(2027, 3, 22),
        "unmeasured": [],
    }
    health = build_portfolio_health({}, pick_scorecard=lambda: sc)
    body = render_health_html(health)
    assert "Scorecard: six months after each idea" in body and "Discover picks" in body
    assert "May 2026" in body and "-8.0%" in body
    assert "18 picks still inside the six months; next results around Mar 22" in body

    quiet = build_portfolio_health(
        {},
        pick_scorecard=lambda: {**sc, "cohorts": [], "maturing": 0, "next_due": None},
    )
    assert "Scorecard" not in render_health_html(quiet)
