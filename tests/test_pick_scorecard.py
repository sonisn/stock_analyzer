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


def _run(
    session,
    run_at: str,
    picks: list[str],
    others: tuple[str, ...] = (),
    scores: dict[str, float] | None = None,
    convictions: dict[str, tuple[int, float]] | None = None,
) -> int:
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
            score=(scores or {}).get(ticker),
            score_components=None,
            score_breakdown=None,
            sources=[],
            conviction=0,
            sector=None,
            price=None,
        )
    for rank, ticker in enumerate(picks, 1):
        conv, agree = (convictions or {}).get(ticker, (None, None))
        insert_pick(
            session, run_id, rank=rank, ticker=ticker, conviction=conv, agreement_ratio=agree
        )
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
        _run(session, p.close["date"][100].isoformat(), ["T0"], others=("T1",))

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


def test_picks_from_different_universes_are_never_blended():
    from datetime import date

    from stock_analyzer.discover.pick_scorecard import _summarize

    eras = ((date.min, "S&P 500"), (date(2026, 9, 27), "US >= $2B quality"))
    entries = [
        ("AAA", date(2026, 9, 20), 10.0, 4.0),
        ("BBB", date(2026, 9, 28), 2.0, 4.0),  # same month, new universe
        ("CCC", date(2026, 10, 5), 8.0, 4.0),
    ]
    card = _summarize(entries, horizon=126, today=date(2027, 6, 1), eras=eras)
    assert [c["cohort"] for c in card["cohorts"]] == [
        "Sep 2026 · S&P 500",
        "Sep 2026 · US >= $2B quality",
        "Oct 2026 · US >= $2B quality",
    ]
    assert card["overall"] is None
    assert [(o["cohort"], o["picks"], o["excess_pct"]) for o in card["overall_by_universe"]] == [
        ("All · S&P 500", 1, 6.0),
        ("All · US >= $2B quality", 2, 1.0),
    ]
    one = _summarize(entries[:1], horizon=126, today=date(2027, 6, 1), eras=eras)
    assert one["cohorts"][0]["cohort"] == "Sep 2026" and one["overall_by_universe"] == []


def test_picks_are_set_against_the_screens_own_choice(tmp_path, monkeypatch):
    from stock_analyzer.discover import pick_scorecard as ps

    monkeypatch.setattr(ps, "SCREEN_TOP", 2)
    db = str(tmp_path / "v.db")
    with get_session(db) as session:
        # The screen ranks TOP1/TOP2 first; the model picked LOW instead.
        run = _run(
            session,
            "2026-01-12T10:00:00",
            ["LOW"],
            others=("TOP1", "TOP2", "MID"),
            scores={"TOP1": 90, "TOP2": 80, "MID": 50, "LOW": 10},
        )
        later = _run(session, "2026-01-13T10:00:00", ["LOW"], others=("TOP1",))
        for t, ret in (("LOW", -2.0), ("TOP1", 12.0), ("TOP2", 6.0), ("MID", 0.0)):
            _outcome(session, run, t, ret, 2.0)
        _outcome(session, later, "TOP1", 50.0, 2.0)  # same month: not counted twice
        _run(session, "2026-01-14T10:00:00", [], others=("NOPICKS",), scores={"NOPICKS": 99})

    (row,) = pick_scorecard(db, today=date(2026, 9, 26))["vs_screen"]
    assert row["cohort"] == "Jan 2026"
    assert (row["picks"]["picks"], row["picks"]["excess_pct"]) == (1, -4.0)
    assert (row["top"]["picks"], row["top"]["excess_pct"]) == (2, 7.0)
    # Every survivor of a run with picks, LOW included; not the pick-less run.
    assert (row["pool"]["picks"], row["pool"]["excess_pct"]) == (4, 2.0)


def test_calibration_splits_graded_picks_by_conviction_and_agreement(tmp_path):
    db = str(tmp_path / "c.db")
    with get_session(db) as session:
        run = _run(
            session,
            "2026-01-12T10:00:00",
            ["HI", "LO", "NONE"],
            convictions={"HI": (8, 1.0), "LO": (5, 1 / 3)},
        )
        _outcome(session, run, "HI", 10.0, 2.0)
        _outcome(session, run, "LO", -4.0, 2.0)
        _outcome(session, run, "NONE", 0.0, 2.0)

    cal = pick_scorecard(db, today=date(2026, 9, 26))["calibration"]
    assert cal["graded"] == 3
    assert [(r["group"], r["picks"], r["excess_pct"]) for r in cal["rows"]] == [
        ("Conviction 7+", 1, 8.0),
        ("Conviction under 7", 1, -6.0),
        ("All providers agreed", 1, 8.0),
        ("Split vote", 1, -6.0),
    ]


def test_screen_comparison_and_calibration_render():
    def c(excess, n):
        return {
            "cohort": "x",
            "picks": n,
            "return_pct": 0,
            "spy_pct": 0,
            "excess_pct": excess,
            "beat_spy": 0.5,
        }

    sc = {
        "horizon": 126,
        "cohorts": [{**c(-8.0, 22), "cohort": "May 2026"}],
        "overall": None,
        "maturing": 0,
        "next_due": None,
        "unmeasured": [],
        "vs_screen": [
            {"cohort": "May 2026", "picks": c(-8.0, 22), "top": c(3.5, 60), "pool": None}
        ],
        "calibration": {"graded": 12, "rows": []},
    }
    body = render_health_html(build_portfolio_health({}, pick_scorecard=lambda: sc))
    assert "Picks vs the screen they came from" in body and "Screen top 10" in body
    assert "+3.5% <small>(50% of 60)</small>" in body
    assert "shown once 50 picks are graded (12 so far)" in body

    rows = [{"group": "Conviction 7+", **c(4.0, 30)}, {"group": "Split vote", **c(-1.0, 25)}]
    sc["calibration"] = {"graded": 55, "rows": rows}
    body = render_health_html(build_portfolio_health({}, pick_scorecard=lambda: sc))
    assert "Does conviction mean anything?" in body and "Conviction 7+" in body
