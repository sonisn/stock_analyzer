"""The evidence score: a control ranking from tested signals only."""

from __future__ import annotations

import json
from datetime import date

from stock_analyzer.db.session import get_session
from stock_analyzer.discover.evidence import WEIGHTS, evidence_scores, hidden_from_model
from stock_analyzer.discover.pick_scorecard import evidence_of, pick_scorecard
from tests.test_pick_scorecard import _outcome, _run


def test_scores_rank_within_the_run_and_absence_is_neutral():
    got = evidence_scores(
        ["aaa", "BBB", "CCC", "DDD"],
        book_yoy={"AAA": 60.0, "BBB": 10.0, "CCC": -20.0, "ZZZ": 999.0},  # ZZZ not in run
        gross_profitability={"AAA": 0.1, "BBB": 0.5},
        insider_clusters={"CCC"},
    )
    assert set(got) == {"AAA", "BBB", "CCC", "DDD"}
    assert got["AAA"]["contracted_book"] == 1.0 and got["CCC"]["contracted_book"] == 0.0
    assert got["DDD"]["contracted_book"] == 0.5 and got["DDD"]["gross_profitability"] == 0.5
    assert got["BBB"]["gross_profitability"] == 1.0 and got["AAA"]["gross_profitability"] == 0.0
    assert got["CCC"]["insider_cluster"] == 1.0
    expected = 100 * (WEIGHTS["contracted_book"] * 1.0 + WEIGHTS["gross_profitability"] * 0.0)
    assert got["AAA"]["score"] == round(expected, 1)
    assert sum(WEIGHTS.values()) == 1.0


def test_the_model_never_sees_the_evidence_score():
    breakdown = {"fundamentals": {"fcf_yield": 3.0}, "evidence": {"score": 80.0}}
    assert hidden_from_model(breakdown) == {"fundamentals": {"fcf_yield": 3.0}}
    assert hidden_from_model(None) == {}


def test_scorecard_grades_the_evidence_top_beside_the_picks(tmp_path, monkeypatch):
    from sqlalchemy import text

    from stock_analyzer.discover import pick_scorecard as ps

    monkeypatch.setattr(ps, "SCREEN_TOP", 1)
    db = str(tmp_path / "e.db")
    with get_session(db) as session:
        run = _run(
            session,
            "2026-01-12T10:00:00",
            ["PICK"],
            others=("EV", "LOW"),
            scores={"PICK": 90, "EV": 10, "LOW": 5},
        )
        session.flush()  # the rows must exist before the raw UPDATE
        for t, ev in (("PICK", 20.0), ("EV", 95.0), ("LOW", 40.0)):
            session.exec(
                text(
                    "UPDATE candidates SET score_breakdown = :b WHERE run_id = :r AND ticker = :t"
                ),
                params={"b": json.dumps({"evidence": {"score": ev}}), "r": run, "t": t},
            )
        for t, ret in (("PICK", 1.0), ("EV", 9.0), ("LOW", -3.0)):
            _outcome(session, run, t, ret, 2.0)

    (row,) = pick_scorecard(db, today=date(2026, 9, 26))["vs_screen"]
    assert (row["picks"]["excess_pct"], row["top"]["excess_pct"]) == (-1.0, -1.0)
    assert (row["evidence"]["picks"], row["evidence"]["excess_pct"]) == (1, 7.0)
    assert evidence_of(None) is None and evidence_of('{"evidence": {"score": 5}}') == 5.0
