"""Per-run history upkeep: what is trimmed, what is always kept."""

from __future__ import annotations

import os
import time
from datetime import date

from sqlalchemy import text

from stock_analyzer.db.repository import (
    insert_candidate,
    insert_holdings_review,
    insert_pick,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from stock_analyzer.db.retention import (
    RetentionPolicy,
    prune_database,
    prune_files,
    run_history_upkeep,
)
from stock_analyzer.db.session import get_session
from stock_analyzer.db.tables import ModelVersion

TODAY = date(2027, 12, 1)


def _run(session, run_at: str) -> int:
    run_id = insert_run(
        session,
        universe_size=2,
        survivors=1,
        picks=1,
        opus_model="o",
        sonnet_model="s",
        cash_budget=None,
    )
    session.exec(
        text("UPDATE runs SET run_at = :r WHERE id = :i"), params={"r": run_at, "i": run_id}
    )
    for ticker, passed in (("PASS", True), ("FAIL", False), ("PICKED", False)):
        insert_candidate(
            session,
            run_id,
            ticker,
            passed_filter=passed,
            fail_reasons=[],
            score=None,
            score_components=None,
            score_breakdown=None,
            sources=[],
            conviction=0,
            sector=None,
            price=10.0,
        )
    insert_pick(session, run_id, rank=1, ticker="PICKED", conviction=7)
    insert_scorecard(session, run_id, "PASS", "long analyst prose")
    insert_holdings_review(
        session, run_id, "HELD", verdict="HOLD", confidence=7, review_text="long review prose"
    )
    insert_run_outputs(
        session,
        run_id,
        ranker_full="ranker prose",
        redteam_full="red team prose",
        sizer_full="TICKER: PICKED\nAllocation: 100%",
        holdings_summary="h",
        rebalance_text="plan",
        dashboard_data={"x": 1},
    )
    return run_id


def _seed(db: str) -> tuple[int, int]:
    with get_session(db) as session:
        old = _run(session, "2026-01-05T10:00:00")  # older than every window
        new = _run(session, "2027-11-20T10:00:00")
        session.exec(text("CREATE TABLE workflow_session (session_id TEXT, created_at INTEGER)"))
        now = int(time.mktime(TODAY.timetuple()))
        session.exec(
            text("INSERT INTO workflow_session VALUES ('old', :o), ('new', :n)"),
            params={"o": now - 90 * 86400, "n": now - 86400},
        )
        for i in range(15):
            session.add(
                ModelVersion(
                    created_at="x",
                    horizon_days=21,
                    population="gated",
                    features="[]",
                    coefficients="{}",
                    metrics="{}",
                    train_start="",
                    train_end="",
                    accepted=int(i == 0),
                )
            )
    return old, new


def test_prune_trims_old_prose_and_logs_but_keeps_analysis_rows(tmp_path):
    db = str(tmp_path / "r.db")
    old, new = _seed(db)
    out = prune_database(db, RetentionPolicy(keep_models=12), today=TODAY)

    assert out["failed_candidates"] == 1 and out["workflow_session"] == 1
    assert out["model_versions"] == 2  # 15 - newest 12 - the accepted v1
    assert out["prose_fields"] == 7  # every prose column of the old run
    with get_session(db) as s:
        q = lambda sql: s.exec(text(sql)).all()  # noqa: E731
        # Old run: prose blanked, numbers and the sizer text kept.
        assert q(
            f"SELECT ranker_full, redteam_full, sizer_full FROM run_outputs WHERE run_id={old}"
        ) == [(None, None, "TICKER: PICKED\nAllocation: 100%")]
        assert q(
            f"SELECT verdict, confidence, review_text FROM holdings_reviews WHERE run_id={old}"
        ) == [("HOLD", 7, None)]
        # Failed candidates go, survivors and picked names stay.
        assert sorted(q(f"SELECT ticker FROM candidates WHERE run_id={old}")) == [
            ("PASS",),
            ("PICKED",),
        ]
        assert q(f"SELECT conviction FROM picks WHERE run_id={old}") == [(7,)]
        # The recent run is untouched.
        assert q(f"SELECT ranker_full FROM run_outputs WHERE run_id={new}") == [("ranker prose",)]
        assert len(q(f"SELECT 1 FROM candidates WHERE run_id={new}")) == 3
        assert q("SELECT session_id FROM workflow_session") == [("new",)]
        assert q("SELECT id FROM model_versions WHERE id = 1") == [(1,)]  # accepted kept
    # Idempotent.
    assert sum(prune_database(db, RetentionPolicy(), today=TODAY).values()) == 0


def test_prune_files_matches_only_app_patterns_and_age(tmp_path):
    old_log, new_log, other = (
        tmp_path / n for n in ("stock-analyzer-1.log", "stock-analyzer-2.log", "notes.log")
    )
    for f in (old_log, new_log, other):
        f.write_text("x")
    past = time.time() - 40 * 86400
    os.utime(old_log, (past, past))
    os.utime(other, (past, past))
    assert prune_files([(tmp_path, "stock-analyzer-*.log")], 30) == 1
    assert not old_log.exists() and new_log.exists() and other.exists()


def test_upkeep_never_raises_and_reports_each_step(tmp_path):
    db = str(tmp_path / "u.db")
    _seed(db)
    report = run_history_upkeep(
        db,
        policy=RetentionPolicy(),
        file_targets=[(tmp_path / "missing-dir", "*.log")],
        today=TODAY,
        label_outcomes=False,
    )
    assert report.errors == []
    assert report.trimmed["failed_candidates"] == 1
    assert "trimmed:" in report.summary()

    broken = run_history_upkeep(
        str(tmp_path / "no" / "such" / "\0bad.db"),
        policy=RetentionPolicy(),
        file_targets=[],
        label_outcomes=False,
    )
    assert broken.errors  # logged, not raised
