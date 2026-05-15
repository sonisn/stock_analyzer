"""End-to-end roundtrip: insert through every repository function, read back,
assert equality. Catches SQLModel mapping regressions (wrong column names,
JSON marshalling bugs, FK direction mistakes) that the existing test suite
doesn't exercise directly."""
from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import select

from stock_analyzer.db.repository import (
    fetch_recent_holdings_history,
    insert_candidate,
    insert_holdings_review,
    insert_pick,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from stock_analyzer.db.session import get_session
from stock_analyzer.db.tables import (
    Candidate,
    HoldingReviewRow,
    Pick,
    Run,
    RunOutput,
    Scorecard,
)


def test_roundtrip_through_repository(tmp_path: Path) -> None:
    db_path = tmp_path / "roundtrip.db"

    # Insert
    with get_session(str(db_path)) as s:
        run_id = insert_run(
            s,
            universe_size=42, survivors=10, picks=3,
            opus_model="claude-opus-4-7", sonnet_model="claude-sonnet-4-6",
            cash_budget=5000.0, kind="rebalance",
        )
        insert_candidate(
            s, run_id, "NVDA",
            passed_filter=True,
            fail_reasons=[],
            score=8.7,
            score_components={"fundamentals": 9, "trend": 8, "conviction": 9},
            score_breakdown={"detail": "fine"},
            sources=["finnhub", "yfinance"],
            conviction=9,
            sector="Technology",
            price=900.25,
        )
        insert_scorecard(s, run_id, "NVDA", "great fundamentals")
        insert_pick(
            s, run_id,
            rank=1, ticker="NVDA",
            ranker_text="pick one prose",
            bear_case_text="bear prose",
            allocation_text="35%",
        )
        insert_holdings_review(
            s, run_id, "NVDA",
            verdict="HOLD", confidence=8, review_text="stay the course",
        )
        insert_run_outputs(
            s, run_id,
            ranker_full="r", redteam_full="rt", sizer_full="sz",
            holdings_summary="h",
            rebalance_text="reb",
            dashboard_data={"x": 1, "y": [2, 3]},
        )

    # Read back
    with get_session(str(db_path)) as s:
        run = s.exec(select(Run).where(Run.id == run_id)).one()
        assert run.universe_size == 42
        assert run.kind == "rebalance"
        assert run.cash_budget == 5000.0

        cand = s.exec(
            select(Candidate).where(
                Candidate.run_id == run_id, Candidate.ticker == "NVDA"
            )
        ).one()
        assert cand.passed_filter == 1
        assert json.loads(cand.fail_reasons) == []
        assert json.loads(cand.score_components) == {
            "fundamentals": 9, "trend": 8, "conviction": 9,
        }
        assert json.loads(cand.sources) == ["finnhub", "yfinance"]
        assert cand.sector == "Technology"

        sc = s.exec(select(Scorecard).where(Scorecard.run_id == run_id)).one()
        assert sc.analyst_text == "great fundamentals"

        pk = s.exec(
            select(Pick).where(Pick.run_id == run_id, Pick.rank == 1)
        ).one()
        assert pk.ticker == "NVDA"
        assert pk.allocation_text == "35%"

        hr = s.exec(
            select(HoldingReviewRow).where(HoldingReviewRow.run_id == run_id)
        ).one()
        assert hr.verdict == "HOLD"
        assert hr.confidence == 8

        ro = s.exec(
            select(RunOutput).where(RunOutput.run_id == run_id)
        ).one()
        assert ro.ranker_full == "r"
        assert json.loads(ro.dashboard_data) == {"x": 1, "y": [2, 3]}


def test_fetch_recent_holdings_history_returns_chronological(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "history.db"
    with get_session(str(db_path)) as s:
        for i in range(3):
            rid = insert_run(
                s,
                universe_size=1, survivors=1, picks=0,
                opus_model="x", sonnet_model="y", cash_budget=None,
                kind="rebalance",
            )
            insert_holdings_review(
                s, rid, "AAPL",
                verdict="HOLD", confidence=7 + i, review_text=f"r{i}",
            )

    with get_session(str(db_path)) as s:
        hist = fetch_recent_holdings_history(s, n_runs=3, kind="rebalance")
    assert "AAPL" in hist
    confidences = [row["confidence"] for row in hist["AAPL"]]
    assert confidences == [7, 8, 9]  # oldest-first
