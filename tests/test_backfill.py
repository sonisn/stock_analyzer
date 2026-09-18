"""Backfilling pick forecast fields from what each run stored at the time."""

from __future__ import annotations

from sqlalchemy import text

from stock_analyzer.db.backfill import backfill_pick_forecasts, parse_pick_forecasts
from stock_analyzer.db.repository import (
    insert_candidate,
    insert_pick,
    insert_run,
    insert_run_outputs,
)
from stock_analyzer.db.session import get_session

RANKER = """\
PICK 1: NVDA — AI franchise.

Conviction (1-10): 7
Time horizon: 6-12 months
---
PICK 2: BRK.B — Compounder.

Conviction (1-10): 11
Time horizon: 12-18 months
---
PICK 3: XYZ — No forecast lines here.
"""


def test_parse_reads_each_pick_block_and_rejects_out_of_range():
    parsed = parse_pick_forecasts(RANKER)
    assert parsed["NVDA"] == {"conviction": 7, "time_horizon": "6-12 months"}
    assert parsed["BRK.B"] == {"time_horizon": "12-18 months"}  # 11 is out of range
    assert parsed["XYZ"] == {}


def test_backfill_fills_only_nulls_and_is_idempotent(tmp_path):
    db = str(tmp_path / "b.db")
    with get_session(db) as session:
        run_id = insert_run(
            session,
            universe_size=3,
            survivors=3,
            picks=3,
            opus_model="o",
            sonnet_model="s",
            cash_budget=None,
        )
        for rank, (ticker, price) in enumerate(
            [("NVDA", 120.5), ("BRK.B", 400.0), ("XYZ", None)], 1
        ):
            insert_candidate(
                session,
                run_id,
                ticker,
                passed_filter=True,
                fail_reasons=[],
                score=50.0,
                score_components={},
                score_breakdown={},
                sources=[],
                conviction=0,
                sector=None,
                price=price,
            )
            insert_pick(
                session,
                run_id,
                rank=rank,
                ticker=ticker,
                conviction=9 if ticker == "BRK.B" else None,
            )
        insert_run_outputs(
            session, run_id, ranker_full=RANKER, redteam_full="", sizer_full="", holdings_summary=""
        )

    assert backfill_pick_forecasts(db) == {"conviction": 1, "time_horizon": 2, "entry_price": 2}
    assert backfill_pick_forecasts(db) == {"conviction": 0, "time_horizon": 0, "entry_price": 0}
    with get_session(db) as session:
        rows = session.exec(
            text("SELECT ticker, conviction, time_horizon, entry_price FROM picks ORDER BY rank")
        ).all()
    assert rows == [
        ("NVDA", 7, "6-12 months", 120.5),
        ("BRK.B", 9, "12-18 months", 400.0),  # existing conviction kept
        ("XYZ", None, None, None),  # nothing recorded, nothing invented
    ]
