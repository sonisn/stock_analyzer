"""A later run on the same day is a revision, not a duplicate.

On 2026-09-20 two rebalances both said BUY LLY — the first at ~$26,000,
the second at $18,450 once it could see the covered calls and the real
cash. The ledger kept the first, which is the sizing the grading would
later assume was acted on.
"""

from __future__ import annotations

from pathlib import Path

from sqlmodel import select

from stock_analyzer.db.repository import record_suggestions
from stock_analyzer.db.session import get_session
from stock_analyzer.db.tables import Suggestion


def _row(**over):
    base = dict(
        suggested_on="2026-09-20",
        source="rebalance",
        action="BUY",
        ticker="LLY",
        detail="~$26,000 in Traditional IRA",
        price=1152.93,
        units_held=0.0,
        run_id=35,
        reinvest_into=None,
    )
    return base | over


def _rows(db: str) -> list[dict]:
    """Plain values — an ORM object read after its session closes raises."""
    with get_session(db) as s:
        return [
            dict(run_id=r, detail=d, source=src, action=a, ticker=t)
            for r, d, src, a, t in s.exec(
                select(
                    Suggestion.run_id,
                    Suggestion.detail,
                    Suggestion.source,
                    Suggestion.action,
                    Suggestion.ticker,
                )
            ).all()
        ]


def test_a_newer_run_replaces_the_advice_on_record(tmp_path: Path):
    db = str(tmp_path / "s.db")
    with get_session(db) as s:
        assert record_suggestions(s, [_row()]) == 1
        s.commit()
    with get_session(db) as s:
        # The same advice, resized by a later run that knew more.
        added = record_suggestions(
            s, [_row(run_id=36, detail="~$18,450 (16 shares) in Traditional IRA")]
        )
        s.commit()
    assert added == 0, "a revision is not new advice"
    rows = _rows(db)
    assert len(rows) == 1, "and it must not become a second row"
    assert rows[0]["run_id"] == 36
    assert "18,450" in rows[0]["detail"]


def test_an_older_run_never_overwrites_a_newer_one(tmp_path: Path):
    db = str(tmp_path / "s.db")
    with get_session(db) as s:
        record_suggestions(s, [_row(run_id=36, detail="newer")])
        record_suggestions(s, [_row(run_id=35, detail="older")])
        s.commit()
    rows = _rows(db)
    assert rows[0]["run_id"] == 36 and rows[0]["detail"] == "newer"


def test_re_running_the_daily_email_is_still_harmless(tmp_path: Path):
    """The daily email carries no run id; repeating it must change nothing."""
    db = str(tmp_path / "s.db")
    daily = _row(source="daily", run_id=None, detail="Consider selling AVGO")
    with get_session(db) as s:
        record_suggestions(s, [daily])
        s.commit()
    with get_session(db) as s:
        assert record_suggestions(s, [daily]) == 0
        s.commit()
    rows = _rows(db)
    assert len(rows) == 1 and rows[0]["detail"] == "Consider selling AVGO"


def test_different_advice_on_the_same_day_is_kept_separately(tmp_path: Path):
    db = str(tmp_path / "s.db")
    with get_session(db) as s:
        record_suggestions(
            s,
            [
                _row(),
                _row(action="TRIM", ticker="TSLA"),
                _row(source="discover", detail="discover pick #1", run_id=31),
            ],
        )
        s.commit()
    assert len(_rows(db)) == 3
