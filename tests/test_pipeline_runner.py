"""The discover/rebalance step runner (pipeline.py)."""

from __future__ import annotations

import sqlite3

import pytest

from stock_analyzer.pipeline import Parallel, PipelineFailed, Step, run_pipeline


def _step(name, calls, *, fails=False):
    def run():
        calls.append(name)
        if fails:
            raise RuntimeError(f"{name} broke")
        return f"{name} done"

    return Step(name, run)


def test_steps_run_in_order_and_parallel_blocks_run_every_step():
    calls: list[str] = []
    result = run_pipeline(
        "t",
        [
            _step("a", calls),
            Parallel(_step("b", calls), _step("c", calls), name="block"),
            _step("d", calls),
        ],
    )
    assert calls[0] == "a" and sorted(calls[1:3]) == ["b", "c"] and calls[3] == "d"
    assert [r.status for r in result.records] == ["ok"] * 4
    assert {r.block for r in result.records} == {None, "block"}


def test_a_failed_step_stops_the_run_and_is_not_retried():
    """A re-run would re-pay every model call the step already made."""
    calls: list[str] = []
    with pytest.raises(PipelineFailed) as e:
        run_pipeline("t", [_step("a", calls, fails=True), _step("b", calls)])
    assert calls == ["a"]
    assert e.value.step == "a"


def test_a_failed_parallel_step_lets_the_block_and_the_run_finish():
    calls: list[str] = []
    result = run_pipeline(
        "t",
        [
            Parallel(_step("news", calls, fails=True), _step("earnings", calls), name="enrich"),
            _step("analyst", calls),
        ],
    )
    assert "analyst" in calls and "earnings" in calls
    (failed,) = result.failed
    assert (failed.step, failed.block) == ("news", "enrich")
    assert "news broke" in (failed.detail or "")


def test_the_step_log_is_kept_in_the_database(tmp_path):
    db = tmp_path / "run.db"
    calls: list[str] = []
    with pytest.raises(PipelineFailed):
        run_pipeline("t", [_step("a", calls), _step("b", calls, fails=True)], db_path=str(db))
    rows = (
        sqlite3.connect(db)
        .execute("SELECT step, status, detail FROM pipeline_steps ORDER BY id")
        .fetchall()
    )
    assert rows == [("a", "ok", "a done"), ("b", "failed", "RuntimeError: b broke")]


def test_a_step_log_that_cannot_be_written_does_not_fail_the_run(tmp_path):
    calls: list[str] = []
    result = run_pipeline("t", [_step("a", calls)], db_path=str(tmp_path / "no" / "\0bad"))
    assert result.records[0].status == "ok"
