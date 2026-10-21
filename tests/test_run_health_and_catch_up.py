"""Run health (a shortfall gets noticed), catch-up after a reboot, and the
quarterly verdict on the model step."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from stock_analyzer import catch_up, run_health
from stock_analyzer.discover.pick_scorecard import model_step_verdict

# --- run health ----------------------------------------------------------------


def test_shortfalls_flag_a_dead_source_not_one_missing_chart():
    assert run_health.shortfalls({"charts": (13, 14)}) == []
    assert run_health.shortfalls({"charts": (0, 14)}) == ["charts 0/14"]
    assert run_health.shortfalls({"charts": (0, 0)}) == []


def test_shortfalls_name_failed_steps():
    got = run_health.shortfalls({}, {"stock analysis": ["NVDA", "AMD"]})
    assert got == ["stock analysis failed (NVDA, AMD)"]


def test_record_and_doctor_check(tmp_path: Path):
    path = tmp_path / "rh.json"
    assert run_health.doctor_check(path) == "no runs recorded yet"
    run_health.record("portfolio", {"charts": (14, 14)}, [], path=path, now=datetime(2026, 10, 21))
    assert "portfolio 2026-10-21 ok" in run_health.doctor_check(path)
    run_health.record("portfolio", {"charts": (0, 14)}, ["charts 0/14"], path=path)
    with pytest.raises(RuntimeError, match="charts 0/14"):
        run_health.doctor_check(path)


# --- catch-up -------------------------------------------------------------------


def _script(tmp_path: Path, job: str) -> Path:
    p = tmp_path / f"run_{job}.sh"
    p.write_text(f'# run_job.sh then something\nexec "$(dirname)/run_job.sh" {job} uv run x\n')
    return p


def _crontab(tmp_path: Path) -> str:
    return "\n".join(
        [
            "# comment",
            f"0 6,7 * * * NY_AT=02:00 {_script(tmp_path, 'backup')}",
            f"0 2,3 * * 2-6 NY_AT=22:00 {_script(tmp_path, 'earnings-watch')}",
            f"30 13,14 * * 3 NY_AT=09:30 {_script(tmp_path, 'portfolio')}",
            "@reboot /x/run_catch_up.sh",
        ]
    )


def test_parse_crontab_reads_job_names_from_exec_line(tmp_path: Path):
    jobs = catch_up.parse_crontab(_crontab(tmp_path))
    assert [j.job for j in jobs] == ["backup", "earnings-watch", "portfolio"]


def test_reboot_outage_finds_the_skipped_jobs(tmp_path: Path):
    """Down Wed Oct 14 23:47 UTC → Thu Oct 15 12:49 UTC (08:49 NY)."""
    jobs = catch_up.parse_crontab(_crontab(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "portfolio_20261014.log").write_text("=== portfolio: 2026-10-14 09:30:01 EDT ===\n")
    now = datetime(2026, 10, 15, 8, 55, tzinfo=catch_up.NY)
    got = [(d.strftime("%a %H:%M"), j.job) for d, j in catch_up.missed(jobs, logs, now)]
    # Wed 22:00 NY is Thu 02:00 UTC: inside cron's UTC day field 2-6.
    assert got == [("Wed 22:00", "earnings-watch"), ("Thu 02:00", "backup")]


def test_a_run_after_the_due_time_counts(tmp_path: Path):
    jobs = catch_up.parse_crontab(_crontab(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "backup_20261015.log").write_text("=== backup: 2026-10-15 02:00:01 EDT ===\n")
    (logs / "portfolio_20261014.log").write_text("=== portfolio: 2026-10-14 09:30:01 EDT ===\n")
    (logs / "earnings-watch_20261014.log").write_text(
        "=== earnings-watch: 2026-10-14 22:00:01 EDT ===\n"
    )
    now = datetime(2026, 10, 15, 8, 55, tzinfo=catch_up.NY)
    assert catch_up.missed(jobs, logs, now) == []


def test_old_misses_are_left_alone(tmp_path: Path):
    """The weekly email missed on Wednesday is not sent on Friday."""
    jobs = [j for j in catch_up.parse_crontab(_crontab(tmp_path)) if j.job == "portfolio"]
    now = datetime(2026, 10, 16, 12, 0, tzinfo=catch_up.NY)
    assert catch_up.missed(jobs, tmp_path, now) == []


# --- model-step verdict ------------------------------------------------------------


def _month(name: str, picks: float, top: float, n: int = 20) -> dict:
    return {
        "cohort": name,
        "picks": {"excess_pct": picks, "picks": n},
        "top": {"excess_pct": top, "picks": 10},
    }


def test_verdict_too_early():
    assert model_step_verdict([_month("Nov 2026", 5, 0)])["verdict"] == "too early"


def test_verdict_not_adding_value():
    rows = [_month("Nov 2026", -3, 2), _month("Dec 2026", -1, 1), _month("Jan 2027", 0, 1)]
    v = model_step_verdict(rows + [_month("All", -1, 1)])
    assert v["verdict"] == "not adding value" and v["won"] == 0 and v["cohorts"] == 3


def test_verdict_adding_value_and_undecided():
    good = [_month("Nov 2026", 6, 1), _month("Dec 2026", 4, 1)]
    assert model_step_verdict(good)["verdict"] == "adding value"
    mixed = [_month("Nov 2026", 6, 1), _month("Dec 2026", -4, 1)]
    assert model_step_verdict(mixed)["verdict"] == "undecided"
