"""Run the scheduled jobs a reboot or outage made the machine miss.

Cron has no catch-up: the Oct 14-15 2026 reboot (machine down 23:47 →
12:49 UTC) silently skipped that night's earnings watch, the nightly
backup, the IBD update and the code update. `ops catch-up` runs at boot
(`@reboot` in the crontab): for every `NY_AT=` job it finds the latest
time the job was due, and when that was in the last CATCH_UP_HOURS and
the job's log shows no run since, it runs the job now — in schedule order,
one at a time.

The schedule is read from the crontab itself, so there is no second copy
to keep in step. Cron's day fields are matched in UTC, as cron does
(this machine's clock is UTC); the hour is the job's New York NY_AT.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .logging import get_logger

logger = get_logger(__name__)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Older than this, a missed run has been overtaken by the next one (or
# would arrive too late to matter — a two-day-old weekly email).
CATCH_UP_HOURS = 36

# The job's log name, from the script's `exec .../run_job.sh" <name> ...` line.
_JOB_RE = re.compile(r'^exec .*run_job\.sh"\s+([\w-]+)', re.M)
_START_RE = re.compile(r"^=== [\w-]+: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class CronJob:
    minute: str
    hours: str
    dom: str
    month: str
    dow: str
    ny_at: str  # "HH:MM"
    script: Path
    job: str  # log name, from the run_job.sh call in the script


def _field_values(field: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in field.split(","):
        rng, _, step = part.partition("/")
        if rng == "*":
            a, b = lo, hi
        elif "-" in rng:
            a, b = (int(x) for x in rng.split("-"))
        else:
            a = b = int(rng)
        out.update(range(a, b + 1, int(step) if step else 1))
    if hi == 7 and 7 in out:  # cron allows 7 for Sunday
        out.add(0)
    return out


def _day_matches(job: CronJob, when: datetime) -> bool:
    """Cron's day rule on a UTC time: with both day-of-month and day-of-week
    restricted, either may match; otherwise both must."""
    dom_ok = when.day in _field_values(job.dom, 1, 31)
    dow_ok = (when.isoweekday() % 7) in _field_values(job.dow, 0, 7)
    month_ok = when.month in _field_values(job.month, 1, 12)
    if job.dom != "*" and job.dow != "*":
        return month_ok and (dom_ok or dow_ok)
    return month_ok and dom_ok and dow_ok


def parse_crontab(text: str) -> list[CronJob]:
    """The `NY_AT=` jobs in a crontab; anything else is left alone."""
    jobs = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 7 or line.lstrip().startswith("#") or not fields[5].startswith("NY_AT="):
            continue
        script = Path(fields[6])
        try:
            name = _JOB_RE.search(script.read_text())
        except OSError:
            continue
        if name:
            minute, hours, dom, month, dow = fields[:5]
            ny_at = fields[5].removeprefix("NY_AT=")
            jobs.append(CronJob(minute, hours, dom, month, dow, ny_at, script, name[1]))
    return jobs


def last_due(job: CronJob, now: datetime) -> datetime | None:
    """The latest NY time at or before `now` the job was due, within
    CATCH_UP_HOURS, or None."""
    hh, mm = (int(x) for x in job.ny_at.split(":"))
    hours = _field_values(job.hours, 0, 23)
    today = now.astimezone(NY).date()
    for back in range(CATCH_UP_HOURS // 24 + 2):
        day = today - timedelta(days=back)
        due = datetime(day.year, day.month, day.day, hh, mm, tzinfo=NY)
        if due > now or now - due > timedelta(hours=CATCH_UP_HOURS):
            continue
        utc = due.astimezone(UTC)
        if utc.hour in hours and _day_matches(job, utc):
            return due
    return None


def ran_since(log_dir: Path, job: str, due: datetime, now: datetime) -> bool:
    """Does any of the job's logs show a start between `due` and `now`?
    Logs are named by the NY day the run started, so look at that day and
    after."""
    day = due.date()
    for path in sorted(log_dir.glob(f"{job}_*.log")):
        stamp = path.stem.rsplit("_", 1)[-1]
        if len(stamp) != 8 or stamp < day.strftime("%Y%m%d"):
            continue
        for line in path.read_text(errors="replace").splitlines():
            m = _START_RE.match(line)
            if (
                m
                and due - timedelta(minutes=5)
                <= datetime.fromisoformat(m[1]).replace(tzinfo=NY)
                <= now
            ):
                return True
    return False


def missed(jobs: list[CronJob], log_dir: Path, now: datetime) -> list[tuple[datetime, CronJob]]:
    """(due, job) for each job that was due and has not run since, oldest first."""
    out = []
    for job in jobs:
        due = last_due(job, now)
        # A job due in the last few minutes may be starting right now.
        if due is None or now - due < timedelta(minutes=5):
            continue
        if not ran_since(log_dir, job.job, due, now):
            out.append((due, job))
    return sorted(out, key=lambda x: x[0])


def catch_up(log_dir: Path, *, dry_run: bool = False, now: datetime | None = None) -> int:
    """Run every missed job; returns how many were missed."""
    now = now or datetime.now(NY)
    crontab = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    todo = missed(parse_crontab(crontab.stdout), log_dir, now)
    if not todo:
        print("Nothing missed.")
    for due, job in todo:
        print(
            f"{job.job}: due {due:%a %b %d %H:%M} NY, not run — {'would run' if dry_run else 'running'}"
        )
        if not dry_run:
            # Without NY_AT, run_job.sh runs the job whatever the hour.
            env = {k: v for k, v in os.environ.items() if k != "NY_AT"}
            result = subprocess.run([str(job.script)], env=env, check=False)
            print(f"{job.job}: finished with status {result.returncode}")
    return len(todo)
