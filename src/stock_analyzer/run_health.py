"""What a scheduled run delivered against what it meant to, so a run that
quietly fell back on something gets noticed.

Every step of the weekly email has a fallback — no chart, no news, a
placeholder block — so one broken source never costs the whole email. The
price was silence: chart-img refused the key on every request from
2026-10-14, the job still exited 0, and two weekly emails went out with
0/14 charts before anyone looked. Now each run records its counts here; a
shortfall is listed in red in the email itself, and `ops doctor` fails
while the latest run of any job has one (which emails an alert).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)

STATE_PATH = Path(os.path.expanduser("~/.stock_analyzer/run_health.json"))

# Below this share of what was expected, a count is a problem. One missing
# chart out of 14 is a symbol chart-img doesn't carry; 0 of 14 is a source down.
MIN_SHARE = 0.8


def shortfalls(
    delivered: dict[str, tuple[int, int]],
    failures: dict[str, list[str]] | None = None,
    *,
    min_share: float = MIN_SHARE,
) -> list[str]:
    """Plain-English problems: each `{name: (got, expected)}` short of
    `min_share`, then each step that fell back (`{step: [what, ...]}`)."""
    out = [
        f"{name} {got}/{expected}"
        for name, (got, expected) in delivered.items()
        if expected and got < expected * min_share
    ]
    for step, what in (failures or {}).items():
        out.append(f"{step} failed ({', '.join(what[:6])}{'…' if len(what) > 6 else ''})")
    return out


def record(
    job: str,
    delivered: dict[str, tuple[int, int]],
    problems: list[str],
    *,
    path: Path = STATE_PATH,
    now: datetime | None = None,
) -> None:
    """Store this run's counts and problems under `job`, replacing the last
    run's. Never raises: the record is a check on the run, not part of it."""
    try:
        state = load(path)
        state[job] = {
            "at": (now or datetime.now()).isoformat(timespec="seconds"),
            "delivered": {k: list(v) for k, v in delivered.items()},
            "problems": problems,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record run health for %s (%s)", job, e)
    if problems:
        logger.warning("Run health for %s: %s", job, "; ".join(problems))


def load(path: Path = STATE_PATH) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return {}


def doctor_check(path: Path = STATE_PATH) -> str:
    """For `ops doctor`: raises while any job's latest run had problems."""
    state = load(path)
    if not state:
        return "no runs recorded yet"
    bad = [
        f"{job} ({r['at'][:10]}): {'; '.join(r['problems'])}"
        for job, r in state.items()
        if r.get("problems")
    ]
    if bad:
        raise RuntimeError(" | ".join(bad))
    return ", ".join(f"{job} {r['at'][:10]} ok" for job, r in sorted(state.items()))
