"""Per-run history upkeep: add what is new, trim what is old.

Runs as the last step of every discover and rebalance run. Nothing here
calls an LLM, and a failure only logs — upkeep must never fail a run.

ADD
  - fill any pick forecast fields still NULL (db/backfill.py);
  - label the realized 21/63-day outcome of this and earlier runs' screen
    survivors whose windows have closed (full candidate labeling, which
    needs a much larger price download, stays in the monthly model-review).

TRIM — only data nothing reads back at full fidelity:
  - LLM prose older than `text_days` (analyst scorecards, holding-review
    text, ranker / red-team / rebalance text, dashboard JSON) is blanked;
    the rows and every number in them stay. `run_outputs.sizer_full` is
    never trimmed: the paper-portfolio ledger re-reads it for every run;
  - agno's step-by-step session log older than `session_days`;
  - candidates that FAILED the screen, older than `candidate_days` (the
    longest window any check reads — validate-screen and calibration look
    back 540 days); screen survivors, picks, scenarios, catalysts, review
    verdicts, outcomes and fundamentals snapshots are always kept, because
    the track record, calibration and model depend on them;
  - model versions beyond the newest `keep_models` (accepted ones are kept);
  - log files and cached price panels untouched for `file_days`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import text

from ..logging import get_logger
from .session import get_session

logger = get_logger(__name__)

# Blanking prose on these columns loses nothing the code reads back.
_PROSE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("scorecards", "analyst_text"),
    ("holdings_reviews", "review_text"),
    ("run_outputs", "ranker_full"),
    ("run_outputs", "redteam_full"),
    ("run_outputs", "holdings_summary"),
    ("run_outputs", "rebalance_text"),
    ("run_outputs", "dashboard_data"),
)


@dataclass
class RetentionPolicy:
    text_days: int = 365
    session_days: int = 30
    candidate_days: int = 540
    keep_models: int = 12
    file_days: int = 30


@dataclass
class UpkeepReport:
    added: dict[str, int] = field(default_factory=dict)
    trimmed: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        def fmt(d: dict[str, int]) -> str:
            return ", ".join(f"{k}={v}" for k, v in d.items() if v) or "nothing"

        text_ = f"History upkeep — added: {fmt(self.added)}; trimmed: {fmt(self.trimmed)}"
        return text_ + (f"; errors: {'; '.join(self.errors)}" if self.errors else "")


def prune_database(db_path: str, policy: RetentionPolicy, *, today: date) -> dict[str, int]:
    text_cutoff = (today - timedelta(days=policy.text_days)).isoformat()
    cand_cutoff = (today - timedelta(days=policy.candidate_days)).isoformat()
    session_cutoff = int(time.mktime((today - timedelta(days=policy.session_days)).timetuple()))
    out: dict[str, int] = {}
    with get_session(db_path) as session:
        old_runs = "SELECT id FROM runs WHERE run_at < :cutoff"
        for table, col in _PROSE_COLUMNS:
            n = session.exec(
                text(
                    f"UPDATE {table} SET {col} = NULL WHERE {col} IS NOT NULL AND {col} != '' "
                    f"AND run_id IN ({old_runs})"
                ),
                params={"cutoff": text_cutoff},
            ).rowcount
            out["prose_fields"] = out.get("prose_fields", 0) + (n or 0)
        out["failed_candidates"] = (
            session.exec(
                text(
                    f"DELETE FROM candidates WHERE passed_filter = 0 AND run_id IN ({old_runs}) "
                    "AND NOT EXISTS (SELECT 1 FROM picks p WHERE p.run_id = candidates.run_id "
                    "AND p.ticker = candidates.ticker)"
                ),
                params={"cutoff": cand_cutoff},
            ).rowcount
            or 0
        )
        tables = {
            r[0] for r in session.exec(text("SELECT name FROM sqlite_master WHERE type = 'table'"))
        }
        for table in ("workflow_session_runs", "workflow_session"):
            if table in tables:
                out[table] = (
                    session.exec(
                        text(f"DELETE FROM {table} WHERE created_at < :cutoff"),
                        params={"cutoff": session_cutoff},
                    ).rowcount
                    or 0
                )
        out["model_versions"] = (
            session.exec(
                text(
                    "DELETE FROM model_versions WHERE accepted = 0 AND id NOT IN "
                    "(SELECT id FROM model_versions ORDER BY id DESC LIMIT :keep)"
                ),
                params={"keep": policy.keep_models},
            ).rowcount
            or 0
        )
    return out


# Files this app writes, per directory; nothing else in them is touched.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_file_targets(cache_dir: str) -> list[tuple[Path, str]]:
    targets = [
        (
            Path(os.path.expanduser(os.getenv("LOG_DIR", "~/.stock_analyzer/logs"))),
            "stock-analyzer-*.log",
        ),
        (Path(os.path.expanduser(cache_dir)), "price_panel_*.pkl"),
    ]
    # Cron wrapper logs (scripts/run_*.sh), only if this is the project checkout.
    if (PROJECT_ROOT / "pyproject.toml").exists():
        targets += [
            (PROJECT_ROOT / "logs", pattern)
            for pattern in ("portfolio_*.log", "insiders_*.log", "model_review_*.log")
        ]
    return targets


def prune_files(targets: list[tuple[Path, str]], days: int) -> int:
    """Delete files matching each (directory, pattern) not modified in `days` days."""
    cutoff = time.time() - days * 86400
    removed = 0
    for root, pattern in targets:
        if not root.is_dir():
            continue
        for f in root.glob(pattern):
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
    return removed


def run_history_upkeep(
    db_path: str,
    *,
    policy: RetentionPolicy,
    file_targets: list[tuple[Path, str]],
    today: date | None = None,
    label_outcomes: bool = True,
) -> UpkeepReport:
    today = today or date.today()
    report = UpkeepReport()

    def attempt(name: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — upkeep must never fail a run
            report.errors.append(f"{name}: {e}")
            logger.warning("History upkeep step %s failed: %s", name, e)

    def backfill() -> None:
        from .backfill import backfill_pick_forecasts

        report.added.update({f"pick_{k}": v for k, v in backfill_pick_forecasts(db_path).items()})

    def labels() -> None:
        from ..model.labels import label_candidates

        report.added["survivor_outcomes"] = label_candidates(db_path, only_passed=True)

    def database() -> None:
        report.trimmed.update(prune_database(db_path, policy, today=today))

    def files() -> None:
        report.trimmed["old_files"] = prune_files(file_targets, policy.file_days)

    attempt("backfill", backfill)
    if label_outcomes:
        attempt("labels", labels)
    attempt("database", database)
    attempt("files", files)
    logger.info(report.summary())
    return report
