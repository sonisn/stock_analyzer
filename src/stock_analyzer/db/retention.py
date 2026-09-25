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
  - stored long-term views (`stock_views`) not refreshed in `stock_view_days`:
    a stock that is no longer held stops being refreshed, and a view older
    than that is rewritten on sight anyway;
  - model versions beyond the newest `keep_models` (accepted ones are kept);
  - log files and cached price panels untouched for `file_days`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ..logging import get_logger
from .session import exec_sql, get_session

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
    reference_days: int = 365  # ticker_reference rows untouched this long go
    stock_view_days: int = 90  # stored long-term views not refreshed this long go
    vacuum_min_free_pct: float = 20.0  # compact the file once this much is free
    warn_mb: float = 50.0  # flag the database in the upkeep summary past this


@dataclass
class UpkeepReport:
    added: dict[str, int] = field(default_factory=dict)
    trimmed: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    size: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        def fmt(d: dict[str, int]) -> str:
            return ", ".join(f"{k}={v}" for k, v in d.items() if v) or "nothing"

        text_ = f"History upkeep — added: {fmt(self.added)}; trimmed: {fmt(self.trimmed)}"
        if self.size:
            text_ += f"\n{size_line(self.size)}"
        return text_ + (f"; errors: {'; '.join(self.errors)}" if self.errors else "")


def prune_database(db_path: str, policy: RetentionPolicy, *, today: date) -> dict[str, int]:
    text_cutoff = (today - timedelta(days=policy.text_days)).isoformat()
    cand_cutoff = (today - timedelta(days=policy.candidate_days)).isoformat()
    session_cutoff = int(time.mktime((today - timedelta(days=policy.session_days)).timetuple()))
    out: dict[str, int] = {}
    with get_session(db_path) as session:
        old_runs = "SELECT id FROM runs WHERE run_at < :cutoff"
        for table, col in _PROSE_COLUMNS:
            n = exec_sql(
                session,
                text(
                    f"UPDATE {table} SET {col} = NULL WHERE {col} IS NOT NULL AND {col} != '' "
                    f"AND run_id IN ({old_runs})"
                ),
                params={"cutoff": text_cutoff},
            ).rowcount
            out["prose_fields"] = out.get("prose_fields", 0) + (n or 0)
        out["failed_candidates"] = (
            exec_sql(
                session,
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
            r[0]
            for r in exec_sql(session, text("SELECT name FROM sqlite_master WHERE type = 'table'"))
        }
        for table in ("workflow_session_runs", "workflow_session"):
            if table in tables:
                out[table] = (
                    exec_sql(
                        session,
                        text(f"DELETE FROM {table} WHERE created_at < :cutoff"),
                        params={"cutoff": session_cutoff},
                    ).rowcount
                    or 0
                )
        if "stock_views" in tables:
            # Views of stocks no longer held stop being refreshed; drop them.
            out["stale_stock_views"] = (
                exec_sql(
                    session,
                    text("DELETE FROM stock_views WHERE written_on < :c"),
                    params={"c": (today - timedelta(days=policy.stock_view_days)).isoformat()},
                ).rowcount
                or 0
            )
        if "ticker_reference" in tables:
            ref_cutoff = (today - timedelta(days=policy.reference_days)).isoformat()
            out["stale_reference_rows"] = (
                exec_sql(
                    session,
                    text(
                        "DELETE FROM ticker_reference WHERE "
                        "COALESCE(profile_updated, '') < :c AND COALESCE(earnings_updated, '') < :c"
                    ),
                    params={"c": ref_cutoff},
                ).rowcount
                or 0
            )
        out["model_versions"] = (
            exec_sql(
                session,
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
    # Cron wrapper logs (scripts/run_job.sh), only if this is the project checkout.
    if (PROJECT_ROOT / "pyproject.toml").exists():
        targets += [
            (PROJECT_ROOT / "logs", pattern)
            # <job>_YYYYMMDD.log, one per job per day (scripts/run_job.sh)
            for pattern in ("*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].log",)
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

    def compact_and_measure() -> None:
        freed = compact_if_worth_it(db_path, min_free_pct=policy.vacuum_min_free_pct)
        if freed:
            report.trimmed["vacuum_kb"] = int(freed / 1024)
        report.size = database_size(db_path, warn_mb=policy.warn_mb)

    attempt("backfill", backfill)
    if label_outcomes:
        attempt("labels", labels)
    attempt("database", database)
    attempt("files", files)
    attempt("compact", compact_and_measure)
    logger.info(report.summary())
    return report


def _sqlite_path(db_path: str) -> str:
    return os.path.expanduser(db_path)


def compact_if_worth_it(db_path: str, *, min_free_pct: float) -> int:
    """VACUUM when at least `min_free_pct` of the file is free pages (space
    left by trimming); returns bytes reclaimed. Skipped otherwise — VACUUM
    rewrites the whole file, so it isn't worth doing for a few pages."""
    import sqlite3

    path = _sqlite_path(db_path)
    if not os.path.exists(path):
        return 0
    before = os.path.getsize(path)
    con = sqlite3.connect(path)
    try:
        pages = con.execute("PRAGMA page_count").fetchone()[0] or 0
        free = con.execute("PRAGMA freelist_count").fetchone()[0] or 0
        if not pages or free / pages * 100 < min_free_pct:
            return 0
        con.execute("VACUUM")
    finally:
        con.close()
    return max(before - os.path.getsize(path), 0)


def database_size(db_path: str, *, warn_mb: float) -> dict[str, Any]:
    """File size and the largest tables (bytes via dbstat when SQLite has
    it, else row counts), with a warning flag past `warn_mb`."""
    import sqlite3

    path = _sqlite_path(db_path)
    if not os.path.exists(path):
        return {}
    mb = os.path.getsize(path) / 1e6
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            top = [
                (name, int(size))
                for name, size in con.execute(
                    "SELECT name, SUM(pgsize) FROM dbstat WHERE name NOT LIKE 'sqlite_%' "
                    "GROUP BY name ORDER BY 2 DESC LIMIT 5"
                )
            ]
            unit = "bytes"
        except sqlite3.OperationalError:
            names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            counts = [(n, con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]) for n in names]
            top = sorted(counts, key=lambda kv: -kv[1])[:5]
            unit = "rows"
    finally:
        con.close()
    return {"mb": mb, "warn_mb": warn_mb, "over": mb > warn_mb, "top": top, "unit": unit}


def size_line(size: dict[str, Any]) -> str:
    def fmt(v: int) -> str:
        return f"{v / 1024:.0f} KB" if size["unit"] == "bytes" else f"{v} rows"

    tops = ", ".join(f"{n} {fmt(v)}" for n, v in size["top"])
    line = f"Database: {size['mb']:.1f} MB (largest: {tops})"
    if size["over"]:
        line += (
            f" — OVER the {size['warn_mb']:.0f} MB guide: lower HISTORY_TEXT_RETENTION_DAYS "
            "or check which table grew"
        )
    return line
