"""SQLite persistence for discovery runs.

Stores every run + every candidate + every pick. Six months from now you'll
want to ask "what did we recommend on May 11 and was it right?" — these
tables are the answer.

Schema is deliberately small. Add columns as needs emerge; don't over-design.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    run_at TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'discover',  -- 'discover' | 'rebalance'
    universe_size INTEGER NOT NULL,
    survivors INTEGER NOT NULL,
    picks INTEGER NOT NULL,
    opus_model TEXT,
    sonnet_model TEXT,
    cash_budget REAL
);

CREATE TABLE IF NOT EXISTS candidates (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    passed_filter INTEGER NOT NULL,
    fail_reasons TEXT,           -- JSON list
    score REAL,                  -- 0-100, null if filter failed
    score_components TEXT,       -- JSON {fundamentals, trend, conviction}
    score_breakdown TEXT,        -- JSON full breakdown
    sources TEXT,                -- JSON list of universe sources
    conviction INTEGER,          -- universe conviction count
    sector TEXT,
    price REAL,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS scorecards (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    analyst_text TEXT,           -- the Sonnet output for this ticker
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS picks (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    ranker_text TEXT NOT NULL,     -- full Opus block from ranker
    bear_case_text TEXT,           -- full Opus block from red-team
    allocation_text TEXT,          -- single-pick excerpt from sizer
    PRIMARY KEY (run_id, rank)
);

CREATE TABLE IF NOT EXISTS holdings_reviews (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    verdict TEXT,           -- HOLD / TRIM / SELL (parsed from review_text)
    confidence INTEGER,     -- 1-10 (parsed from review_text)
    review_text TEXT,       -- full Sonnet review for this holding
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS run_outputs (
    run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    ranker_full TEXT,
    redteam_full TEXT,
    sizer_full TEXT,
    holdings_summary TEXT,
    rebalance_text TEXT,     -- full Opus rebalance plan (rebalance kind only)
    dashboard_data TEXT      -- JSON snapshot for the dashboard (holdings,
                             -- metrics, sector, status, pdf_path, etc.)
);
"""

# Columns added after the initial schema — apply via ALTER TABLE so older
# DBs migrate forward without losing data.
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("runs", "ALTER TABLE runs ADD COLUMN kind TEXT NOT NULL DEFAULT 'discover'"),
    ("run_outputs", "ALTER TABLE run_outputs ADD COLUMN rebalance_text TEXT"),
    ("run_outputs", "ALTER TABLE run_outputs ADD COLUMN dashboard_data TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply additive column migrations; ignore 'duplicate column' errors so
    the function is idempotent on already-migrated DBs."""
    for _table, ddl in _MIGRATIONS:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise


def _expanded_path(path: str) -> Path:
    return Path(os.path.expanduser(path))


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    p = _expanded_path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_run(
    conn: sqlite3.Connection,
    *,
    universe_size: int,
    survivors: int,
    picks: int,
    opus_model: str,
    sonnet_model: str,
    cash_budget: float | None,
    kind: str = "discover",
) -> int:
    cur = conn.execute(
        "INSERT INTO runs (run_at, kind, universe_size, survivors, picks, "
        "opus_model, sonnet_model, cash_budget) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(timespec="seconds"),
            kind,
            universe_size,
            survivors,
            picks,
            opus_model,
            sonnet_model,
            cash_budget,
        ),
    )
    return int(cur.lastrowid or 0)


def insert_candidate(
    conn: sqlite3.Connection,
    run_id: int,
    ticker: str,
    *,
    passed_filter: bool,
    fail_reasons: list[str],
    score: float | None,
    score_components: dict[str, Any] | None,
    score_breakdown: dict[str, Any] | None,
    sources: list[str],
    conviction: int,
    sector: str | None,
    price: float | None,
) -> None:
    conn.execute(
        "INSERT INTO candidates (run_id, ticker, passed_filter, fail_reasons, "
        "score, score_components, score_breakdown, sources, conviction, "
        "sector, price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            ticker,
            int(passed_filter),
            json.dumps(fail_reasons),
            score,
            json.dumps(score_components) if score_components else None,
            json.dumps(score_breakdown) if score_breakdown else None,
            json.dumps(sources),
            conviction,
            sector,
            price,
        ),
    )


def insert_scorecard(
    conn: sqlite3.Connection, run_id: int, ticker: str, text: str
) -> None:
    conn.execute(
        "INSERT INTO scorecards (run_id, ticker, analyst_text) VALUES (?, ?, ?)",
        (run_id, ticker, text),
    )


def insert_holdings_review(
    conn: sqlite3.Connection,
    run_id: int,
    ticker: str,
    *,
    verdict: str | None,
    confidence: int | None,
    review_text: str,
) -> None:
    conn.execute(
        "INSERT INTO holdings_reviews (run_id, ticker, verdict, "
        "confidence, review_text) VALUES (?, ?, ?, ?, ?)",
        (run_id, ticker, verdict, confidence, review_text),
    )


def fetch_recent_holdings_history(
    conn: sqlite3.Connection, *, n_runs: int = 3, kind: str = "rebalance"
) -> dict[str, list[dict[str, Any]]]:
    """Return {ticker: [{run_at, verdict, confidence}, ...]} oldest-first
    for the last `n_runs` rebalance runs. Used to build the cross-run
    'Previous decisions' block for the rebalancer prompt."""
    cur = conn.execute(
        "SELECT id, run_at FROM runs WHERE kind = ? "
        "ORDER BY id DESC LIMIT ?",
        (kind, n_runs),
    )
    runs = list(cur.fetchall())
    if not runs:
        return {}
    runs.reverse()  # oldest first so the LLM sees chronological drift
    out: dict[str, list[dict[str, Any]]] = {}
    for run_id, run_at in runs:
        cur = conn.execute(
            "SELECT ticker, verdict, confidence FROM holdings_reviews "
            "WHERE run_id = ?",
            (run_id,),
        )
        for ticker, verdict, confidence in cur.fetchall():
            out.setdefault(ticker, []).append({
                "run_at": run_at,
                "verdict": verdict,
                "confidence": confidence,
            })
    return out


def insert_pick(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    rank: int,
    ticker: str,
    ranker_text: str,
    bear_case_text: str | None,
    allocation_text: str | None,
) -> None:
    conn.execute(
        "INSERT INTO picks (run_id, rank, ticker, ranker_text, "
        "bear_case_text, allocation_text) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, rank, ticker, ranker_text, bear_case_text, allocation_text),
    )


def insert_run_outputs(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    ranker_full: str,
    redteam_full: str,
    sizer_full: str,
    holdings_summary: str,
    rebalance_text: str | None = None,
    dashboard_data: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        "INSERT INTO run_outputs (run_id, ranker_full, redteam_full, "
        "sizer_full, holdings_summary, rebalance_text, dashboard_data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            ranker_full,
            redteam_full,
            sizer_full,
            holdings_summary,
            rebalance_text,
            json.dumps(dashboard_data) if dashboard_data is not None else None,
        ),
    )
