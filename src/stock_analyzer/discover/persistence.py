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

CREATE TABLE IF NOT EXISTS run_outputs (
    run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    ranker_full TEXT,
    redteam_full TEXT,
    sizer_full TEXT,
    holdings_summary TEXT
);
"""


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
) -> int:
    cur = conn.execute(
        "INSERT INTO runs (run_at, universe_size, survivors, picks, "
        "opus_model, sonnet_model, cash_budget) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(timespec="seconds"),
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
) -> None:
    conn.execute(
        "INSERT INTO run_outputs (run_id, ranker_full, redteam_full, "
        "sizer_full, holdings_summary) VALUES (?, ?, ?, ?, ?)",
        (run_id, ranker_full, redteam_full, sizer_full, holdings_summary),
    )
