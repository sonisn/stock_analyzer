"""Backfill realized forward returns for every screened candidate in the DB.

Each (run, ticker) gets one `candidate_outcomes` row per horizon once the
window has closed: entry at the first close after the run (the pipeline
runs intraday), exit `horizon` trading days later, minus SPY over the same
bars — the same label definition the training set uses, so the stored
candidates can be scored with the same yardstick as the backtest.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime

import numpy as np
import polars as pl
from sqlalchemy import text

from ..data.frames import DATE
from ..db.session import exec_sql, get_session
from ..db.tables import CandidateOutcome
from ..logging import get_logger
from .dataset import HORIZONS, PricePanel, download_panel

logger = get_logger(__name__)

_PENDING_SCHEMA = {
    "run_id": pl.Int64,
    "ticker": pl.String,
    "run_date": pl.Date,
    "horizon": pl.Int64,
}


def pending_labels(
    db_path: str, *, only_passed: bool = False, only_picks: bool = False
) -> pl.DataFrame:
    """(run_id, ticker, run_date, horizon) rows that have no outcome yet;
    `only_passed` limits it to screen survivors (the per-run upkeep),
    `only_picks` to the top picks (the daily email's scorecard)."""
    if only_picks:
        sql = "SELECT p.run_id, p.ticker, r.run_at FROM picks p JOIN runs r ON r.id = p.run_id"
    else:
        where = " WHERE c.passed_filter = 1" if only_passed else ""
        sql = "SELECT c.run_id, c.ticker, r.run_at FROM candidates c JOIN runs r ON r.id = c.run_id"
        sql += where
    with get_session(db_path) as session:
        rows = exec_sql(session, text(sql)).all()
        done = set(
            exec_sql(
                session, text("SELECT run_id, ticker, horizon_days FROM candidate_outcomes")
            ).all()
        )
    out = [
        (run_id, ticker, datetime.fromisoformat(str(run_at)).date(), h)
        for run_id, ticker, run_at in rows
        for h in HORIZONS
        if (run_id, ticker, h) not in done
    ]
    return pl.DataFrame(out, schema=_PENDING_SCHEMA, orient="row")


def label_candidates(
    db_path: str,
    *,
    fetch_panel: Callable[[list[str]], PricePanel] | None = None,
    only_passed: bool = False,
    only_picks: bool = False,
) -> int:
    """Write every outcome whose window has closed; returns rows written."""
    pending = pending_labels(db_path, only_passed=only_passed, only_picks=only_picks)
    # A window needs 1 + horizon trading days; ~7/5 calendar days each plus
    # a holiday margin. Skip rows that cannot have closed so a routine run
    # doesn't download prices for hundreds of names it can't label yet.
    if not pending.is_empty():
        age = (pl.lit(date.today()) - pl.col("run_date")).dt.total_days()
        pending = pending.filter(age >= (pl.col("horizon") + 1) * 7 / 5 + 3)
    if pending.is_empty():
        return 0
    tickers = sorted(pending["ticker"].unique().to_list())
    panel = (fetch_panel or (lambda t: download_panel(t, years=2)))(tickers)
    spy_frame = panel.spy.filter(pl.col("SPY").is_not_null() & pl.col("SPY").is_not_nan())
    cal = spy_frame[DATE].to_numpy().astype("datetime64[D]")
    close = spy_frame.select(DATE).join(panel.close, on=DATE, how="left")
    spy = spy_frame["SPY"].to_numpy()

    rows: list[CandidateOutcome] = []
    for rec in pending.iter_rows(named=True):
        if rec["ticker"] not in close.columns:
            continue
        # First bar strictly after the run date, then `horizon` bars on.
        entry_pos = int(np.searchsorted(cal, np.datetime64(rec["run_date"]), side="right"))
        exit_pos = entry_pos + int(rec["horizon"])
        if exit_pos >= len(cal):
            continue  # window still open
        px = close[rec["ticker"]]
        px_in, px_out = px[entry_pos], px[exit_pos]
        spy_in, spy_out = spy[entry_pos], spy[exit_pos]
        values = (px_in, px_out, spy_in, spy_out)
        if any(v is None or np.isnan(v) or v <= 0 for v in values):
            continue
        ret = (px_out / px_in - 1) * 100
        spy_ret = (spy_out / spy_in - 1) * 100
        rows.append(
            CandidateOutcome(
                run_id=int(rec["run_id"]),
                ticker=rec["ticker"],
                horizon_days=int(rec["horizon"]),
                entry_date=str(cal[entry_pos]),
                exit_date=str(cal[exit_pos]),
                return_pct=float(ret),
                spy_return_pct=float(spy_ret),
                excess_pct=float(ret - spy_ret),
            )
        )
    with get_session(db_path) as session:
        for row in rows:
            session.add(row)
    logger.info("Labeled %d candidate outcomes (%d pending)", len(rows), pending.height)
    return len(rows)


def grade_shadow_scores(db_path: str, horizon: int = 21) -> dict[str, float | int | None]:
    """Live, truly out-of-sample check of the model: the percentile each
    run recorded for its survivors (score_breakdown["model"]) against the
    excess return those names then realized. Per-run Spearman IC, averaged."""
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT c.run_id, c.score_breakdown, o.excess_pct FROM candidates c "
                "JOIN candidate_outcomes o ON o.run_id = c.run_id AND o.ticker = c.ticker "
                "WHERE o.horizon_days = :h AND c.score_breakdown LIKE '%\"model\"%'"
            ),
            params={"h": horizon},
        ).all()
    frame = pl.DataFrame(
        [
            (run_id, (json.loads(bd).get("model") or {}).get("percentile"), excess)
            for run_id, bd, excess in rows
        ],
        schema={"run_id": pl.Int64, "pct": pl.Float64, "excess": pl.Float64},
        orient="row",
    ).drop_nulls()
    ics = []
    for _, g in frame.group_by("run_id"):
        if g.height >= 5:
            ic = g.select(pl.corr(pl.col("pct").rank(), pl.col("excess").rank())).item()
            if ic is not None and not np.isnan(ic):
                ics.append(ic)
    return {
        "runs": len(ics),
        "names": frame.height,
        "mean_ic": float(np.mean(ics)) if ics else None,
        "hit_rate": float(np.mean([ic > 0 for ic in ics])) if ics else None,
    }
