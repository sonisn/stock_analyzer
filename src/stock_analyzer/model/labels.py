"""Backfill realized forward returns for every screened candidate in the DB.

Each (run, ticker) gets one `candidate_outcomes` row per horizon once the
window has closed: entry at the first close after the run (the pipeline
runs intraday), exit `horizon` trading days later, minus SPY over the same
bars — the same label definition the training set uses, so the stored
candidates can be scored with the same yardstick as the backtest.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from ..db.session import get_session
from ..db.tables import CandidateOutcome
from ..logging import get_logger
from .dataset import HORIZONS, PricePanel, download_panel

logger = get_logger(__name__)


def pending_labels(db_path: str) -> pd.DataFrame:
    """(run_id, ticker, run_date, horizon) rows that have no outcome yet."""
    with get_session(db_path) as session:
        rows = session.exec(
            text(
                "SELECT c.run_id, c.ticker, r.run_at FROM candidates c JOIN runs r ON r.id = c.run_id"
            )
        ).all()
        done = set(
            session.exec(text("SELECT run_id, ticker, horizon_days FROM candidate_outcomes")).all()
        )
    out = [
        (run_id, ticker, datetime.fromisoformat(str(run_at)).date(), h)
        for run_id, ticker, run_at in rows
        for h in HORIZONS
        if (run_id, ticker, h) not in done
    ]
    return pd.DataFrame(out, columns=["run_id", "ticker", "run_date", "horizon"])


def label_candidates(
    db_path: str,
    *,
    fetch_panel: Callable[[list[str]], PricePanel] | None = None,
) -> int:
    """Write every outcome whose window has closed; returns rows written."""
    pending = pending_labels(db_path)
    # A window needs 1 + horizon trading days; ~7/5 calendar days each plus
    # a holiday margin. Skip rows that cannot have closed so a routine run
    # doesn't download prices for hundreds of names it can't label yet.
    if not pending.empty:
        cutoff = pd.Timestamp.today().normalize()
        age = (cutoff - pd.to_datetime(pending["run_date"])).dt.days
        pending = pending[age >= (pending["horizon"] + 1) * 7 / 5 + 3]
    if pending.empty:
        return 0
    tickers = sorted(pending["ticker"].unique())
    panel = (fetch_panel or (lambda t: download_panel(t, years=2)))(tickers)
    cal = panel.spy.dropna().index
    close = panel.close.reindex(cal)
    spy = panel.spy.reindex(cal)

    rows: list[CandidateOutcome] = []
    for rec in pending.itertuples(index=False):
        if rec.ticker not in close:
            continue
        # First bar strictly after the run date, then `horizon` bars on.
        entry_pos = int(cal.searchsorted(pd.Timestamp(rec.run_date), side="right"))
        exit_pos = entry_pos + int(rec.horizon)
        if exit_pos >= len(cal):
            continue  # window still open
        px_in, px_out = close[rec.ticker].iloc[entry_pos], close[rec.ticker].iloc[exit_pos]
        spy_in, spy_out = spy.iloc[entry_pos], spy.iloc[exit_pos]
        if any(np.isnan(v) or v <= 0 for v in (px_in, px_out, spy_in, spy_out)):
            continue
        ret = (px_out / px_in - 1) * 100
        spy_ret = (spy_out / spy_in - 1) * 100
        rows.append(
            CandidateOutcome(
                run_id=int(rec.run_id),
                ticker=rec.ticker,
                horizon_days=int(rec.horizon),
                entry_date=str(cal[entry_pos].date()),
                exit_date=str(cal[exit_pos].date()),
                return_pct=float(ret),
                spy_return_pct=float(spy_ret),
                excess_pct=float(ret - spy_ret),
            )
        )
    with get_session(db_path) as session:
        for row in rows:
            session.add(row)
    logger.info("Labeled %d candidate outcomes (%d pending)", len(rows), len(pending))
    return len(rows)


def grade_shadow_scores(db_path: str, horizon: int = 21) -> dict[str, float | int | None]:
    """Live, truly out-of-sample check of the model: the percentile each
    run recorded for its survivors (score_breakdown["model"]) against the
    excess return those names then realized. Per-run Spearman IC, averaged."""
    import json

    with get_session(db_path) as session:
        rows = session.exec(
            text(
                "SELECT c.run_id, c.score_breakdown, o.excess_pct FROM candidates c "
                "JOIN candidate_outcomes o ON o.run_id = c.run_id AND o.ticker = c.ticker "
                "WHERE o.horizon_days = :h AND c.score_breakdown LIKE '%\"model\"%'"
            ),
            params={"h": horizon},
        ).all()
    frame = pd.DataFrame(
        [
            (run_id, (json.loads(bd).get("model") or {}).get("percentile"), excess)
            for run_id, bd, excess in rows
        ],
        columns=["run_id", "pct", "excess"],
    ).dropna()
    ics = [
        g["pct"].rank().corr(g["excess"].rank()) for _, g in frame.groupby("run_id") if len(g) >= 5
    ]
    ics = [ic for ic in ics if pd.notna(ic)]
    return {
        "runs": len(ics),
        "names": int(len(frame)),
        "mean_ic": float(np.mean(ics)) if ics else None,
        "hit_rate": float(np.mean([ic > 0 for ic in ics])) if ics else None,
    }
