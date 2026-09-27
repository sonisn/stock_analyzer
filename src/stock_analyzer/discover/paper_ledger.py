"""Paper-trading ledger: what following every run's picks would have earned.

The track record grades picks one at a time. This answers the portfolio
question: if each day the pipeline produced picks you had invested a fixed
tranche ($1,000) across them at the Sizer's weights and held, how does that
book compare with putting the same tranche into SPY on the same day?

  - One tranche per calendar day with picks (the day's last run wins —
    same-day reruns are retries, not new advice).
  - Weights are parsed from the stored Sizer text ("Allocation: 27% ..." or
    a dollar amount — normalized either way); equal weight when unparseable.
  - Entry is the first close on/after the run date; prices are dividend-
    adjusted. A pick with no price history is dropped from its tranche and
    the rest renormalized.
  - No trading costs, taxes or trims — an upper bound on "follow the
    email", not a replica of the real account.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from ..data import frames
from ..data.frames import DATE
from ..db.session import exec_sql, get_session
from ..logging import get_logger
from .report_sections import _split_by_ticker_blocks
from .track_record import _close_on_or_after, _fetch_history

logger = get_logger(__name__)

TRANCHE_USD = 1000.0
_MAX_CURVE_POINTS = 120
_ALLOC_RE = re.compile(r"Allocation:\s*\$?\s*([\d,]+(?:\.\d+)?)")


@dataclass(frozen=True)
class Tranche:
    run_id: int
    run_date: date
    weights: dict[str, float]


@dataclass(frozen=True)
class TrancheResult:
    run_date: date
    tickers: list[str]
    strategy_return_pct: float
    spy_return_pct: float


@dataclass
class LedgerReport:
    tranches: list[TrancheResult] = field(default_factory=list)
    curve_dates: list[date] = field(default_factory=list)
    strategy_values: list[float] = field(default_factory=list)
    spy_values: list[float] = field(default_factory=list)
    invested_values: list[float] = field(default_factory=list)
    skipped_tickers: list[str] = field(default_factory=list)

    @property
    def invested(self) -> float:
        return self.invested_values[-1] if self.invested_values else 0.0

    def _ret(self, values: list[float]) -> float | None:
        if not values or not self.invested:
            return None
        return (values[-1] / self.invested - 1) * 100

    @property
    def strategy_return_pct(self) -> float | None:
        return self._ret(self.strategy_values)

    @property
    def spy_return_pct(self) -> float | None:
        return self._ret(self.spy_values)


def parse_weights(sizer_text: str, tickers: list[str]) -> dict[str, float]:
    blocks = _split_by_ticker_blocks(sizer_text or "")
    raw: dict[str, float] = {}
    for t in tickers:
        m = _ALLOC_RE.search(blocks.get(t, ""))
        if m:
            raw[t] = float(m.group(1).replace(",", ""))
    total = sum(raw.values())
    if len(raw) != len(tickers) or total <= 0:
        return {t: 1 / len(tickers) for t in tickers} if tickers else {}
    return {t: v / total for t, v in raw.items()}


def load_tranches(db_path: str) -> list[Tranche]:
    from sqlalchemy import text

    with get_session(db_path) as session:
        runs = exec_sql(
            session,
            text(
                "SELECT r.id, r.run_at, o.sizer_full FROM runs r "
                "LEFT JOIN run_outputs o ON o.run_id = r.id "
                "WHERE r.picks > 0 ORDER BY r.id ASC"
            ),
        ).all()
        picks = exec_sql(
            session, text("SELECT run_id, ticker FROM picks ORDER BY run_id, rank")
        ).all()
    tickers_by_run: dict[int, list[str]] = {}
    for run_id, ticker in picks:
        tickers_by_run.setdefault(run_id, []).append(ticker)
    by_day: dict[date, Tranche] = {}
    for run_id, run_at, sizer_full in runs:
        tickers = tickers_by_run.get(run_id)
        if not tickers:
            continue
        day = datetime.fromisoformat(run_at).date()
        by_day[day] = Tranche(run_id, day, parse_weights(sizer_full or "", tickers))
    return [by_day[d] for d in sorted(by_day)]


def build_ledger(
    tranches: list[Tranche],
    *,
    today: date | None = None,
    fetch: Callable[[str, date, date], Any] = _fetch_history,
) -> LedgerReport:
    report = LedgerReport()
    if not tranches:
        return report
    today = today or date.today()
    start = tranches[0].run_date - timedelta(days=5)

    spy_closes = frames.closes(fetch("SPY", start, today))
    if spy_closes is None:
        logger.warning("Paper ledger: no SPY history — skipping")
        return report
    calendar = spy_closes[DATE]
    days = calendar.to_numpy().astype("datetime64[D]")
    spy_px = spy_closes["Close"].to_numpy()

    closes: dict[str, np.ndarray] = {}
    for ticker in sorted({t for tr in tranches for t in tr.weights}):
        own = frames.closes(fetch(ticker, start, today))
        if own is None:
            report.skipped_tickers.append(ticker)
            continue
        # Each trading day's latest close at or before it (forward-filled).
        filled = pl.DataFrame({DATE: calendar}).join_asof(own.sort(DATE), on=DATE)
        closes[ticker] = filled["Close"].to_numpy().astype(float)

    n = len(days)
    strategy, benchmark, invested = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr in tranches:
        spy_entry = _close_on_or_after(spy_closes, tr.run_date)
        if spy_entry is None:
            continue  # run is newer than the last available close
        entry_day = np.datetime64(spy_entry[1])
        weights = {t: w for t, w in tr.weights.items() if t in closes}
        total_w = sum(weights.values())
        if not weights or total_w <= 0:
            continue
        live = days >= entry_day
        at_entry = int(np.searchsorted(days, entry_day))
        tranche_value = np.zeros(n)
        for t, w in weights.items():
            entry_px = closes[t][at_entry]
            if np.isnan(entry_px) or entry_px <= 0:
                continue
            shares = TRANCHE_USD * (w / total_w) / entry_px
            tranche_value += np.where(live, np.nan_to_num(closes[t] * shares), 0.0)
        if tranche_value[-1] <= 0:
            continue
        spy_value = np.where(live, spy_px * (TRANCHE_USD / spy_entry[0]), 0.0)
        strategy += tranche_value
        benchmark += spy_value
        invested += np.where(live, TRANCHE_USD, 0.0)
        report.tranches.append(
            TrancheResult(
                run_date=tr.run_date,
                tickers=sorted(weights),
                strategy_return_pct=(tranche_value[-1] / TRANCHE_USD - 1) * 100,
                spy_return_pct=(spy_value[-1] / TRANCHE_USD - 1) * 100,
            )
        )

    active = invested > 0
    if not active.any():
        return report
    strategy, benchmark, invested = strategy[active], benchmark[active], invested[active]
    active_days = days[active]
    step = max(1, math.ceil(len(strategy) / _MAX_CURVE_POINTS))
    keep = list(range(0, len(strategy), step))
    if keep[-1] != len(strategy) - 1:
        keep.append(len(strategy) - 1)  # always end on the latest close
    report.curve_dates = [active_days[i].astype(object) for i in keep]
    report.strategy_values = [float(strategy[i]) for i in keep]
    report.spy_values = [float(benchmark[i]) for i in keep]
    report.invested_values = [float(invested[i]) for i in keep]
    return report


def ledger_report_data(report: LedgerReport) -> dict[str, Any] | None:
    """Section-data shape for the `equity_curve` report section."""
    if not report.curve_dates:
        return None
    return {
        "dates": [d.isoformat() for d in report.curve_dates],
        "strategy": report.strategy_values,
        "benchmark": report.spy_values,
        "invested": report.invested_values,
        "strategy_return_pct": report.strategy_return_pct,
        "benchmark_return_pct": report.spy_return_pct,
        "tranches": [
            {
                "run_date": t.run_date.isoformat(),
                "tickers": t.tickers,
                "strategy_return_pct": t.strategy_return_pct,
                "benchmark_return_pct": t.spy_return_pct,
            }
            for t in report.tranches
        ],
        "skipped_tickers": report.skipped_tickers,
    }
