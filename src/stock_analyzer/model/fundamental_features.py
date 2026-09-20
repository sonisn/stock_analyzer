"""Fundamentals as they were known on the day, for the training set.

`features.py` is price-only, and its docstring says why: a historical
close is the same number today as it was then, while yfinance's
fundamentals are a live snapshot with no history. Training on today's
margins to predict 2023's returns is look-ahead bias — the model learns
from figures that had not been filed yet.

Wisesheets' `asof:` selector with `asReported=true` fixes exactly that.
The check that matters: with as-reported off, `asof:2025-05-01` returns
NVDA's quarter ending 2025-04-27, which was not filed until 2025-05-28.
With it on, the same date returns the quarter ending 2025-01-26, filed
2025-02-26 — what an investor could actually have read.

Two consequences shape this module:

  - Only ratios are usable. In as-reported mode the newest filing on a
    date is a 10-Q for one company and a 10-K for another, so revenue
    means a quarter here and a year there. A margin taken from inside one
    filing is comparable across both.
  - Fundamentals are sampled monthly, not weekly. They change when a
    filing lands, roughly quarterly, so twelve as-of dates a year carry
    the whole signal at a twelfth of the request budget. Each as-of date
    costs one request per 100 tickers: the S&P 500 over five years is
    ~300 requests against a 5,000/month free plan.

Results are cached to disk per as-of date, so a retrain spends nothing.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from ..data import wisesheets
from ..logging import get_logger

logger = get_logger(__name__)

FUNDAMENTAL_FEATURES: tuple[str, ...] = wisesheets.POINT_IN_TIME_RATIOS

# Leave this much of the monthly quota unspent. A training run must never
# be the reason the daily email loses its filed fundamentals.
QUOTA_RESERVE = 200


def _cache_path(cache_dir: str, as_of: date) -> Path:
    return Path(cache_dir).expanduser() / "pit_fundamentals" / f"{as_of.isoformat()}.json"


def _load_cached(cache_dir: str, as_of: date) -> tuple[dict[str, dict[str, float]], set[str]]:
    """({ticker: ratios}, tickers already asked about) for one as-of date.

    The second half is what makes the cache safe to reuse across
    universes. Keyed on the date alone, a file written while testing 20
    tickers was served whole to a 500-ticker run, which then median-filled
    96% of its rows and measured nothing. `asked` records who was
    requested, so a later run fetches only the names it is missing — and
    a ticker Wisesheets does not cover stays cached as a known absence
    rather than being re-requested every time.
    """
    path = _cache_path(cache_dir, as_of)
    if not path.exists():
        return {}, set()
    try:
        body = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable point-in-time cache %s (%s)", path, e)
        return {}, set()
    if isinstance(body, dict) and "values" in body:
        values = body.get("values") or {}
        return values, set(body.get("tickers") or values)
    # Files written before the universe was recorded: only the tickers
    # they contain can be trusted as asked-about.
    values = body if isinstance(body, dict) else {}
    return values, set(values)


def _store(
    cache_dir: str, as_of: date, values: dict[str, dict[str, float]], asked: set[str]
) -> None:
    path = _cache_path(cache_dir, as_of)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tickers": sorted(asked), "values": values}))
    except OSError as e:
        logger.warning("Could not cache point-in-time fundamentals (%s)", e)


def month_ends(start: date, end: date) -> list[date]:
    """One as-of date per month over the span, month-end."""
    stamps = pd.date_range(start=start, end=end, freq="ME")
    return [d.date() for d in stamps]


def fetch_history(
    tickers: list[str],
    as_of_dates: list[date],
    *,
    cache_dir: str,
    quota_reserve: int = QUOTA_RESERVE,
) -> dict[date, dict[str, dict[str, float]]]:
    """{as_of: {ticker: ratios}} for every date, cache first.

    Stops early — with what it has — rather than spending the month's
    remaining quota down to zero.
    """
    wanted = sorted({t.upper() for t in tickers})
    out: dict[date, dict[str, dict[str, float]]] = {}
    todo: dict[date, list[str]] = {}
    for as_of in as_of_dates:
        cached, asked = _load_cached(cache_dir, as_of)
        out[as_of] = cached
        outstanding = [t for t in wanted if t not in asked]
        if outstanding:
            todo[as_of] = outstanding
    if not todo:
        return {d: v for d, v in out.items() if v}
    if not wisesheets.is_configured():
        logger.warning("WISESHEETS_API_KEY not set — no point-in-time fundamentals")
        return {d: v for d, v in out.items() if v}

    missing = sorted(todo)
    widest = max(len(v) for v in todo.values())
    chunks = max(1, -(-widest // wisesheets.AS_REPORTED_MAX_TICKERS))
    quota = wisesheets.quota() or {}
    remaining = quota.get("monthly_remaining")
    if remaining is not None:
        affordable = max(0, (remaining - quota_reserve)) // chunks
        if affordable < len(missing):
            logger.warning(
                "Quota allows %d more as-of dates (%d requests each, %s left this month) — "
                "fetching those and using the cache for the rest",
                affordable,
                chunks,
                remaining,
            )
            missing = missing[-affordable:] if affordable else []

    for as_of in missing:
        outstanding = todo[as_of]
        values, answered = wisesheets.fetch_point_in_time_ratios(outstanding, as_of)
        if not answered:
            # Nothing came back at all (a denied date, or an outage):
            # record nothing, so the next run retries these names.
            logger.info("Point-in-time fundamentals %s: nothing returned", as_of)
            continue
        merged = {**out.get(as_of, {}), **values}
        _, asked = _load_cached(cache_dir, as_of)
        # Only the names whose request was answered count as asked. A
        # chunk lost to a 503 must come back next run, or half a date
        # quietly becomes a cross-section of medians.
        _store(cache_dir, as_of, merged, asked | set(answered))
        out[as_of] = merged
        logger.info(
            "Point-in-time fundamentals %s: %d with data of %d answered (%d already cached)",
            as_of,
            len(values),
            len(answered),
            len(merged) - len(values),
        )
    return {d: v for d, v in out.items() if v}


def to_frame(history: dict[date, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Long frame indexed (date, ticker) with one column per ratio."""
    rows = []
    for as_of, by_ticker in sorted(history.items()):
        for ticker, ratios in by_ticker.items():
            rows.append({"date": pd.Timestamp(as_of), "ticker": ticker, **ratios})
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", *FUNDAMENTAL_FEATURES]).set_index(
            ["date", "ticker"]
        )
    frame = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    for column in FUNDAMENTAL_FEATURES:
        # float("nan"), not pd.NA: the column has to cast to float64, and
        # NAType does not. A ratio nobody reported is a missing number.
        if column not in frame:
            frame[column] = float("nan")
    return frame[list(FUNDAMENTAL_FEATURES)].astype("float64")


def align_to_dates(
    fundamentals: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]
) -> pd.DataFrame:
    """Carry each month's figures forward onto the model's weekly dates.

    A filing stays the latest known fact until the next one lands, so the
    value is held forward — never interpolated, and never pulled backward
    from a month that had not happened yet.
    """
    if fundamentals.empty:
        return pd.DataFrame(index=pd.MultiIndex.from_product([dates, tickers]))
    wide = {
        column: fundamentals[column]
        .unstack("ticker")
        .reindex(columns=tickers)
        .sort_index()
        .reindex(fundamentals.index.get_level_values("date").unique().union(dates))
        .ffill()
        .reindex(dates)
        for column in FUNDAMENTAL_FEATURES
    }
    stacked = {name: frame.stack(future_stack=True) for name, frame in wide.items()}
    out = pd.DataFrame(stacked)
    out.index.names = ["date", "ticker"]
    return out


def fill_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace a missing ratio with that date's median across the market.

    Not every filer tags every concept — GOOGL reports no gross profit —
    and dropping those rows would quietly remove whole companies from the
    training set. A median-filled row says "unremarkable", which is the
    honest prior for a number that was never disclosed.
    """
    filled = frame.copy()
    for column in FUNDAMENTAL_FEATURES:
        if column not in filled:
            continue
        medians = filled[column].groupby(level="date").transform("median")
        filled[column] = filled[column].fillna(medians)
    return filled


__all__ = [
    "FUNDAMENTAL_FEATURES",
    "align_to_dates",
    "fetch_history",
    "fill_cross_section",
    "month_ends",
    "to_frame",
]
