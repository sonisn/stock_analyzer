"""The one shape price data takes inside the package, and the conversions to it.

Every daily-bar frame is a Polars DataFrame sorted by a `date` column
(pl.Date: the trading day, no clock time, no zone) with yfinance's own
column names — Open, High, Low, Close, Volume, Dividends, Stock Splits —
so code reads `frame["Close"]` as it always did.

yfinance hands back pandas, indexed by timezone-aware midnights (from
`Ticker.history`) or naive ones (from `download`); `bars_from_pandas` is the
single place that crosses from that to this. The bar store's Parquet files
written before the switch are the same pandas shape and go through it too.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

DATE = "date"
BAR_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")


def bars_from_pandas(frame: Any) -> pl.DataFrame | None:
    """A yfinance / pandas bar frame as the package's shape, or None when
    empty. The index's date part is the trading day: yfinance stamps daily
    bars at midnight in the exchange's zone."""
    if frame is None or getattr(frame, "empty", True):
        return None
    days = [date.fromisoformat(str(i)[:10]) for i in frame.index]
    data: dict[str, Any] = {DATE: days}
    for col in frame.columns:
        name = str(col)
        if name in BAR_COLUMNS:
            data[name] = frame[col].to_numpy(dtype=float)
    out = pl.DataFrame(data, schema_overrides={DATE: pl.Date})
    return normalize(out)


def table_from_pandas(frame: Any, index: str = "index") -> pl.DataFrame | None:
    """Any other yfinance table (estimates, statements, holders, analyst
    actions) as Polars, its row labels kept as a string column `index` and
    every column name a string. None when empty. Values keep their types;
    NaN in a float column becomes null."""
    if frame is None or getattr(frame, "empty", True):
        return None
    if not hasattr(frame, "columns"):  # a Series (share counts, splits)
        frame = frame.to_frame(name=str(getattr(frame, "name", None) or "value"))
    data: dict[str, Any] = {index: [str(i) for i in frame.index]}
    for col in frame.columns:
        values = frame[col].tolist()
        data[str(col)] = [None if isinstance(v, float) and v != v else v for v in values]
    return pl.DataFrame(data, strict=False)


def cell(table: pl.DataFrame | None, row: str, col: str, index: str = "index") -> Any:
    """table[row, col] by row label, or None."""
    if table is None or col not in table.columns:
        return None
    hit = table.filter(pl.col(index) == row)
    return None if hit.is_empty() else hit[col][0]


def normalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Sorted by date, one row per day (the last wins), floats for prices,
    and a missing value as null rather than NaN — so a mean, max or rolling
    window skips it, as pandas skipped NaN."""
    casts = [pl.col(c).cast(pl.Float64).fill_nan(None) for c in frame.columns if c != DATE]
    return (
        frame.with_columns(casts).unique(subset=DATE, keep="last", maintain_order=True).sort(DATE)
    )


def from_parquet_table(frame: pl.DataFrame) -> pl.DataFrame:
    """A stored file in either layout: this package's (a `date` column) or
    the pandas one written before the switch (the index saved as `Date`, a
    timezone-aware timestamp whose local date is the trading day)."""
    if DATE in frame.columns:
        return normalize(frame.with_columns(pl.col(DATE).cast(pl.Date)))
    if "Date" in frame.columns:
        col = frame["Date"]
        days = col.dt.date() if col.dtype != pl.Date else col
        frame = frame.drop("Date").with_columns(days.alias(DATE))
        keep = [DATE, *(c for c in BAR_COLUMNS if c in frame.columns)]
        return normalize(frame.select(keep))
    raise ValueError("no date column")


def since(frame: pl.DataFrame, start: date, end: date | None = None) -> pl.DataFrame:
    """Rows from `start` through `end` (inclusive)."""
    cond = pl.col(DATE) >= start
    if end is not None:
        cond &= pl.col(DATE) <= end
    return frame.filter(cond)


def closes(frame: pl.DataFrame | None) -> pl.DataFrame | None:
    """The two columns most callers need: date and Close, no missing closes."""
    if frame is None or frame.is_empty() or "Close" not in frame.columns:
        return None
    return frame.select(DATE, pl.col("Close").fill_nan(None)).drop_nulls("Close")


def by_day(frame: pl.DataFrame | None, col: str = "Close") -> dict[date, float]:
    """{day: value} for the rows where `col` is present."""
    if frame is None or frame.is_empty() or col not in frame.columns:
        return {}
    rows = frame.select(DATE, pl.col(col).fill_nan(None)).drop_nulls(col)
    return dict(zip(rows[DATE].to_list(), rows[col].to_list(), strict=True))


def _present(value: Any) -> bool:
    return value is not None and value == value  # NaN is the one value unequal to itself


def value_on_or_before(
    frame: pl.DataFrame, on: date, col: str = "Close"
) -> tuple[float, date] | None:
    """The `col` of the last row at or before `on`, with its date; None when
    there is no such row or its value is missing (not the last present one)."""
    rows = frame.filter(pl.col(DATE) <= on)
    if rows.is_empty():
        return None
    last = rows.row(-1, named=True)
    return (float(last[col]), last[DATE]) if _present(last[col]) else None


def value_on_or_after(
    frame: pl.DataFrame, on: date, col: str = "Close"
) -> tuple[float, date] | None:
    """The `col` of the first row at or after `on`, with its date; None when
    there is no such row or its value is missing."""
    rows = frame.filter(pl.col(DATE) >= on)
    if rows.is_empty():
        return None
    first = rows.row(0, named=True)
    return (float(first[col]), first[DATE]) if _present(first[col]) else None
