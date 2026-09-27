"""Build price frames for tests in the package's bar shape (data/frames.py)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl


def days(start: date, end: date) -> list[date]:
    """Every calendar day from `start` through `end` (pandas' freq="D")."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def bdays(
    start: date | str, end: date | str | None = None, *, periods: int | None = None
) -> list[date]:
    """Weekdays from `start` through `end`, or `periods` of them (pandas'
    bdate_range)."""
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else end
    out, d = [], start
    while (periods is not None and len(out) < periods) or (end is not None and d <= end):
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def bars(index: list[date], columns: dict[str, Any]) -> pl.DataFrame:
    """A bar frame: `date` plus the given columns (lists or scalars)."""
    n = len(index)
    data: dict[str, Any] = {"date": list(index)}
    for name, values in columns.items():
        data[name] = (
            [float(v) for v in values] if hasattr(values, "__len__") else [float(values)] * n
        )
    return pl.DataFrame(data, schema_overrides={"date": pl.Date})
