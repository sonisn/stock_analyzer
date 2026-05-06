"""Thin wrapper over pandas-market-calendars for NYSE trading days/holidays."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import pandas_market_calendars as mcal


@lru_cache(maxsize=1)
def _nyse() -> Any:
    return mcal.get_calendar("XNYS")


def is_trading_day(d: date) -> bool:
    schedule = _nyse().valid_days(start_date=d, end_date=d)
    return len(schedule) == 1


def is_market_holiday(d: date) -> bool:
    if d.weekday() >= 5:  # weekends are not "holidays" per se
        return False
    return not is_trading_day(d)


def next_trading_day(d: date) -> date:
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def previous_trading_day(d: date) -> date:
    candidate = d - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate
