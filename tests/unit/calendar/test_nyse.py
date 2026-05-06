from __future__ import annotations

from datetime import date

from stock_analyzer.calendar.nyse import is_market_holiday, is_trading_day, next_trading_day


def test_christmas_is_holiday() -> None:
    assert is_market_holiday(date(2026, 12, 25)) is True


def test_random_weekday_is_trading_day() -> None:
    # Wed, May 6 2026 — not a known NYSE holiday
    assert is_trading_day(date(2026, 5, 6)) is True


def test_saturday_is_not_trading_day() -> None:
    # Sat, May 9 2026
    assert is_trading_day(date(2026, 5, 9)) is False


def test_next_trading_day_skips_weekend() -> None:
    # Friday → Monday
    assert next_trading_day(date(2026, 5, 8)) == date(2026, 5, 11)


def test_next_trading_day_skips_holiday() -> None:
    # Day before Independence Day observed (in 2026, July 4 falls Saturday → observed Friday July 3)
    fri = date(2026, 7, 3)
    nxt = next_trading_day(fri)
    assert nxt > fri
    assert is_trading_day(nxt)
