"""Is the portfolio on pace for its goal? The daily email's one-line answer.

The plan check (goal_projection) gives odds from ten thousand simulated
futures once in a while. This is the everyday number: the steady yearly
return that takes today's total, plus the usual monthly contribution, to
GOAL_TARGET_USD by GOAL_DATE. Computed again from the first stored daily
total, it shows the drift: a required return that keeps rising means the
portfolio is falling behind the path, a falling one means it is ahead.

Nominal dollars, contributions at the end of each month. No LLM calls.
"""

from __future__ import annotations

from datetime import date
from typing import Any

# Bisection bounds on the yearly return. Beyond 100% a year the goal is
# out of reach in any useful sense; below -50% it is already met.
_LOWEST, _HIGHEST = -0.5, 1.0


def months_until(today: date, goal: date) -> int:
    return max((goal.year - today.year) * 12 + goal.month - today.month, 0)


def future_value(start: float, monthly: float, months: int, annual_return: float) -> float:
    rate = (1 + annual_return) ** (1 / 12) - 1
    growth = (1 + rate) ** months
    if abs(rate) < 1e-12:
        return start + monthly * months
    return start * growth + monthly * (growth - 1) / rate


def required_return(start: float, monthly: float, months: int, target: float) -> float | None:
    """The steady yearly return that reaches `target`; None when no
    return up to 100% a year does, or there is no time left."""
    if months <= 0 or future_value(start, monthly, months, _HIGHEST) < target:
        return None
    lo, hi = _LOWEST, _HIGHEST
    if future_value(start, monthly, months, lo) >= target:
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if future_value(start, monthly, months, mid) < target else (lo, mid)
    return hi


def goal_pace(
    totals: list[tuple[date, float]],
    *,
    target: float,
    goal_date: date,
    monthly: float,
) -> dict[str, Any] | None:
    """{"needed", "was", "since", ...} from the stored daily totals (oldest
    first); "was" is the same figure from the first total, None when there
    is only one."""
    if not totals:
        return None
    day, value = totals[-1]
    first_day, first_value = totals[0]
    was = (
        required_return(first_value, monthly, months_until(first_day, goal_date), target)
        if first_day < day
        else None
    )
    return {
        "target": target,
        "goal_date": goal_date,
        "monthly": monthly,
        "value": value,
        "needed": required_return(value, monthly, months_until(day, goal_date), target),
        "was": was,
        "since": first_day if was is not None else None,
    }
