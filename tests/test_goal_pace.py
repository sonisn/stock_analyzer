"""The daily email's goal line: the steady return still needed, and its drift."""

from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.discover.goal_pace import (
    future_value,
    goal_pace,
    months_until,
    required_return,
)
from stock_analyzer.reporting.health import PortfolioHealth, _goal_pace_html

GOAL = date(2050, 1, 1)


def test_the_required_return_is_the_one_that_lands_on_the_target():
    needed = required_return(490_876, 2_690, 280, 10_000_000)
    assert needed == pytest.approx(0.117, abs=0.001)
    assert future_value(490_876, 2_690, 280, needed) == pytest.approx(10_000_000, rel=1e-6)


def test_no_return_at_all_is_needed_when_contributions_alone_get_there():
    assert future_value(0, 1_000, 12, 0.0) == 12_000
    assert required_return(0, 1_000, 12, 12_000) == pytest.approx(0.0, abs=1e-6)


def test_an_unreachable_goal_or_no_time_left_has_no_answer():
    assert required_return(1_000, 0, 12, 10_000_000) is None
    assert required_return(1_000, 0, 0, 2_000) is None


def test_months_until_counts_calendar_months():
    assert months_until(date(2026, 9, 27), GOAL) == 280
    assert months_until(date(2051, 1, 1), GOAL) == 0


def test_falling_value_raises_the_return_needed():
    pace = goal_pace(
        [(date(2026, 9, 21), 500_000), (date(2026, 9, 26), 480_000)],
        target=10_000_000,
        goal_date=GOAL,
        monthly=2_690,
    )
    assert pace["needed"] > pace["was"]
    assert pace["since"] == date(2026, 9, 21)


def test_one_stored_total_has_no_drift_yet():
    pace = goal_pace([(date(2026, 9, 26), 480_000)], target=1e7, goal_date=GOAL, monthly=0)
    assert pace["was"] is None and pace["since"] is None
    assert goal_pace([], target=1e7, goal_date=GOAL, monthly=0) is None


def _line(**pace):
    base = {"target": 1e7, "goal_date": GOAL, "monthly": 2_690, "value": 490_876}
    return _goal_pace_html(PortfolioHealth(goal_pace={**base, **pace}))


def test_the_line_says_behind_when_the_needed_return_rose():
    text = _line(needed=0.117, was=0.115, since=date(2026, 9, 21))
    assert "needs 11.7% a year" in text
    assert "was 11.5% on Sep 21, behind: +0.2 pts" in text
    assert "#9c1010" in text  # above SPY's long-run ~10%


def test_the_line_says_ahead_and_stays_grey_within_reach():
    text = _line(needed=0.08, was=0.09, since=date(2026, 9, 21))
    assert "ahead: -1.0 pts" in text
    assert "#9c1010" not in text


def test_an_unreachable_goal_says_so_and_no_goal_says_nothing():
    assert "out of reach" in _line(needed=None, was=None, since=None)
    assert _goal_pace_html(PortfolioHealth()) == ""
