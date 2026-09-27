"""Holding alerts between the weekly emails (reporting/drop_alert.py)."""

from __future__ import annotations

import math
from datetime import date, timedelta

from stock_analyzer.reporting.drop_alert import (
    Drop,
    about_company,
    build_alert,
    find_calls_near_strike,
    find_drops,
)

TODAY = date(2026, 9, 25)


def _series(daily_moves: list[float], last: float, *, end: date = TODAY):
    """Closes whose returns alternate +/- `daily_moves`, then `last` today."""
    closes = [100.0]
    for i, m in enumerate(daily_moves):
        closes.append(closes[-1] * (1 + (m if i % 2 else -m)))
    closes.append(closes[-1] * (1 + last))
    days = [end - timedelta(days=len(closes) - 1 - i) for i in range(len(closes))]
    return list(zip(days, closes, strict=True))


def test_drop_is_measured_against_the_stocks_own_usual_move():
    series = {
        "OKLO": _series([0.06] * 70, -0.07),  # 7% on a 6%-a-day stock: ordinary
        "GOOGL": _series([0.01] * 70, -0.05),  # 5% on a 1%-a-day stock: 5 sigma
        "CALM": _series([0.01] * 70, -0.02),  # 2 sigma: not enough
    }
    drops = find_drops(series, today=TODAY)
    assert [d.ticker for d in drops] == ["GOOGL"]
    d = drops[0]
    assert d.change_pct == -5.0 and math.isclose(d.usual_pct, 1.0) and d.sigmas >= 3


def test_a_stale_series_never_alerts_twice():
    series = {"GOOGL": _series([0.01] * 70, -0.05, end=TODAY - timedelta(days=1))}
    assert find_drops(series, today=TODAY) == []
    assert find_drops({"NEW": _series([0.01] * 10, -0.5)}, today=TODAY) == []  # too little history


def test_call_alerts_on_the_day_the_strike_comes_within_reach_only():
    calls = {"TSLA": {"legs": [{"strike": 500.0, "expiry": "2026-11-20", "contracts": 2}]}}
    y = TODAY - timedelta(days=1)
    crossed = {"TSLA": [(y, 470.0), (TODAY, 480.0)]}  # 4.2% below the strike
    near = find_calls_near_strike(calls, crossed, today=TODAY)
    assert [(c.ticker, c.strike, c.contracts) for c in near] == [("TSLA", 500.0, 2)]
    assert round(near[0].gap_pct, 1) == 4.2
    stayed = {"TSLA": [(y, 480.0), (TODAY, 485.0)]}  # already close yesterday
    assert find_calls_near_strike(calls, stayed, today=TODAY) == []
    far = {"TSLA": [(y, 400.0), (TODAY, 420.0)]}
    assert find_calls_near_strike(calls, far, today=TODAY) == []


def test_alert_email_explains_itself_and_is_silent_when_nothing_happened():
    assert build_alert([], []) is None
    d = Drop("BE", TODAY, -21.0, 6.5)
    d.context = {
        "spy_pct": -0.4,
        "etf": "XLI",
        "etf_pct": -0.9,
        "sector": "Industrials",
        "news": [
            {"title": "Short seller report", "url": "https://x", "published_date": "2026-09-25"}
        ],
        "thesis": {
            "pick_date": "2026-05-01",
            "status": "WATCH",
            "signals": [{"text": "below 200d"}],
        },
        "events": [{"issue": "material weakness", "category": "material_weakness"}],
        "filing": "10-Q for 2026-06-30",
    }
    subject, body = build_alert([d], [])
    assert subject == "Holding alert: BE -21%"
    for text in (
        "3.2x its usual",
        "SPY -0.4%",
        "Industrials (XLI) -0.9%",
        "Short seller report",
        "WATCH",
        "material weakness",
        "No action needed unless the thesis changed",
    ):
        assert text in body, text


def test_news_keeps_only_headlines_about_the_company():
    items = [
        {"title": "Why the Nasdaq refuses to break even"},
        {"title": "Alphabet faces new EU fine over ad tech"},
        {"title": "GOOGL slides after antitrust ruling"},
    ]
    kept = about_company(items, "GOOGL", "Alphabet Inc.")
    assert kept == items[1:]
