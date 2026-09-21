"""Measuring the screen's own components — and refusing to when it can't.

Ten components award points and none had been measured against a realized
return. The danger is not getting it wrong; it is producing a confident
table from three run dates and letting it change a weight.
"""

from __future__ import annotations

from stock_analyzer.discover.score_attribution import (
    MIN_DATES,
    MIN_ROWS_PER_DATE,
    ComponentResult,
    attribute,
    enough_data,
    format_report,
)


def _rows(dates: int, per_date: int, *, signal: bool) -> list[dict]:
    """`signal=True` builds a component that perfectly ranks the outcome."""
    out = []
    for d in range(dates):
        for i in range(per_date):
            out.append(
                {
                    "day": f"2026-05-{d + 1:02d}",
                    "ticker": f"T{i}",
                    "excess": float(i),
                    "trend.good": float(i) if signal else float(per_date - i),
                    "trend.noise": float((i * 7) % per_date),
                }
            )
    return out


def test_a_component_that_ranks_the_outcome_is_found():
    results = attribute(_rows(MIN_DATES, MIN_ROWS_PER_DATE, signal=True))
    best = next(r for r in results if r.component == "trend.good")
    assert best.ic > 0.99
    assert best.verdict == "predicts"
    assert best.dates == MIN_DATES


def test_a_component_that_ranks_it_backwards_is_named_as_such():
    results = attribute(_rows(MIN_DATES, MIN_ROWS_PER_DATE, signal=False))
    worst = next(r for r in results if r.component == "trend.good")
    assert worst.ic < -0.99
    assert worst.verdict == "predicts inversely"


def test_thin_days_are_dropped_not_averaged_in():
    """A date with a handful of candidates cannot support a cross-section."""
    rows = _rows(MIN_DATES, MIN_ROWS_PER_DATE, signal=True)
    rows += [
        {"day": "2026-07-01", "ticker": "X", "excess": 1.0, "trend.good": 1.0, "trend.noise": 1.0}
    ]
    results = attribute(rows)
    assert max(r.dates for r in results) == MIN_DATES, "the thin day must not count"


def test_too_few_dates_is_reported_as_not_enough_data():
    results = attribute(_rows(3, MIN_ROWS_PER_DATE, signal=True))
    assert results, "it still computes — the guard is about what it claims"
    assert not enough_data(results)
    report = format_report(results, 63, 60)
    assert "NOT ENOUGH DATA" in report
    assert "no power" in report
    # And the opposite case says nothing of the kind.
    ok = attribute(_rows(MIN_DATES, MIN_ROWS_PER_DATE, signal=True))
    assert enough_data(ok)
    assert "NOT ENOUGH DATA" not in format_report(ok, 63, 600)


def test_nothing_joinable_says_so_plainly():
    assert attribute([]) == []
    assert "No component could be measured" in format_report([], 63, 0)


def test_a_weak_signal_is_called_no_evidence_not_a_finding():
    r = ComponentResult(
        component="trend.noise", ic=0.05, t_stat=0.9, dates=20, rows=400, coverage=1.0
    )
    assert r.verdict == "no evidence"
    assert ComponentResult("x", 0.2, 2.5, 20, 400, 1.0).verdict == "predicts"
