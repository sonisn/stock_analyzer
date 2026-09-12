"""Screen-score validation — the retrospective check on the 0-105 composite.

These tests seed a database the way the discover pipeline does, hand the
validator synthetic price paths with a KNOWN relationship between score and
forward return, and assert that the harness recovers that relationship.
Without this, a validation tool could report a comforting number while
measuring nothing.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import text

from stock_analyzer.db.session import get_session
from stock_analyzer.discover import score_validation as sv

_HORIZON = 90


def _price_frame(start: date, *, entry: float, at_horizon: float) -> pd.DataFrame:
    idx = pd.date_range(start - timedelta(days=10), start + timedelta(days=120), freq="D")
    per_day = (at_horizon - entry) / _HORIZON
    closes = [
        entry if ts.date() <= start else entry + per_day * (ts.date() - start).days for ts in idx
    ]
    return pd.DataFrame({"Close": closes}, index=idx)


def _seed_db(
    db_path: str,
    rows: list[tuple[str, float, dict, dict]],
    *,
    age_days: int = 200,
) -> str:
    """Insert one run with the given (ticker, score, components, breakdown)."""
    run_at = (datetime.now() - timedelta(days=age_days)).isoformat(timespec="seconds")
    with get_session(db_path) as session:
        result = session.exec(
            text(
                "INSERT INTO runs (run_at, kind, universe_size, survivors, picks, "
                "opus_model, sonnet_model) "
                "VALUES (:run_at, 'discover', :n, :n, 5, 'o', 's')"
            ),
            params={"run_at": run_at, "n": len(rows)},
        )
        run_id = result.lastrowid
        for ticker, score, components, breakdown in rows:
            session.exec(
                text(
                    "INSERT INTO candidates (run_id, ticker, passed_filter, score, "
                    "score_components, score_breakdown, conviction) "
                    "VALUES (:rid, :ticker, 1, :score, :components, :breakdown, 3)"
                ),
                params={
                    "rid": run_id,
                    "ticker": ticker,
                    "score": score,
                    "components": json.dumps(components),
                    "breakdown": json.dumps(breakdown),
                },
            )
    return run_at.split("T")[0]


# --- rank statistics ------------------------------------------------------


def test_spearman_is_one_for_a_monotone_relationship():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [10.0, 20.0, 25.0, 40.0, 100.0]  # monotone but not linear
    assert sv.spearman(xs, ys) == pytest.approx(1.0)


def test_spearman_is_minus_one_when_reversed():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert sv.spearman(xs, ys) == pytest.approx(-1.0)


def test_spearman_handles_ties_without_blowing_up():
    xs = [1.0, 1.0, 2.0, 2.0, 3.0]
    ys = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = sv.spearman(xs, ys)
    assert out is not None
    assert 0.8 < out <= 1.0


def test_spearman_none_when_a_series_is_constant():
    """A component every candidate scores identically carries no ranking
    information — and must not raise a divide-by-zero either."""
    assert sv.spearman([2.0] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) is None


# --- component flattening -------------------------------------------------


def test_flatten_components_keeps_numeric_leaves_and_skips_metadata():
    components = {"fundamentals": 30.0, "trend": 25.0, "theme_bonus": 2.0}
    breakdown = {
        "fundamentals": {"revenue_growth": 12.0, "fcf_yield": 8.0},
        "conviction": {"mentions": 7.5, "source_diversity": 3.0},
        # The theme block carries prose metadata, not scores.
        "theme": {"name": "AI capex", "trend": "rising"},
    }
    out = sv._flatten_components(json.dumps(components), json.dumps(breakdown))
    assert out["total.fundamentals"] == 30.0
    assert out["fundamentals.revenue_growth"] == 12.0
    assert out["conviction.mentions"] == 7.5
    assert not any(k.startswith("theme.") for k in out)


def test_flatten_components_survives_malformed_json():
    assert sv._flatten_components("{not json", None) == {}


# --- end-to-end validation ------------------------------------------------


def test_validation_detects_a_score_that_predicts_returns():
    """High scores that really did outperform must produce a positive score
    IC and a rising quintile curve."""
    rows = []
    for i in range(30):
        score = 40.0 + i * 2  # 40 .. 98
        rows.append(
            (
                f"T{i:02d}",
                score,
                {"fundamentals": score * 0.4, "conviction": 10.0},
                {"conviction": {"mentions": 10.0}},
            )
        )

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        run_date = _seed_db(db_path, rows)
        start = date.fromisoformat(run_date)

        def fake_history(ticker, _start, _end):
            if ticker == "SPY":
                return _price_frame(start, entry=400.0, at_horizon=400.0)
            # Higher index (higher score) → higher forward return.
            idx = int(ticker[1:])
            return _price_frame(start, entry=100.0, at_horizon=100.0 + idx)

        with patch.object(sv, "_fetch_history", side_effect=fake_history):
            report = sv.validate_score(db_path, horizon_days=_HORIZON)

    assert report.n_candidates == 30
    assert report.score_ic is not None
    assert report.score_ic == pytest.approx(1.0, abs=1e-9)
    assert len(report.buckets) == 5
    # Rising curve, Q1 through Q5.
    means = [b.mean_alpha_pct for b in report.buckets]
    assert means == sorted(means)
    assert report.buckets[-1].mean_alpha_pct > report.buckets[0].mean_alpha_pct

    out = sv.format_validation_report(report)
    assert "score is separating" in out


def test_validation_flags_a_component_with_the_wrong_sign():
    """The finding that matters most: a sub-score that points the wrong way.

    `conviction.mentions` is built here to correlate NEGATIVELY with
    forward alpha — the crowding/attention failure mode — and the report
    must name it as such rather than burying it in a table.
    """
    rows = []
    for i in range(24):
        # mentions score runs opposite to realized performance
        mentions = float(15 - i // 2)
        rows.append(
            (
                f"X{i:02d}",
                60.0,
                {"conviction": mentions},
                {"conviction": {"mentions": mentions}},
            )
        )

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        run_date = _seed_db(db_path, rows)
        start = date.fromisoformat(run_date)

        def fake_history(ticker, _start, _end):
            if ticker == "SPY":
                return _price_frame(start, entry=400.0, at_horizon=400.0)
            idx = int(ticker[1:])
            return _price_frame(start, entry=100.0, at_horizon=100.0 + idx)

        with patch.object(sv, "_fetch_history", side_effect=fake_history):
            report = sv.validate_score(db_path, horizon_days=_HORIZON)

    mentions = next(c for c in report.component_ics if c.component == "conviction.mentions")
    assert mentions.ic < -0.05
    assert mentions.verdict == "WRONG SIGN — actively hurting"

    out = sv.format_validation_report(report)
    assert "wrong sign" in out.lower()
    assert "conviction.mentions" in out


def test_validation_reports_a_flat_score_as_no_separation():
    """If score and outcome are unrelated the report must say so plainly
    instead of implying the composite works."""
    rows = [
        (f"F{i:02d}", 50.0 + i, {"trend": 20.0}, {"trend": {"rs_6mo": 10.0}}) for i in range(25)
    ]
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        run_date = _seed_db(db_path, rows)
        start = date.fromisoformat(run_date)

        def fake_history(ticker, _start, _end):
            if ticker == "SPY":
                return _price_frame(start, entry=400.0, at_horizon=400.0)
            # Same forward return for everyone → zero separation.
            return _price_frame(start, entry=100.0, at_horizon=105.0)

        with patch.object(sv, "_fetch_history", side_effect=fake_history):
            report = sv.validate_score(db_path, horizon_days=_HORIZON)

    assert report.buckets
    spread = report.buckets[-1].mean_alpha_pct - report.buckets[0].mean_alpha_pct
    assert spread == pytest.approx(0.0, abs=1e-9)
    assert "NO separation" in sv.format_validation_report(report)


def test_candidates_younger_than_the_horizon_are_excluded():
    """A run from last week has no finished 90-day window and must not be
    graded on a partial one."""
    rows = [("NEW", 80.0, {"trend": 20.0}, {})]
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        _seed_db(db_path, rows, age_days=7)
        loaded = sv._load_candidates(db_path, 540, _HORIZON)
    assert loaded == []


def test_repeat_appearances_count_once():
    """A ticker surfaced by many runs contributes one observation, not one
    per run — otherwise frequently-surfaced names dominate the sample."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        _seed_db(db_path, [("NVDA", 70.0, {}, {})], age_days=200)
        _seed_db(db_path, [("NVDA", 90.0, {}, {})], age_days=150)
        loaded = sv._load_candidates(db_path, 540, _HORIZON)
    assert len(loaded) == 1
    assert loaded[0][2] == 70.0  # the earliest score is kept


def test_empty_db_reports_nothing_to_grade():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        with get_session(db_path):
            pass
        report = sv.validate_score(db_path, horizon_days=_HORIZON)
    assert report.n_candidates == 0
    assert "nothing to grade yet" in sv.format_validation_report(report)


def test_large_spread_without_rank_support_is_not_called_separation():
    """The guard against false comfort: a noisy score can produce a big
    Q5-Q1 gap on small buckets while the overall ranking is uncorrelated.

    These are the real numbers a randomly-generated score produced on 30
    S&P names: a +11.95% Q5-Q1 spread on an IC of 0.057, which is well
    inside the noise bar for n=30. The verdict must refuse to call that
    separation, and must say how large an IC would be needed.
    """
    report = sv.ValidationReport(
        horizon_days=90,
        n_candidates=30,
        n_unmeasurable=0,
        buckets=[
            sv.Bucket(
                label="Q1",
                n=6,
                mean_score=41.0,
                mean_alpha_pct=-8.0,
                median_alpha_pct=-8.0,
                hit_rate=0.17,
            ),
            sv.Bucket(
                label="Q2",
                n=6,
                mean_score=54.0,
                mean_alpha_pct=+13.7,
                median_alpha_pct=-4.3,
                hit_rate=0.33,
            ),
            sv.Bucket(
                label="Q3",
                n=6,
                mean_score=63.6,
                mean_alpha_pct=+11.1,
                median_alpha_pct=+6.8,
                hit_rate=0.50,
            ),
            sv.Bucket(
                label="Q4",
                n=6,
                mean_score=80.9,
                mean_alpha_pct=-8.1,
                median_alpha_pct=-11.1,
                hit_rate=0.17,
            ),
            sv.Bucket(
                label="Q5",
                n=6,
                mean_score=93.1,
                mean_alpha_pct=+3.9,
                median_alpha_pct=+0.2,
                hit_rate=0.50,
            ),
        ],
        component_ics=[],
        score_ic=0.057,
        mean_alpha_pct=2.49,
    )
    out = sv.format_validation_report(report)
    assert "NO reliable separation" in out
    assert "NOT monotone" in out


def test_monotone_curve_with_real_ic_is_called_separation():
    report = sv.ValidationReport(
        horizon_days=90,
        n_candidates=40,
        n_unmeasurable=0,
        buckets=[
            sv.Bucket(
                label=f"Q{i}",
                n=8,
                mean_score=40.0 + i * 10,
                mean_alpha_pct=-6.0 + i * 4.0,
                median_alpha_pct=-6.0 + i * 4.0,
                hit_rate=0.4 + i * 0.05,
            )
            for i in range(1, 6)
        ],
        component_ics=[],
        score_ic=0.45,
        mean_alpha_pct=4.0,
    )
    out = sv.format_validation_report(report)
    assert "the score is separating" in out
    assert "is monotone" in out
    assert "noise bar" in out


def test_component_ic_inside_the_noise_bar_is_inconclusive_not_wrong_sign():
    """A small negative IC on a small sample must NOT be reported as a
    broken component — that would send someone deleting a signal over noise.

    -0.059 at n=30 is the real number a randomly-generated score produced.
    """
    component = sv.ComponentIC(component="conviction.mentions", n=30, ic=-0.059, mean_value=3.0)
    assert component.is_significant is False
    assert "inconclusive" in component.verdict
    assert "WRONG SIGN" not in component.verdict


def test_component_ic_clearing_the_bar_is_reported_as_wrong_sign():
    component = sv.ComponentIC(component="conviction.mentions", n=400, ic=-0.21, mean_value=3.0)
    assert component.is_significant is True
    assert component.verdict == "WRONG SIGN — actively hurting"


def test_action_block_only_lists_components_that_clear_the_bar():
    report = sv.ValidationReport(
        horizon_days=90,
        n_candidates=30,
        n_unmeasurable=0,
        buckets=[],
        component_ics=[
            sv.ComponentIC(component="noise.one", n=30, ic=-0.06, mean_value=1.0),
            sv.ComponentIC(component="real.bad", n=400, ic=-0.25, mean_value=1.0),
        ],
        score_ic=0.02,
        mean_alpha_pct=1.0,
    )
    out = sv.format_validation_report(report)
    assert "real.bad" in out.split("ACTION:")[1]
    assert "noise.one" not in out.split("ACTION:")[1]


def test_no_significant_components_says_sample_size_not_all_clear():
    report = sv.ValidationReport(
        horizon_days=90,
        n_candidates=30,
        n_unmeasurable=0,
        buckets=[],
        component_ics=[
            sv.ComponentIC(component="a", n=30, ic=0.05, mean_value=1.0),
        ],
        score_ic=0.05,
        mean_alpha_pct=1.0,
    )
    out = sv.format_validation_report(report)
    assert "sample-size result, not a clean bill of health" in out
