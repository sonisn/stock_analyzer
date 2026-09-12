"""Forecast calibration — grading the ranker's own conviction and EV.

The ranker prompt claims "you ARE measured on it". These tests hold that
claim up: a forecast is persisted, a known outcome is supplied, and the
calibration pass must recover the error, the conviction ordering, and the
scenario reliability — then say something actionable in the prompt block.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import text

from stock_analyzer.db.repository import insert_pick, insert_run
from stock_analyzer.db.session import get_session
from stock_analyzer.discover import calibration as cal
from stock_analyzer.models.calibration import (
    CalibrationRecord,
    ConvictionBucket,
)

_EV_HORIZON = cal._EV_HORIZON_DAYS


def _price_frame(start: date, *, entry: float, at_horizon: float, horizon: int) -> pd.DataFrame:
    idx = pd.date_range(start - timedelta(days=10), start + timedelta(days=horizon + 30), freq="D")
    per_day = (at_horizon - entry) / horizon
    closes = [
        entry if ts.date() <= start else entry + per_day * (ts.date() - start).days for ts in idx
    ]
    return pd.DataFrame({"Close": closes}, index=idx)


def _seed_pick(
    db_path: str,
    *,
    ticker: str,
    conviction: int,
    scenarios: list[dict],
    entry_price: float = 100.0,
    age_days: int = _EV_HORIZON + 30,
    rank: int = 1,
) -> str:
    """Persist one pick through the real repository path."""
    run_at = (datetime.now() - timedelta(days=age_days)).isoformat(timespec="seconds")
    with get_session(db_path) as session:
        run_id = insert_run(
            session,
            universe_size=10,
            survivors=5,
            picks=1,
            opus_model="opus-test",
            sonnet_model="sonnet-test",
            cash_budget=None,
        )
        # insert_run stamps "now"; rewrite it so the pick has the age we want.
        session.exec(
            text("UPDATE runs SET run_at = :run_at WHERE id = :rid"),
            params={"run_at": run_at, "rid": run_id},
        )
        ev = sum(s["probability"] * s["target_return_pct"] for s in scenarios)
        insert_pick(
            session,
            run_id,
            rank=rank,
            ticker=ticker,
            ranker_text="",
            bear_case_text=None,
            allocation_text=None,
            conviction=conviction,
            ev_pct=ev,
            entry_price=entry_price,
            time_horizon="6-12 months",
            scenarios=scenarios,
        )
    return run_at.split("T")[0]


_BALANCED = [
    {"label": "bull", "probability": 0.5, "target_return_pct": 40.0},
    {"label": "base", "probability": 0.35, "target_return_pct": 10.0},
    {"label": "bear", "probability": 0.15, "target_return_pct": -20.0},
]


# --- persistence round trip -----------------------------------------------


def test_forecast_round_trips_through_the_repository():
    """The whole point of F1: conviction, EV and every scenario survive a
    write/read cycle. Before this, only prose reached disk."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        _seed_pick(db_path, ticker="NVDA", conviction=9, scenarios=_BALANCED)
        loaded = cal._load_forecasts(db_path, 540)

    assert len(loaded) == 1
    forecast = loaded[0]
    assert forecast.ticker == "NVDA"
    assert forecast.conviction == 9
    # EV = .5*40 + .35*10 + .15*-20 = 20.5
    assert forecast.ev_pct == pytest.approx(20.5)
    assert forecast.entry_price == pytest.approx(100.0)
    assert set(forecast.scenarios) == {"bull", "base", "bear"}
    assert forecast.scenarios["bear"] == (0.15, -20.0)


def test_scenarios_with_an_unknown_label_are_rejected_not_stored():
    """Only bull/base/bear are meaningful; anything else would corrupt the
    reliability table."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        _seed_pick(
            db_path,
            ticker="X",
            conviction=7,
            scenarios=[
                {"label": "bull", "probability": 0.6, "target_return_pct": 30.0},
                {"label": "moon", "probability": 0.4, "target_return_pct": 300.0},
            ],
        )
        loaded = cal._load_forecasts(db_path, 540)
    assert set(loaded[0].scenarios) == {"bull"}


# --- EV error -------------------------------------------------------------


def test_ev_error_is_negative_when_the_forecast_was_too_optimistic():
    """EV said +20.5%, the stock did +5% → error -15.5%, and the block must
    tell the ranker it overshot."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        run_date = _seed_pick(db_path, ticker="NVDA", conviction=9, scenarios=_BALANCED)
        start = date.fromisoformat(run_date)

        def fake_history(ticker, _s, _e):
            if ticker == "SPY":
                return _price_frame(start, entry=400.0, at_horizon=400.0, horizon=_EV_HORIZON)
            return _price_frame(start, entry=100.0, at_horizon=105.0, horizon=_EV_HORIZON)

        with patch.object(cal, "_fetch_history", side_effect=fake_history):
            record = cal.measure_calibration(db_path)

    assert record.n_scored == 1
    assert record.mean_ev_error_pct == pytest.approx(5.0 - 20.5, abs=0.2)
    block = cal.format_calibration_block(record)
    assert "OVERSHOT" in block
    assert "EV error" in block


def test_ev_error_is_positive_when_the_forecast_was_too_conservative():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        run_date = _seed_pick(db_path, ticker="AAPL", conviction=6, scenarios=_BALANCED)
        start = date.fromisoformat(run_date)

        def fake_history(ticker, _s, _e):
            if ticker == "SPY":
                return _price_frame(start, entry=400.0, at_horizon=400.0, horizon=_EV_HORIZON)
            return _price_frame(start, entry=100.0, at_horizon=150.0, horizon=_EV_HORIZON)

        with patch.object(cal, "_fetch_history", side_effect=fake_history):
            record = cal.measure_calibration(db_path)

    assert record.mean_ev_error_pct == pytest.approx(50.0 - 20.5, abs=0.5)
    assert "UNDERSHOT" in cal.format_calibration_block(record)


def test_forecast_younger_than_the_ev_horizon_is_pending_not_scored():
    """A 6-12 month forecast must not be graded against a 90-day outcome —
    that would manufacture a pessimism bias that isn't there."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        _seed_pick(
            db_path,
            ticker="NEW",
            conviction=8,
            scenarios=_BALANCED,
            age_days=100,
        )
        with patch.object(cal, "_fetch_history", return_value=None):
            record = cal.measure_calibration(db_path)

    assert record.n_scored == 0
    assert record.n_pending == 1
    block = cal.format_calibration_block(record)
    assert "Not scorable yet" in block


# --- scenario reliability -------------------------------------------------


def test_scenario_reliability_attributes_the_outcome_to_the_nearest_target():
    """Realized -18% is nearest the bear target (-20%), so bear is the
    scenario that landed — and the stated 15% shows against that."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        run_date = _seed_pick(db_path, ticker="BAD", conviction=8, scenarios=_BALANCED)
        start = date.fromisoformat(run_date)

        def fake_history(ticker, _s, _e):
            if ticker == "SPY":
                return _price_frame(start, entry=400.0, at_horizon=400.0, horizon=_EV_HORIZON)
            return _price_frame(start, entry=100.0, at_horizon=82.0, horizon=_EV_HORIZON)

        with patch.object(cal, "_fetch_history", side_effect=fake_history):
            record = cal.measure_calibration(db_path)

    bear = next(s for s in record.scenario_reliability if s.label == "bear")
    assert bear.n_landed == 1
    assert bear.mean_stated_probability == pytest.approx(0.15)
    assert bear.observed_frequency == pytest.approx(1.0)
    assert "bear: said 15% avg, landed 100%" in cal.format_calibration_block(record)


def test_which_scenario_landed_picks_the_closest_target():
    scenarios = {"bull": (0.5, 40.0), "base": (0.35, 10.0), "bear": (0.15, -20.0)}
    assert cal._which_scenario_landed(38.0, scenarios) == "bull"
    assert cal._which_scenario_landed(8.0, scenarios) == "base"
    assert cal._which_scenario_landed(-30.0, scenarios) == "bear"
    assert cal._which_scenario_landed(5.0, {}) is None


# --- conviction ordering --------------------------------------------------


def test_conviction_monotonicity_holds_when_high_beats_low():
    record = CalibrationRecord(
        conviction_buckets=[
            ConvictionBucket(label="low 1-5", n=4, mean_alpha_pct=-1.0),
            ConvictionBucket(label="mid 6-7", n=5, mean_alpha_pct=2.0),
            ConvictionBucket(label="high 8-10", n=6, mean_alpha_pct=7.0),
        ],
        conviction_horizon_days=90,
    )
    assert record.is_conviction_monotone is True
    block = cal.format_calibration_block(record)
    assert "Conviction vs realized alpha at 90d" in block
    assert "WARNING" not in block


def test_conviction_inversion_produces_an_explicit_warning():
    """The actionable case: confidence that doesn't pay. The prompt has to
    be told, or the ranker keeps emitting meaningless conviction scores."""
    record = CalibrationRecord(
        conviction_buckets=[
            ConvictionBucket(label="low 1-5", n=4, mean_alpha_pct=9.0),
            ConvictionBucket(label="mid 6-7", n=5, mean_alpha_pct=3.0),
            ConvictionBucket(label="high 8-10", n=6, mean_alpha_pct=-2.0),
        ],
        conviction_horizon_days=90,
    )
    assert record.is_conviction_monotone is False
    block = cal.format_calibration_block(record)
    assert "WARNING" in block
    assert "NOT ordered by realized alpha" in block


def test_single_bucket_is_not_flagged_as_inverted():
    """One bucket has no ordering to violate — don't cry wolf on a young DB."""
    record = CalibrationRecord(
        conviction_buckets=[
            ConvictionBucket(label="high 8-10", n=3, mean_alpha_pct=4.0),
        ],
    )
    assert record.is_conviction_monotone is True
    assert "WARNING" not in cal.format_calibration_block(record)


def test_conviction_buckets_are_built_from_realized_alpha():
    """End-to-end: two picks at different conviction levels, different
    outcomes, bucketed correctly against SPY."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        # Three high-conviction picks that beat SPY, three low that lag.
        for i in range(3):
            _seed_pick(
                db_path,
                ticker=f"HI{i}",
                conviction=9,
                scenarios=_BALANCED,
                age_days=120,
                rank=i + 1,
            )
        for i in range(3):
            _seed_pick(
                db_path,
                ticker=f"LO{i}",
                conviction=3,
                scenarios=_BALANCED,
                age_days=120,
                rank=i + 1,
            )
        horizon = cal._CONVICTION_HORIZON_DAYS
        start = date.today() - timedelta(days=120)

        def fake_history(ticker, _s, _e):
            if ticker == "SPY":
                return _price_frame(start, entry=400.0, at_horizon=420.0, horizon=horizon)
            if ticker.startswith("HI"):
                return _price_frame(start, entry=100.0, at_horizon=120.0, horizon=horizon)
            return _price_frame(start, entry=100.0, at_horizon=100.0, horizon=horizon)

        with patch.object(cal, "_fetch_history", side_effect=fake_history):
            record = cal.measure_calibration(db_path)

    by_label = {b.label: b for b in record.conviction_buckets}
    assert by_label["high 8-10"].n == 3
    assert by_label["low 1-5"].n == 3
    # HI: +20% vs SPY +5% → +15 alpha. LO: 0% vs +5% → -5 alpha.
    assert by_label["high 8-10"].mean_alpha_pct == pytest.approx(15.0, abs=0.5)
    assert by_label["low 1-5"].mean_alpha_pct == pytest.approx(-5.0, abs=0.5)
    assert record.is_conviction_monotone is True


# --- degenerate inputs ----------------------------------------------------


def test_empty_db_yields_an_empty_block():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        with get_session(db_path):
            pass
        record = cal.measure_calibration(db_path)
    assert record.n_scored == 0
    assert cal.format_calibration_block(record) == ""


def test_legacy_picks_without_a_forecast_are_counted_separately():
    """Rows written before the forecast columns existed can't be graded, and
    the count makes that explicit instead of shrinking the sample silently."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        run_at = (datetime.now() - timedelta(days=300)).isoformat(timespec="seconds")
        with get_session(db_path) as session:
            result = session.exec(
                text(
                    "INSERT INTO runs (run_at, kind, universe_size, survivors, "
                    "picks, opus_model, sonnet_model) "
                    "VALUES (:run_at, 'discover', 1, 1, 1, 'o', 's')"
                ),
                params={"run_at": run_at},
            )
            rid = result.lastrowid
            session.exec(
                text(
                    "INSERT INTO picks (run_id, rank, ticker, ranker_text) "
                    "VALUES (:rid, 1, 'OLD', '')"
                ),
                params={"rid": rid},
            )
        with patch.object(cal, "_fetch_history", return_value=None):
            record = cal.measure_calibration(db_path)

    assert record.n_without_forecast == 1
    assert record.n_scored == 0
