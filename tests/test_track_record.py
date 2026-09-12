"""Track-record measurement math — the number that feeds the ranker prompt.

Four properties are pinned here because each one, when wrong, corrupts the
feedback the LLM reasons about:

  1. Direction-aware alpha sign. A wrong flip would mean the system claims
     its WORST sell calls were its best.
  2. One horizon per aggregate. A 30-day outcome must never be averaged
     into the same mean as a 90-day one.
  3. Unmeasurable decisions are reported, not dropped — dropping delistings
     removes the left tail and inflates measured alpha.
  4. Beta-adjusted alpha, estimated on pre-decision data only, so market
     exposure isn't credited as stock-selection skill.
"""

from __future__ import annotations

import os
import statistics
import tempfile
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import text

from stock_analyzer.db.session import get_session
from stock_analyzer.db.track_record import fetch_recent_sell_runs
from stock_analyzer.discover import track_record as tr

# --- synthetic price frames ----------------------------------------------


def _flat_then_path(
    pick_date: date,
    *,
    entry: float,
    at_90: float,
    pre_days: int = 200,
    post_days: int = 120,
) -> pd.DataFrame:
    """A daily frame that is exactly `entry` on pick_date and moves linearly
    to `at_90` by pick_date + 90 days (so a 30-day read is one third of the
    way). Pre-decision bars are flat, which makes beta inestimable — tests
    that care about beta build their own path.
    """
    start = pick_date - timedelta(days=pre_days)
    end = pick_date + timedelta(days=post_days)
    idx = pd.date_range(start, end, freq="D")
    per_day = (at_90 - entry) / 90.0
    closes = [
        entry if ts.date() <= pick_date else entry + per_day * (ts.date() - pick_date).days
        for ts in idx
    ]
    return pd.DataFrame({"Close": closes, "High": closes}, index=idx)


def _beta_path(
    pick_date: date,
    *,
    beta: float,
    entry: float,
    at_90: float,
    pre_days: int = 200,
) -> pd.DataFrame:
    """Pre-decision bars whose daily returns are exactly `beta` times SPY's
    (see `_spy_beta_path`), then the same linear forward path as above.

    Built so cov(t, s) / var(s) == beta exactly, which lets the test assert
    the beta estimate rather than merely its sign.
    """
    start = pick_date - timedelta(days=pre_days)
    idx = pd.date_range(start, pick_date + timedelta(days=120), freq="D")
    closes: list[float] = []
    price = entry
    pre = [ts for ts in idx if ts.date() < pick_date]
    # Walk backward from `entry` so the bar ON pick_date is exactly `entry`.
    rets = [(0.005 if i % 2 == 0 else -0.005) * beta for i in range(len(pre))]
    path = [entry]
    for r in reversed(rets):
        path.append(path[-1] / (1 + r))
    path = list(reversed(path[1:]))
    per_day = (at_90 - entry) / 90.0
    for ts in idx:
        if ts.date() < pick_date:
            closes.append(path[len([c for c in closes])])
        elif ts.date() == pick_date:
            price = entry
            closes.append(price)
        else:
            closes.append(entry + per_day * (ts.date() - pick_date).days)
    return pd.DataFrame({"Close": closes, "High": closes}, index=idx)


def _spy_beta_path(
    pick_date: date, *, entry: float, at_90: float, pre_days: int = 200
) -> pd.DataFrame:
    return _beta_path(pick_date, beta=1.0, entry=entry, at_90=at_90, pre_days=pre_days)


def _score(
    ticker: str,
    pick_date: date,
    age_days: int,
    direction: str,
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> list:
    rows, bad = tr._score_decision(
        tr._Decision(
            ticker=ticker,
            pick_date=pick_date.isoformat(),
            age_days=age_days,
            direction=direction,  # type: ignore[arg-type]
        ),
        ticker_df,
        spy_df,
    )
    assert bad is None, f"unexpectedly unmeasurable: {bad}"
    return rows


def _at_90(rows: list):
    matches = [r for r in rows if r.horizon_days == 90]
    assert len(matches) == 1
    return matches[0]


def _empty_dir() -> tr.DirectionStats:
    return tr._empty_direction()


# --- alpha sign convention -----------------------------------------------


def test_buy_alpha_is_stock_minus_spy_when_stock_beats_spy():
    """Stock +20%, SPY +5% → buy alpha = +15% (wise buy)."""
    pd_ = date(2026, 2, 1)
    rows = _score(
        "NVDA",
        pd_,
        120,
        "buy",
        _flat_then_path(pd_, entry=100.0, at_90=120.0),
        _flat_then_path(pd_, entry=400.0, at_90=420.0),
    )
    row = _at_90(rows)
    assert row.pick_return_pct == pytest.approx(20.0)
    assert row.spy_return_pct == pytest.approx(5.0)
    assert row.alpha_pct == pytest.approx(15.0)
    assert row.is_mature is True
    assert row.horizon_days == 90


def test_sell_alpha_sign_flips_so_underperforming_stock_is_a_win():
    """Stock -15%, SPY +5% — raw alpha = -20% but we said SELL, so the
    call was right. Sign-flip: sell alpha = +20% (wise sell)."""
    pd_ = date(2026, 2, 1)
    rows = _score(
        "TSLA",
        pd_,
        120,
        "sell",
        _flat_then_path(pd_, entry=200.0, at_90=170.0),
        _flat_then_path(pd_, entry=400.0, at_90=420.0),
    )
    row = _at_90(rows)
    assert row.pick_return_pct == pytest.approx(-15.0)
    assert row.spy_return_pct == pytest.approx(5.0)
    assert row.alpha_pct == pytest.approx(20.0)


def test_sell_alpha_is_negative_when_stock_outperforms_spy():
    """If we said SELL and the stock then ripped +20% vs SPY +5%, that's
    a BAD sell call. Sell alpha must be negative."""
    pd_ = date(2026, 2, 1)
    rows = _score(
        "AAPL",
        pd_,
        120,
        "sell",
        _flat_then_path(pd_, entry=100.0, at_90=120.0),
        _flat_then_path(pd_, entry=400.0, at_90=420.0),
    )
    assert _at_90(rows).alpha_pct == pytest.approx(-15.0)


def test_hold_alpha_uses_buy_sign_convention():
    """HOLD vindicated when the stock outperforms SPY — same sign as BUY."""
    pd_ = date(2026, 2, 1)
    rows = _score(
        "AAPL",
        pd_,
        120,
        "hold",
        _flat_then_path(pd_, entry=100.0, at_90=110.0),
        _flat_then_path(pd_, entry=400.0, at_90=408.0),
    )
    row = _at_90(rows)
    assert row.pick_return_pct == pytest.approx(10.0)
    assert row.spy_return_pct == pytest.approx(2.0)
    assert row.alpha_pct == pytest.approx(8.0)
    assert row.direction == "hold"


def test_trim_alpha_sign_flips_so_underperforming_stock_is_a_win():
    """TRIM right when the stock underperforms SPY — same sign-flip as SELL."""
    pd_ = date(2026, 2, 1)
    rows = _score(
        "INTC",
        pd_,
        120,
        "trim",
        _flat_then_path(pd_, entry=100.0, at_90=88.0),
        _flat_then_path(pd_, entry=400.0, at_90=408.0),
    )
    row = _at_90(rows)
    assert row.pick_return_pct == pytest.approx(-12.0)
    assert row.spy_return_pct == pytest.approx(2.0)
    assert row.alpha_pct == pytest.approx(14.0)
    assert row.direction == "trim"


# --- horizon discipline (the mixed-window bug) ---------------------------


def test_one_decision_is_scored_separately_at_each_finished_horizon():
    """A 120-day-old decision yields a 30d row AND a 90d row, each measured
    to its own anniversary — never one blended number.

    The old code measured every decision older than 14 days to
    min(pick+90d, today) and averaged the results together, so a 15-day
    outcome and a 90-day outcome landed in the same mean.
    """
    pd_ = date(2026, 2, 1)
    rows = _score(
        "NVDA",
        pd_,
        120,
        "buy",
        _flat_then_path(pd_, entry=100.0, at_90=190.0),  # +90% over 90d
        _flat_then_path(pd_, entry=400.0, at_90=400.0),  # SPY flat
    )
    by_h = {r.horizon_days: r for r in rows}
    assert sorted(by_h) == [30, 90]
    # Linear path: one third of the way at 30 days.
    assert by_h[30].pick_return_pct == pytest.approx(30.0)
    assert by_h[90].pick_return_pct == pytest.approx(90.0)
    assert by_h[30].measured_date == (pd_ + timedelta(days=30)).isoformat()
    assert by_h[90].measured_date == (pd_ + timedelta(days=90)).isoformat()


def test_decision_too_young_for_long_horizon_only_scores_short_one():
    """At 45 days old only the 30d window has finished."""
    pd_ = date.today() - timedelta(days=45)
    rows = _score(
        "NVDA",
        pd_,
        45,
        "buy",
        _flat_then_path(pd_, entry=100.0, at_90=130.0),
        _flat_then_path(pd_, entry=400.0, at_90=400.0),
    )
    assert [r.horizon_days for r in rows] == [30]


def test_pending_when_age_below_shortest_horizon():
    """Younger than the shortest horizon → a single live-mark row that is
    explicitly not a finished measurement."""
    pd_ = date.today() - timedelta(days=10)
    rows = _score(
        "NVDA",
        pd_,
        10,
        "buy",
        _flat_then_path(pd_, entry=100.0, at_90=130.0),
        _flat_then_path(pd_, entry=400.0, at_90=410.0),
    )
    assert len(rows) == 1
    assert rows[0].is_mature is False
    assert rows[0].horizon_days == 0


def test_horizons_aggregate_independently():
    """Two horizons of the same decisions produce two separate stat blocks
    and the 90d mean is not contaminated by the 30d reads."""
    pd_ = date(2026, 2, 1)
    rows: list = []
    for i in range(3):
        rows += _score(
            f"T{i}",
            pd_,
            120,
            "buy",
            _flat_then_path(pd_, entry=100.0, at_90=200.0),  # +100% at 90d
            _flat_then_path(pd_, entry=400.0, at_90=400.0),
        )
    thirty = tr._aggregate([r for r in rows if r.horizon_days == 30])
    ninety = tr._aggregate([r for r in rows if r.horizon_days == 90])
    assert thirty.n_mature == 3
    assert ninety.n_mature == 3
    assert thirty.mean_alpha_pct == pytest.approx(100.0 / 3, rel=1e-3)
    assert ninety.mean_alpha_pct == pytest.approx(100.0)


# --- beta adjustment -----------------------------------------------------


def test_beta_adjusted_alpha_removes_market_exposure():
    """A beta-2 name that returns +10% while SPY returns +5% has no
    selection alpha at all: 10 - 2*5 == 0, even though raw alpha is +5%.

    This is the confound the screen creates on purpose — it filters for
    names above their 200DMA with positive RS, which skews high-beta.
    """
    pd_ = date(2026, 2, 1)
    rows = _score(
        "HIBETA",
        pd_,
        120,
        "buy",
        _beta_path(pd_, beta=2.0, entry=100.0, at_90=110.0),
        _spy_beta_path(pd_, entry=400.0, at_90=420.0),
    )
    row = _at_90(rows)
    assert row.beta == pytest.approx(2.0, rel=1e-6)
    assert row.pick_return_pct == pytest.approx(10.0)
    assert row.spy_return_pct == pytest.approx(5.0)
    assert row.alpha_pct == pytest.approx(5.0)  # raw: looks skillful
    assert row.beta_adjusted_alpha_pct == pytest.approx(0.0, abs=1e-6)


def test_beta_is_none_when_pre_window_has_no_variance():
    """Flat pre-decision history → no estimable beta, and the row sits out
    the beta-adjusted mean rather than defaulting to 1.0."""
    pd_ = date(2026, 2, 1)
    rows = _score(
        "FLAT",
        pd_,
        120,
        "buy",
        _flat_then_path(pd_, entry=100.0, at_90=110.0),
        _flat_then_path(pd_, entry=400.0, at_90=420.0),
    )
    row = _at_90(rows)
    assert row.beta is None
    assert row.beta_adjusted_alpha_pct is None
    stats = tr._aggregate([row])
    assert stats.mean_beta_adjusted_alpha_pct is None
    assert stats.n_beta_adjusted == 0


def test_beta_adjusted_mean_uses_only_rows_with_a_beta():
    """Mixed sample: the beta-adjusted mean is computed over the subset that
    has a beta, and n_beta_adjusted reports that subset's size."""
    pd_ = date(2026, 2, 1)
    with_beta = _at_90(
        _score(
            "B",
            pd_,
            120,
            "buy",
            _beta_path(pd_, beta=1.0, entry=100.0, at_90=110.0),
            _spy_beta_path(pd_, entry=400.0, at_90=420.0),
        )
    )
    without = _at_90(
        _score(
            "F",
            pd_,
            120,
            "buy",
            _flat_then_path(pd_, entry=100.0, at_90=120.0),
            _flat_then_path(pd_, entry=400.0, at_90=420.0),
        )
    )
    stats = tr._aggregate([with_beta, without])
    assert stats.n_mature == 2
    assert stats.n_beta_adjusted == 1
    # beta 1, stock +10%, SPY +5% → 10 - 1*5 = +5%
    assert stats.mean_beta_adjusted_alpha_pct == pytest.approx(5.0, abs=1e-6)


# --- unmeasurable decisions are reported, not dropped --------------------


def test_missing_forward_prices_are_reported_as_unmeasurable():
    """A decision old enough to score but with no price data is a delisting
    or a bad symbol. Dropping it silently removes the worst outcomes from
    the mean, so it must surface with a reason instead."""
    rows, bad = tr._score_decision(
        tr._Decision(ticker="MCAH", pick_date="2026-02-01", age_days=120, direction="buy"),
        None,
        _flat_then_path(date(2026, 2, 1), entry=400.0, at_90=420.0),
    )
    assert rows == []
    assert bad is not None
    assert bad.reason == "no_price_data"
    assert bad.ticker == "MCAH"


def test_young_decision_with_no_data_is_too_young_not_delisted():
    """The two cases must not be conflated: a same-day pick has no elapsed
    window yet, which is not evidence of a delisting."""
    rows, bad = tr._score_decision(
        tr._Decision(
            ticker="NEW",
            pick_date=date.today().isoformat(),
            age_days=0,
            direction="buy",
        ),
        None,
        None,
    )
    assert rows == []
    assert bad is not None
    assert bad.reason == "too_young"


def test_measure_track_record_counts_unmeasurable_instead_of_hiding_it():
    """End-to-end: a delisted pick stays in n_picks_total and lands in the
    unmeasurable list; the measurable one is scored normally."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        old = (datetime.now() - timedelta(days=120)).isoformat(timespec="seconds")
        with get_session(db_path) as session:
            result = session.exec(
                text(
                    "INSERT INTO runs (run_at, kind, universe_size, survivors, "
                    "picks, opus_model, sonnet_model) VALUES (:run_at, 'discover', 1, 1, "
                    "2, 'o', 's')"
                ),
                params={"run_at": old},
            )
            rid = result.lastrowid
            for rank, ticker in ((1, "NVDA"), (2, "MCAH")):
                session.exec(
                    text(
                        "INSERT INTO picks (run_id, rank, ticker, ranker_text, "
                        "bear_case_text, allocation_text) "
                        "VALUES (:rid, :rank, :ticker, '', '', '')"
                    ),
                    params={"rid": rid, "rank": rank, "ticker": ticker},
                )

        pick_date = date.fromisoformat(old.split("T")[0])

        def fake_history(ticker, start, end):
            if ticker == "MCAH":
                return None  # delisted — no forward data
            if ticker == "SPY":
                return _flat_then_path(pick_date, entry=400.0, at_90=420.0)
            return _flat_then_path(pick_date, entry=100.0, at_90=120.0)

        with patch.object(tr, "_fetch_history", side_effect=fake_history):
            record = tr.measure_track_record(db_path)

    assert record.n_picks_total == 2  # MCAH still counted
    assert record.n_unmeasurable == 1
    assert [u.ticker for u in record.unmeasurable] == ["MCAH"]
    assert {p.ticker for p in record.picks} == {"NVDA"}
    assert record.reported_horizon_days == 90
    assert record.buy_stats.mean_alpha_pct == pytest.approx(15.0)


# --- DB query: sell pulls SELL + TRIM, skips HOLD -----------------------


def test_fetch_recent_sells_excludes_hold_includes_sell_and_trim():
    """The SQL filter is the source of truth on what counts as a sell
    signal. HOLD must not leak in."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        now = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
        with get_session(db_path) as session:
            result = session.exec(
                text(
                    "INSERT INTO runs (run_at, kind, universe_size, survivors, "
                    "picks, opus_model, sonnet_model) VALUES (:run_at, 'rebalance', 0, 0, 0, "
                    "'opus', 'sonnet')"
                ),
                params={"run_at": now},
            )
            run_id = result.lastrowid
            for ticker, verdict in [
                ("TSLA", "SELL"),
                ("AAPL", "TRIM"),
                ("GOOGL", "HOLD"),
                ("MSFT", None),
            ]:
                session.exec(
                    text(
                        "INSERT INTO holdings_reviews (run_id, ticker, verdict, "
                        "confidence, review_text) VALUES (:rid, :ticker, :verdict, 7, '')"
                    ),
                    params={"rid": run_id, "ticker": ticker, "verdict": verdict},
                )
            sells = fetch_recent_sell_runs(session, lookback_days=180)
    tickers = {ticker for _, ticker in sells}
    assert tickers == {"TSLA", "AAPL"}  # SELL + TRIM, no HOLD, no NULL


def test_dedup_oldest_keeps_first_decision_per_ticker():
    """If we said SELL on TSLA twice (run 5 + run 8), the OLDEST date
    is what the user would have acted on. Dedupe to oldest, same as
    the buy-pick logic, so re-decisions don't double-count."""
    today = datetime.now().date()
    rows = [
        # (run_at, ticker) — oldest first
        ("2026-03-01T10:00:00", "TSLA"),
        ("2026-04-01T10:00:00", "TSLA"),  # later — must be dropped
        ("2026-03-15T10:00:00", "AAPL"),
    ]
    out = tr._dedup_oldest(rows)
    pairs = {(t, d) for t, d, _ in out}
    assert pairs == {("TSLA", "2026-03-01"), ("AAPL", "2026-03-15")}
    # Sorted oldest first.
    assert [t for t, _, _ in out] == ["TSLA", "AAPL"]
    # age_days is derived from today.
    for _, decision_date, age in out:
        expected = (today - datetime.fromisoformat(decision_date).date()).days
        assert age == expected


def test_overall_dedups_a_ticker_held_and_picked():
    """A name that is both a BUY pick and a HOLD verdict belongs in both
    direction stats, but must count once in the cross-direction headline —
    otherwise the overall number double-weights whatever you hold."""
    rows = [
        tr.PickReturn(
            ticker="NVDA",
            pick_date="2026-02-01",
            age_days=120,
            direction="buy",
            horizon_days=90,
            pick_price=100.0,
            measured_price=120.0,
            pick_return_pct=20.0,
            spy_return_pct=5.0,
            alpha_pct=15.0,
            is_mature=True,
        ),
        tr.PickReturn(
            ticker="NVDA",
            pick_date="2026-03-01",
            age_days=95,
            direction="hold",
            horizon_days=90,
            pick_price=110.0,
            measured_price=120.0,
            pick_return_pct=9.1,
            spy_return_pct=5.0,
            alpha_pct=4.1,
            is_mature=True,
        ),
    ]
    deduped = tr._dedup_for_overall(rows)
    assert len(deduped) == 1
    assert deduped[0].pick_date == "2026-02-01"  # earliest kept
    # Per-direction stats keep both.
    assert tr._aggregate([r for r in rows if r.direction == "buy"]).n_mature == 1
    assert tr._aggregate([r for r in rows if r.direction == "hold"]).n_mature == 1


# --- Sharpe sample-size and zero-stdev guards ----------------------------


def test_sharpe_returns_none_below_n5():
    """Sharpe is None when the mature sample has fewer than 5 entries."""
    alphas = [1.0, 2.0, 3.0, 4.0]
    assert tr._sharpe(alphas) is None


def test_sharpe_computes_at_n5():
    alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = tr._sharpe(alphas)
    assert out is not None
    # mean 3.0 / stdev 1.5811 (sample stdev of 1..5)
    assert out == pytest.approx(3.0 / statistics.stdev(alphas), rel=1e-9)


def test_sharpe_returns_none_when_stdev_essentially_zero():
    alphas = [2.0, 2.0, 2.0, 2.0, 2.0]
    assert tr._sharpe(alphas) is None


# --- model breakdown -----------------------------------------------------


def _buy_row(ticker: str, alpha: float, ret: float) -> tr.PickReturn:
    return tr.PickReturn(
        ticker=ticker,
        pick_date="2026-02-01",
        age_days=120,
        direction="buy",
        horizon_days=90,
        pick_price=100.0,
        measured_price=100.0 + ret,
        pick_return_pct=ret,
        spy_return_pct=2.0,
        alpha_pct=alpha,
        is_mature=True,
    )


def test_compute_model_breakdown_drops_models_below_n3():
    """Models with fewer than 3 mature decisions are dropped (still
    counted in the overall buy aggregate, just not surfaced as a row)."""
    picks = [_buy_row(f"T{i}", 8.0, 10.0) for i in range(4)]
    ticker_model = {
        "T0": "opus-4-7",
        "T1": "opus-4-7",
        "T2": "opus-4-7",
        "T3": "opus-4-6",
    }
    out = tr._compute_model_breakdown(picks, ticker_model)
    assert len(out) == 1
    assert out[0].opus_model == "opus-4-7"
    assert out[0].n_mature == 3
    assert out[0].mean_alpha_pct == pytest.approx(8.0)


def test_compute_model_breakdown_groups_none_as_unknown():
    """Picks whose opus_model is None bucket under 'unknown'."""
    picks = [_buy_row(f"X{i}", 2.0, 4.0) for i in range(3)]
    out = tr._compute_model_breakdown(picks, {"X0": None, "X1": None, "X2": None})
    assert len(out) == 1
    assert out[0].opus_model == "unknown"
    assert out[0].n_mature == 3


def test_compute_model_breakdown_sorted_by_mean_alpha_desc():
    """Strongest model listed first."""
    picks = [_buy_row(f"A{i}", 18.0, 20.0) for i in range(3)]
    picks += [_buy_row(f"B{i}", 2.0, 4.0) for i in range(3)]
    ticker_model = {f"A{i}": "weak" for i in range(3)}
    ticker_model.update({f"B{i}": "strong" for i in range(3)})
    # Intentionally mis-labeled to make sure sorting is by alpha not name.
    out = tr._compute_model_breakdown(picks, ticker_model)
    assert [m.opus_model for m in out] == ["weak", "strong"]
    assert out[0].mean_alpha_pct == pytest.approx(18.0)
    assert out[1].mean_alpha_pct == pytest.approx(2.0)


# --- format_track_record_block rendering ---------------------------------


def _dir_stats(
    *, n: int, alpha: float, sharpe: float | None, beta_alpha: float | None = None
) -> tr.DirectionStats:
    return tr.DirectionStats(
        n_mature=n,
        n_pending=0,
        mean_return_pct=alpha + 2.0,
        mean_spy_return_pct=2.0,
        mean_alpha_pct=alpha,
        mean_beta_adjusted_alpha_pct=beta_alpha,
        n_beta_adjusted=n if beta_alpha is not None else 0,
        winners=n,
        losers=0,
        flats=0,
        sharpe=sharpe,
    )


def _record(
    *,
    horizons: list[tr.HorizonStats],
    reported: int,
    buy: tr.DirectionStats,
    hold: tr.DirectionStats | None = None,
    model_breakdown: list | None = None,
    picks: list | None = None,
) -> tr.TrackRecord:
    return tr.TrackRecord(
        n_picks_total=max(buy.n_mature, 1),
        n_mature=buy.n_mature,
        n_pending=0,
        reported_horizon_days=reported,
        horizons=horizons,
        n_unmeasurable=0,
        unmeasurable=[],
        mean_return_pct=buy.mean_return_pct,
        mean_spy_return_pct=buy.mean_spy_return_pct,
        mean_alpha_pct=buy.mean_alpha_pct,
        winners=buy.winners,
        losers=buy.losers,
        flats=buy.flats,
        overall_sharpe=buy.sharpe,
        buy_stats=buy,
        hold_stats=hold or _empty_dir(),
        trim_stats=_empty_dir(),
        sell_stats=_empty_dir(),
        model_breakdown=model_breakdown or [],
        picks=picks or [],
        pending=[],
    )


def _horizon(
    days: int,
    *,
    buy: tr.DirectionStats,
    hold: tr.DirectionStats | None = None,
    model_breakdown: list | None = None,
    decisions: list | None = None,
) -> tr.HorizonStats:
    return tr.HorizonStats(
        horizon_days=days,
        overall=buy,
        buy_stats=buy,
        hold_stats=hold or _empty_dir(),
        trim_stats=_empty_dir(),
        sell_stats=_empty_dir(),
        model_breakdown=model_breakdown or [],
        decisions=decisions or [],
    )


def test_block_renders_all_directions_with_data():
    """Every direction with n_mature >= 1 renders one line; model_breakdown
    renders when non-empty; first Sharpe label is spelled out."""
    buy = _dir_stats(n=6, alpha=8.0, sharpe=0.42, beta_alpha=3.0)
    hold = _dir_stats(n=3, alpha=1.0, sharpe=None)
    models = [
        tr.ModelStats(
            opus_model="claude-opus-4-7",
            n_mature=4,
            mean_alpha_pct=10.0,
            sharpe=0.55,
        ),
    ]
    rec = _record(
        horizons=[_horizon(90, buy=buy, hold=hold, model_breakdown=models)],
        reported=90,
        buy=buy,
        hold=hold,
        model_breakdown=models,
    )
    out = tr.format_track_record_block(rec)
    assert "Measured over 90 days" in out
    assert "Buy: 6 scored" in out
    assert "Hold: 3 scored" in out
    assert "Trim:" not in out
    assert "Sell:" not in out
    assert "Model breakdown:" in out
    assert "claude-opus-4-7 (4 picks, +10.0%)" in out
    assert "Sharpe (per-decision)" in out
    # Beta-adjusted alpha travels with raw alpha wherever it's shown.
    assert "beta-adj +3.0%" in out


def test_block_labels_each_horizon_separately():
    """Two horizons render as two labeled blocks — the reader can never
    mistake a 30-day number for a 90-day one."""
    short = _dir_stats(n=8, alpha=3.0, sharpe=0.2, beta_alpha=1.0)
    long_ = _dir_stats(n=5, alpha=9.0, sharpe=0.5, beta_alpha=4.0)
    rec = _record(
        horizons=[_horizon(30, buy=short), _horizon(90, buy=long_)],
        reported=90,
        buy=long_,
    )
    out = tr.format_track_record_block(rec)
    assert "Measured over 30 days" in out
    assert "Measured over 90 days" in out
    assert "Buy: 8 scored" in out
    assert "Buy: 5 scored" in out


def test_block_omits_directions_with_zero_mature():
    """Directions with n_mature == 0 are completely absent from the block."""
    buy = _dir_stats(n=3, alpha=8.0, sharpe=None)
    rec = _record(horizons=[_horizon(90, buy=buy)], reported=90, buy=buy)
    out = tr.format_track_record_block(rec)
    assert "Buy: 3 scored" in out
    assert "Hold:" not in out
    assert "Trim:" not in out
    assert "Sell:" not in out
    assert "Model breakdown:" not in out


def test_block_renders_sharpe_na_when_none():
    """Sharpe None renders as either 'n/a (n<5)' or 'n/a (flat)'."""
    small_sample = _dir_stats(n=3, alpha=8.0, sharpe=None)
    flat_sample = _dir_stats(n=5, alpha=3.0, sharpe=None)
    rec = _record(
        horizons=[_horizon(90, buy=small_sample, hold=flat_sample)],
        reported=90,
        buy=small_sample,
        hold=flat_sample,
    )
    out = tr.format_track_record_block(rec)
    assert "n/a (n<5)" in out
    assert "n/a (flat)" in out


def test_block_returns_empty_when_no_decisions():
    """Empty record renders as empty string (prompt-context gets nothing)."""
    assert tr.format_track_record_block(tr._empty_record()) == ""


def test_summary_names_the_horizon_it_reports():
    """The one-line summary must say which window it measured, so the
    number is never implicitly a blend."""
    buy = _dir_stats(n=6, alpha=8.0, sharpe=0.42, beta_alpha=3.0)
    rec = _record(horizons=[_horizon(90, buy=buy)], reported=90, buy=buy)
    out = tr.format_track_record_summary(rec)
    assert "90d horizon" in out
    assert "Buy 6 scored" in out
    assert "beta-adj +3.0%" in out
