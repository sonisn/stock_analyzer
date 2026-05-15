"""Track-record direction-aware alpha — math we just shipped.

A wrong sign on the sell-alpha flip would mean the system claims its
WORST sell calls were its best (and vice versa). These tests pin the
sign convention and the buy/sell scoring split.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import text

from stock_analyzer.db.session import get_session
from stock_analyzer.db.track_record import fetch_recent_sell_runs
from stock_analyzer.discover import track_record as tr

# --- alpha sign convention -----------------------------------------------


def _spy_quote(start: float, end: float) -> tr.Quote:
    return tr.Quote(pick_price=start, measured_price=end)


def _empty_dir() -> tr.DirectionStats:
    return tr.DirectionStats(
        n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, sharpe=None,
    )


def test_buy_alpha_is_stock_minus_spy_when_stock_beats_spy():
    """Stock +20%, SPY +5% → buy alpha = +15% (wise buy)."""
    with patch.object(
        tr, "_fetch_quote",
        return_value=tr.Quote(pick_price=100.0, measured_price=120.0),
    ):
        result = tr._score_pick(
            "NVDA", "2026-02-01", age_days=60,
            spy_quote=_spy_quote(400.0, 420.0),
            direction="buy",
        )
    assert result.pick_return_pct == pytest.approx(20.0)
    assert result.spy_return_pct == pytest.approx(5.0)
    assert result.alpha_pct == pytest.approx(15.0)
    assert result.is_mature is True


def test_sell_alpha_sign_flips_so_underperforming_stock_is_a_win():
    """Stock -15%, SPY +5% — raw alpha = -20% but we said SELL, so the
    call was right. Sign-flip: sell alpha = +20% (wise sell)."""
    with patch.object(
        tr, "_fetch_quote",
        return_value=tr.Quote(pick_price=200.0, measured_price=170.0),
    ):
        result = tr._score_pick(
            "TSLA", "2026-02-01", age_days=60,
            spy_quote=_spy_quote(400.0, 420.0),
            direction="sell",
        )
    assert result.pick_return_pct == pytest.approx(-15.0)
    assert result.spy_return_pct == pytest.approx(5.0)
    # Raw alpha is -20; sell flips it to +20.
    assert result.alpha_pct == pytest.approx(20.0)


def test_sell_alpha_is_negative_when_stock_outperforms_spy():
    """If we said SELL and the stock then ripped +20% vs SPY +5%, that's
    a BAD sell call. Sell alpha must be negative."""
    with patch.object(
        tr, "_fetch_quote",
        return_value=tr.Quote(pick_price=100.0, measured_price=120.0),
    ):
        result = tr._score_pick(
            "AAPL", "2026-02-01", age_days=60,
            spy_quote=_spy_quote(400.0, 420.0),
            direction="sell",
        )
    assert result.alpha_pct == pytest.approx(-15.0)  # bad sell — we missed +20% upside


def test_pending_when_age_below_mature_threshold():
    """Decisions younger than _MIN_AGE_DAYS don't count toward stats —
    they show as pending with live return only."""
    with patch.object(
        tr, "_fetch_quote",
        return_value=tr.Quote(pick_price=100.0, measured_price=105.0),
    ):
        result = tr._score_pick(
            "NVDA", "2026-05-01", age_days=10,
            spy_quote=_spy_quote(400.0, 410.0),
            direction="buy",
        )
    assert result.is_mature is False


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


def test_delisted_tickers_are_dropped_not_pending():
    """yfinance returns no price for delisted tickers (e.g. MCAH after a
    SPAC unwind). The old behavior bucketed those as 'pending' forever,
    inflating the count with zombie entries. New behavior drops them
    from the output entirely so the user only sees real decisions."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        old = (datetime.now() - timedelta(days=60)).isoformat(timespec="seconds")
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
            session.exec(
                text(
                    "INSERT INTO picks (run_id, rank, ticker, ranker_text, "
                    "bear_case_text, allocation_text) VALUES (:rid, 1, 'NVDA', '', '', '')"
                ),
                params={"rid": rid},
            )
            session.exec(
                text(
                    "INSERT INTO picks (run_id, rank, ticker, ranker_text, "
                    "bear_case_text, allocation_text) VALUES (:rid, 2, 'MCAH', '', '', '')"
                ),
                params={"rid": rid},
            )

        def fake_quote(ticker, pick_date, age_days):
            # MCAH is delisted — yfinance returns empty quote.
            if ticker == "MCAH":
                return tr.Quote(pick_price=None, measured_price=None)
            if ticker == "SPY":
                return tr.Quote(pick_price=400.0, measured_price=420.0)
            return tr.Quote(pick_price=100.0, measured_price=120.0)  # NVDA

        with patch.object(tr, "_fetch_quote", side_effect=fake_quote):
            record = tr.measure_track_record(db_path)
    tickers = {p.ticker for p in record.picks + record.pending}
    assert "MCAH" not in tickers           # delisted dropped entirely
    assert "NVDA" in tickers               # measurable ticker retained
    assert record.n_picks_total == 1       # MCAH not counted


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


# --- Sharpe sample-size and zero-stdev guards ----------------------------


def test_sharpe_returns_none_below_n5():
    """Sharpe is None when the mature sample has fewer than 5 entries."""
    alphas = [1.0, 2.0, 3.0, 4.0]
    assert tr._sharpe(alphas) is None


def test_sharpe_computes_at_n5():
    """Sharpe is mean/stdev once the sample reaches 5 entries."""
    alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = pytest.approx(
        sum(alphas) / 5 / 1.5811388300841898  # statistics.stdev(1..5)
    )
    assert tr._sharpe(alphas) == expected


def test_sharpe_returns_none_when_stdev_essentially_zero():
    """Sharpe is None when every alpha is identical (stdev < 0.001)."""
    alphas = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert tr._sharpe(alphas) is None


# --- hold / trim alpha sign conventions ----------------------------------


def test_hold_alpha_uses_buy_sign_convention():
    """HOLD vindicated when the stock outperforms SPY — same sign as BUY."""
    with patch.object(
        tr, "_fetch_quote",
        return_value=tr.Quote(pick_price=100.0, measured_price=110.0),
    ):
        result = tr._score_pick(
            "AAPL", "2026-02-01", age_days=60,
            spy_quote=_spy_quote(400.0, 408.0),
            direction="hold",
        )
    # Stock +10%, SPY +2% → HOLD alpha = +8% (HOLD was vindicated).
    assert result.pick_return_pct == pytest.approx(10.0)
    assert result.spy_return_pct == pytest.approx(2.0)
    assert result.alpha_pct == pytest.approx(8.0)
    assert result.direction == "hold"


def test_trim_alpha_sign_flips_so_underperforming_stock_is_a_win():
    """TRIM right when the stock underperforms SPY — same sign-flip as SELL."""
    with patch.object(
        tr, "_fetch_quote",
        return_value=tr.Quote(pick_price=100.0, measured_price=88.0),
    ):
        result = tr._score_pick(
            "INTC", "2026-02-01", age_days=60,
            spy_quote=_spy_quote(400.0, 408.0),
            direction="trim",
        )
    # Stock -12%, SPY +2% → raw alpha -14% → TRIM flips → +14% (wise trim).
    assert result.pick_return_pct == pytest.approx(-12.0)
    assert result.spy_return_pct == pytest.approx(2.0)
    assert result.alpha_pct == pytest.approx(14.0)
    assert result.direction == "trim"


# --- model breakdown -----------------------------------------------------


def test_compute_model_breakdown_drops_models_below_n3():
    """Models with fewer than 3 mature decisions are dropped (still
    counted in the overall buy aggregate, just not surfaced as a row)."""
    picks = [
        tr.PickReturn(
            ticker=f"T{i}", pick_date="2026-02-01", age_days=60,
            direction="buy",
            pick_price=100.0, measured_price=110.0,
            pick_return_pct=10.0, spy_return_pct=2.0, alpha_pct=8.0,
            is_mature=True,
        )
        for i in range(4)
    ]
    # 3 picks on opus-4-7, 1 on opus-4-6.
    ticker_model = {"T0": "opus-4-7", "T1": "opus-4-7", "T2": "opus-4-7", "T3": "opus-4-6"}
    out = tr._compute_model_breakdown(picks, ticker_model)
    assert len(out) == 1
    assert out[0].opus_model == "opus-4-7"
    assert out[0].n_mature == 3
    assert out[0].mean_alpha_pct == pytest.approx(8.0)


def test_compute_model_breakdown_groups_none_as_unknown():
    """Picks whose opus_model is None bucket under 'unknown'."""
    picks = [
        tr.PickReturn(
            ticker=f"X{i}", pick_date="2026-02-01", age_days=60,
            direction="buy",
            pick_price=100.0, measured_price=104.0,
            pick_return_pct=4.0, spy_return_pct=2.0, alpha_pct=2.0,
            is_mature=True,
        )
        for i in range(3)
    ]
    ticker_model = {"X0": None, "X1": None, "X2": None}
    out = tr._compute_model_breakdown(picks, ticker_model)
    assert len(out) == 1
    assert out[0].opus_model == "unknown"
    assert out[0].n_mature == 3


def test_compute_model_breakdown_sorted_by_mean_alpha_desc():
    """Strongest model listed first."""
    picks = [
        tr.PickReturn(
            ticker=f"A{i}", pick_date="2026-02-01", age_days=60,
            direction="buy", pick_price=100.0, measured_price=120.0,
            pick_return_pct=20.0, spy_return_pct=2.0, alpha_pct=18.0,
            is_mature=True,
        ) for i in range(3)
    ] + [
        tr.PickReturn(
            ticker=f"B{i}", pick_date="2026-02-01", age_days=60,
            direction="buy", pick_price=100.0, measured_price=104.0,
            pick_return_pct=4.0, spy_return_pct=2.0, alpha_pct=2.0,
            is_mature=True,
        ) for i in range(3)
    ]
    ticker_model = {f"A{i}": "weak" for i in range(3)}
    ticker_model.update({f"B{i}": "strong" for i in range(3)})
    # Intentionally mis-labeled to make sure sorting is by alpha not name.
    # A* picks have +18% alpha but are labeled "weak"; B* picks +2% labeled "strong".
    # After sorting, the "weak" model should appear FIRST (alpha 18 > alpha 2).
    out = tr._compute_model_breakdown(picks, ticker_model)
    assert len(out) == 2
    assert out[0].opus_model == "weak"
    assert out[0].mean_alpha_pct == pytest.approx(18.0)
    assert out[1].opus_model == "strong"
    assert out[1].mean_alpha_pct == pytest.approx(2.0)


# --- format_track_record_block rendering ---------------------------------


def test_block_renders_all_directions_with_data():
    """Every direction with n_mature >= 1 renders one line; model_breakdown
    renders when non-empty; first Sharpe label is spelled out."""
    buy = tr.DirectionStats(
        n_mature=6, n_pending=0,
        mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
        winners=4, losers=1, flats=1, sharpe=0.42,
    )
    hold = tr.DirectionStats(
        n_mature=3, n_pending=0,
        mean_return_pct=4.0, mean_spy_return_pct=3.0, mean_alpha_pct=1.0,
        winners=2, losers=1, flats=0, sharpe=None,
    )
    rec = tr.TrackRecord(
        n_picks_total=9, n_mature=9, n_pending=0,
        mean_return_pct=7.0, mean_spy_return_pct=2.5, mean_alpha_pct=4.5,
        winners=6, losers=2, flats=1, overall_sharpe=0.30,
        buy_stats=buy, hold_stats=hold, trim_stats=_empty_dir(), sell_stats=_empty_dir(),
        model_breakdown=[
            tr.ModelStats(opus_model="claude-opus-4-7", n_mature=4, mean_alpha_pct=10.0, sharpe=0.55),
        ],
        picks=[], pending=[],
    )
    out = tr.format_track_record_block(rec)
    assert "Buy track record:" in out
    assert "Hold track record:" in out
    assert "Trim track record:" not in out
    assert "Sell track record:" not in out
    assert "Model breakdown:" in out
    assert "claude-opus-4-7 (4 picks, +10.0%)" in out
    # First Sharpe label is the full one; subsequent lines just say Sharpe.
    assert "Sharpe (per-decision)" in out


def test_block_omits_directions_with_zero_mature():
    """Directions with n_mature == 0 are completely absent from the block."""
    buy = tr.DirectionStats(
        n_mature=3, n_pending=0,
        mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
        winners=2, losers=0, flats=1, sharpe=None,
    )
    rec = tr.TrackRecord(
        n_picks_total=3, n_mature=3, n_pending=0,
        mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
        winners=2, losers=0, flats=1, overall_sharpe=None,
        buy_stats=buy, hold_stats=_empty_dir(), trim_stats=_empty_dir(), sell_stats=_empty_dir(),
        model_breakdown=[],
        picks=[], pending=[],
    )
    out = tr.format_track_record_block(rec)
    assert "Buy track record:" in out
    assert "Hold track record:" not in out
    assert "Trim track record:" not in out
    assert "Sell track record:" not in out
    assert "Model breakdown:" not in out


def test_block_renders_sharpe_na_when_none():
    """Sharpe None renders as either 'n/a (n<5)' or 'n/a (flat)'."""
    small_sample = tr.DirectionStats(
        n_mature=3, n_pending=0,
        mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
        winners=2, losers=0, flats=1, sharpe=None,
    )
    flat_sample = tr.DirectionStats(
        n_mature=5, n_pending=0,
        mean_return_pct=5.0, mean_spy_return_pct=2.0, mean_alpha_pct=3.0,
        winners=5, losers=0, flats=0, sharpe=None,
    )
    rec = tr.TrackRecord(
        n_picks_total=8, n_mature=8, n_pending=0,
        mean_return_pct=7.0, mean_spy_return_pct=2.0, mean_alpha_pct=5.0,
        winners=7, losers=0, flats=1, overall_sharpe=None,
        buy_stats=small_sample, hold_stats=flat_sample,
        trim_stats=_empty_dir(), sell_stats=_empty_dir(),
        model_breakdown=[], picks=[], pending=[],
    )
    out = tr.format_track_record_block(rec)
    assert "n/a (n<5)" in out
    assert "n/a (flat)" in out


def test_block_returns_empty_when_no_decisions():
    """Empty record renders as empty string (prompt-context gets nothing)."""
    rec = tr.TrackRecord(
        n_picks_total=0, n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, overall_sharpe=None,
        buy_stats=_empty_dir(), hold_stats=_empty_dir(),
        trim_stats=_empty_dir(), sell_stats=_empty_dir(),
        model_breakdown=[], picks=[], pending=[],
    )
    assert tr.format_track_record_block(rec) == ""
