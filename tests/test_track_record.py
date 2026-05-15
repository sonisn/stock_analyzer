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

from stock_analyzer.discover import track_record as tr
from stock_analyzer.discover.persistence import connect

# --- alpha sign convention -----------------------------------------------


def _spy_quote(start: float, end: float) -> tr.Quote:
    return tr.Quote(pick_price=start, measured_price=end)


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
        with connect(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO runs (run_at, kind, universe_size, survivors, "
                "picks, opus_model, sonnet_model) VALUES (?, 'rebalance', 0, 0, 0, "
                "'opus', 'sonnet')",
                (now,),
            )
            run_id = cur.lastrowid
            for ticker, verdict in [
                ("TSLA", "SELL"),
                ("AAPL", "TRIM"),
                ("GOOGL", "HOLD"),
                ("MSFT", None),
            ]:
                conn.execute(
                    "INSERT INTO holdings_reviews (run_id, ticker, verdict, "
                    "confidence, review_text) VALUES (?, ?, ?, 7, '')",
                    (run_id, ticker, verdict),
                )
            sells = tr._fetch_recent_sells(conn, lookback_days=180)
    tickers = {t for t, _, _ in sells}
    assert tickers == {"TSLA", "AAPL"}  # SELL + TRIM, no HOLD, no NULL


def test_delisted_tickers_are_dropped_not_pending():
    """yfinance returns no price for delisted tickers (e.g. MCAH after a
    SPAC unwind). The old behavior bucketed those as 'pending' forever,
    inflating the count with zombie entries. New behavior drops them
    from the output entirely so the user only sees real decisions."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "discover.db")
        old = (datetime.now() - timedelta(days=60)).isoformat(timespec="seconds")
        with connect(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO runs (run_at, kind, universe_size, survivors, "
                "picks, opus_model, sonnet_model) VALUES (?, 'discover', 1, 1, "
                "2, 'o', 's')",
                (old,),
            )
            rid = cur.lastrowid
            conn.execute(
                "INSERT INTO picks (run_id, rank, ticker, ranker_text, "
                "bear_case_text, allocation_text) VALUES (?, 1, 'NVDA', '', '', '')",
                (rid,),
            )
            conn.execute(
                "INSERT INTO picks (run_id, rank, ticker, ranker_text, "
                "bear_case_text, allocation_text) VALUES (?, 2, 'MCAH', '', '', '')",
                (rid,),
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
