"""Tests for WRITE_CALL outcome scoring."""
from __future__ import annotations

from unittest.mock import patch

from stock_analyzer.discover.track_record import score_covered_call


def test_expired_otm_keeps_full_premium():
    with patch(
        "stock_analyzer.discover.track_record._spot_at",
        return_value=240.0,
    ):
        out = score_covered_call(
            ticker="NVDA", strike=260.0, expiry="2026-06-20",
            contracts=3, est_premium_per_share=2.40,
        )
    assert out["outcome"] == "EXPIRED_OTM"
    assert out["pnl_usd"] == 3 * 2.40 * 100
    assert out["opportunity_cost_usd"] == 0.0


def test_assigned_records_opportunity_cost():
    with patch(
        "stock_analyzer.discover.track_record._spot_at",
        return_value=280.0,
    ):
        out = score_covered_call(
            ticker="NVDA", strike=260.0, expiry="2026-06-20",
            contracts=3, est_premium_per_share=2.40,
        )
    assert out["outcome"] == "ASSIGNED"
    assert out["pnl_usd"] == 720 - 6000
    assert out["opportunity_cost_usd"] == 6000


def test_missing_spot_returns_unknown():
    with patch(
        "stock_analyzer.discover.track_record._spot_at",
        return_value=None,
    ):
        out = score_covered_call(
            ticker="X", strike=100.0, expiry="2026-06-20",
            contracts=1, est_premium_per_share=1.0,
        )
    assert out["outcome"] == "UNKNOWN"
    assert out["pnl_usd"] is None


def test_spot_at_first_of_month_picks_up_prior_close():
    """Regression: previously `end.replace(day=end.day - 5)` produced a
    zero-day window for first-of-month expiries, returning None."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    from stock_analyzer.discover.track_record import _spot_at

    real_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = real_df
    with patch("yfinance.Ticker", return_value=fake_ticker):
        out = _spot_at("X", "2026-06-01")
    assert out == 102.0
    # The fix uses timedelta(days=7), so the start arg should be 2026-05-25.
    call_kwargs = fake_ticker.history.call_args.kwargs
    assert call_kwargs["start"] == "2026-05-25"
    assert call_kwargs["end"] == "2026-06-01"
