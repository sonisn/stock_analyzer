"""Finnhub 429 handling.

The free tier is 60 calls/min and the pipeline paces at 55, but the quota
is per key — another process, or a burst across a minute boundary, still
trips it. Dropping the ticker on a 429 is how a run silently ends up with
half its Finnhub signals missing, so a rate limit must be retried.
"""

from __future__ import annotations

import pytest

from stock_analyzer.data import finnhub as fh


class _RateLimited(Exception):
    status_code = 429

    def __str__(self) -> str:  # what finnhub-python raises looks like this
        return "FinnhubAPIException(status_code: 429): API limit reached"


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fh, "_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(fh, "_BASE_COOLDOWN", 0.0)
    monkeypatch.setattr(fh, "_cooldown_until", 0.0)


def test_rate_limited_call_is_retried_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _RateLimited()
        return {"ok": True}

    assert fh._safe_call("price_target", "NVDA", flaky) == {"ok": True}
    assert calls["n"] == 3


def test_rate_limited_call_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fh, "_MAX_ATTEMPTS", 2)
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise _RateLimited()

    assert fh._safe_call("price_target", "NVDA", always) is None
    assert calls["n"] == 2


def test_non_rate_limit_errors_are_not_retried():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("no data for symbol")

    assert fh._safe_call("price_target", "XYZ", broken) is None
    assert calls["n"] == 1


def test_rate_limit_sets_a_shared_cooldown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fh, "_BASE_COOLDOWN", 5.0)
    before = fh._cooldown_until
    fh._cool_down(1)
    assert fh._cooldown_until > before
    fh._cooldown_until = 0.0


def test_fetch_quote_reads_price_and_skips_uncovered_symbols(monkeypatch: pytest.MonkeyPatch):
    class _Client:
        def quote(self, symbol):
            return {"c": 265.63, "pc": 260.0} if symbol == "BE" else {"c": 0, "pc": 0}

    monkeypatch.setattr(fh, "_client", lambda: _Client())
    assert fh.fetch_quote("be") == {"price": 265.63, "prev_close": 260.0}
    assert fh.fetch_quote("SPAXX") is None

    monkeypatch.setattr(fh, "_client", lambda: None)  # no API key
    assert fh.fetch_quote("BE") is None


def test_daily_email_price_falls_back_to_finnhub_when_yahoo_has_none(monkeypatch):
    from stock_analyzer.data import ticker, yf_gateway

    monkeypatch.setattr(yf_gateway, "ticker_call", lambda *a, **k: k.get("default"))
    monkeypatch.setattr(yf_gateway, "call", lambda *a, **k: None)
    monkeypatch.setattr(yf_gateway, "daily_bars", lambda *a, **k: None)
    monkeypatch.setattr(fh, "fetch_quote", lambda s: {"price": 100.0, "prev_close": 98.0})

    data = ticker.fetch_ticker_data("BE")
    assert data["price_value"] == 100.0
