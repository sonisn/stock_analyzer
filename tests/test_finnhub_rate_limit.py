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
