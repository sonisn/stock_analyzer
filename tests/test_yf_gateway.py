"""The yfinance gateway: pacing, retries, negative caching, fan-out.

These are the guarantees the whole pipeline leans on to stay under
Yahoo's throttle, so they are tested against fakes rather than the
network (the suite blocks sockets anyway).
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from yfinance.exceptions import YFRateLimitError

from stock_analyzer.data import yf_gateway


@pytest.fixture(autouse=True)
def _fast_pacer(monkeypatch: pytest.MonkeyPatch):
    """Keep the tests quick: no real cooldown sleeps."""
    monkeypatch.setattr(yf_gateway, "BASE_COOLDOWN_SECONDS", 0)
    yf_gateway.reset()


def test_call_returns_result_and_counts_it():
    assert yf_gateway.call(lambda: 42, what="answer") == 42
    assert yf_gateway.stats()["calls"] == 1


def test_rate_limit_is_retried_then_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise YFRateLimitError()
        return "data"

    assert yf_gateway.call(flaky, symbol="NVDA", what="test") == "data"
    assert attempts["n"] == 2
    assert yf_gateway.stats()["rate_limited"] == 1


def test_rate_limit_halves_the_rate():
    before = yf_gateway._pacer.rate_per_min

    def always_limited():
        raise YFRateLimitError()

    yf_gateway.call(always_limited, symbol="NVDA", what="test", attempts=2)
    assert yf_gateway._pacer.rate_per_min < before


def test_rate_limit_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_limited():
        calls["n"] += 1
        raise YFRateLimitError()

    assert yf_gateway.call(always_limited, symbol="NVDA", what="test", attempts=3) is None
    assert calls["n"] == 3


def test_missing_symbol_is_not_retried_and_is_remembered():
    calls = {"n": 0}

    def missing():
        calls["n"] += 1
        raise RuntimeError("$SPAXX: possibly delisted; no price data found")

    assert yf_gateway.call(missing, symbol="SPAXX", what="test") is None
    assert calls["n"] == 1, "a 404 must not be retried"
    assert yf_gateway.is_unavailable("SPAXX")

    # Second request for the same symbol never reaches the network.
    assert yf_gateway.call(missing, symbol="SPAXX", what="test") is None
    assert calls["n"] == 1


def test_no_fundamentals_404_marks_symbol_unavailable():
    def missing():
        raise RuntimeError('404 {"error": "No fundamentals data found for symbol: IBIT"}')

    yf_gateway.call(missing, symbol="IBIT", what="test")
    assert yf_gateway.is_unavailable("IBIT")


def test_unexpected_error_returns_default_without_marking_symbol():
    def boom():
        raise RuntimeError("connection reset by peer")

    assert yf_gateway.call(boom, symbol="NVDA", what="test", default={}) == {}
    assert not yf_gateway.is_unavailable("NVDA")
    assert yf_gateway.stats()["failed"] == 1


def test_ticker_instances_are_shared_across_callers():
    fake = MagicMock()
    with patch("stock_analyzer.data.yf_gateway.yf.Ticker", return_value=fake) as ctor:
        yf_gateway.ticker_call("NVDA", "info", lambda t: t.info)
        yf_gateway.ticker_call("NVDA", "cashflow", lambda t: t.quarterly_cashflow)
    assert ctor.call_count == 1, "the second caller must reuse the cached Ticker"


def test_ticker_cache_expires(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yf_gateway, "TICKER_CACHE_TTL", 0)
    with patch("stock_analyzer.data.yf_gateway.yf.Ticker", return_value=MagicMock()) as ctor:
        yf_gateway.get_ticker("NVDA")
        yf_gateway.get_ticker("NVDA")
    assert ctor.call_count == 2


def test_history_returns_none_for_empty_frame():
    import pandas as pd

    fake = MagicMock()
    fake.history.return_value = pd.DataFrame()
    with patch("stock_analyzer.data.yf_gateway.yf.Ticker", return_value=fake):
        assert yf_gateway.history("NVDA", period="2y") is None


def test_concurrency_never_exceeds_the_cap(monkeypatch: pytest.MonkeyPatch):
    """The point of the module: no matter how wide the fan-out, only
    MAX_CONCURRENCY requests are ever in flight."""
    monkeypatch.setattr(yf_gateway, "RATE_LIMIT_PER_MIN", 100_000)
    yf_gateway._pacer.reset()

    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def work(symbol: str) -> str:
        def _hit() -> str:
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.01)
            with lock:
                state["now"] -= 1
            return symbol

        return yf_gateway.call(_hit, symbol=symbol, what="test")

    out = dict(yf_gateway.map_symbols(work, [f"T{i}" for i in range(24)], workers=12))
    assert len(out) == 24
    assert state["peak"] <= yf_gateway.MAX_CONCURRENCY


def test_map_symbols_pairs_results_with_their_symbol():
    out = dict(yf_gateway.map_symbols(lambda s: s.lower(), ["AAPL", "MSFT"]))
    assert out == {"AAPL": "aapl", "MSFT": "msft"}


def test_map_symbols_on_empty_input_is_a_no_op():
    assert list(yf_gateway.map_symbols(lambda s: s, [])) == []


def test_pacer_spaces_calls_at_the_configured_rate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yf_gateway, "RATE_LIMIT_PER_MIN", 600)  # 0.1s apart
    yf_gateway._pacer.reset()
    start = time.monotonic()
    for _ in range(4):
        yf_gateway.call(lambda: None, what="test")
    assert time.monotonic() - start >= 0.25


def test_map_symbols_survives_one_bad_symbol():
    """A single unexpected failure must not take the other 499 names down."""

    def flaky(symbol: str) -> str:
        if symbol == "BAD":
            raise ValueError("unparseable frame")
        return symbol.lower()

    out = dict(yf_gateway.map_symbols(flaky, ["AAPL", "BAD", "MSFT"]))
    assert out == {"AAPL": "aapl", "BAD": None, "MSFT": "msft"}


def test_concurrent_rate_limits_count_as_one_penalty(monkeypatch: pytest.MonkeyPatch):
    """Four workers hitting the wall at once is one event: the rate must
    halve once, not drop to the floor."""
    monkeypatch.setattr(yf_gateway, "BASE_COOLDOWN_SECONDS", 30)
    yf_gateway._pacer.reset()
    start = yf_gateway._pacer.rate_per_min

    for _ in range(4):
        yf_gateway._pacer.penalize()

    assert yf_gateway._pacer.rate_per_min == pytest.approx(start / 2)


# --- dropped connections ----------------------------------------------------

# Verbatim from a daily-email run (2026-09-18): AMD and ARM each lost their
# earnings dates to one of these, with no second attempt.
_SSL_DROP = (
    "Failed to perform, curl: (35) BoringSSL SSL_connect: Connection closed "
    "abruptly (SSL_ERROR_SYSCALL; error queue empty) in connection to "
    "guce.yahoo.com:443. See https://curl.se/libcurl/c/libcurl-errors.html "
    "first for more details."
)


@pytest.fixture(autouse=True)
def _no_transient_backoff(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yf_gateway, "_TRANSIENT_BACKOFF_SECONDS", 0)


@pytest.mark.parametrize(
    "message",
    [
        _SSL_DROP,
        "Connection reset by peer",
        "HTTPSConnectionPool(host='query2.finance.yahoo.com'): Read timed out.",
        "502 Bad Gateway",
    ],
)
def test_dropped_connection_is_retried_then_succeeds(message):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(message)
        return "earnings"

    assert yf_gateway.call(flaky, symbol="AMD", what="earnings_dates") == "earnings"
    assert attempts["n"] == 2
    assert yf_gateway.stats()["transient"] == 1


def test_dropped_connection_gives_up_after_max_attempts():
    attempts = {"n": 0}

    def always_dropped():
        attempts["n"] += 1
        raise RuntimeError(_SSL_DROP)

    assert yf_gateway.call(always_dropped, symbol="ARM", what="earnings_dates", attempts=3) is None
    assert attempts["n"] == 3
    assert yf_gateway.stats()["failed"] == 1


def test_a_dropped_connection_does_not_slow_the_whole_process():
    """Unlike a 429, one dead socket says nothing about our request rate."""
    before = yf_gateway._pacer.rate_per_min
    yf_gateway.call(lambda: (_ for _ in ()).throw(RuntimeError(_SSL_DROP)), attempts=2)
    assert yf_gateway._pacer.rate_per_min == before


def test_missing_symbol_is_still_not_retried():
    attempts = {"n": 0}

    def delisted():
        attempts["n"] += 1
        raise RuntimeError("No price data found, symbol may be delisted")

    assert yf_gateway.call(delisted, symbol="DEAD", what="history") is None
    assert attempts["n"] == 1


def _bars_frame(days: int):
    import pandas as pd

    idx = pd.date_range(
        end=pd.Timestamp.today().normalize(), periods=days, freq="D", tz="America/New_York"
    )
    return pd.DataFrame({"Close": range(days)}, index=idx)


def test_daily_bars_serves_every_window_from_one_download():
    from datetime import date, timedelta

    fake = MagicMock()
    fake.history.return_value = _bars_frame(800)
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        year = yf_gateway.daily_bars("NVDA", start=date.today() - timedelta(days=365))
        month = yf_gateway.daily_bars("nvda", start=date.today() - timedelta(days=30))
        week_ago = date.today() - timedelta(days=7)
        upto = yf_gateway.daily_bars("NVDA", start=week_ago - timedelta(days=3), end=week_ago)

    assert fake.history.call_count == 1
    kwargs = fake.history.call_args.kwargs
    assert kwargs["auto_adjust"] is True
    assert date.fromisoformat(kwargs["start"]) <= date.today() - timedelta(days=730)
    assert len(year) == 366 and len(month) == 31
    assert len(upto) == 4 and upto.index[-1].date() == week_ago  # end is inclusive


def test_daily_bars_refetches_for_a_longer_window():
    from datetime import date, timedelta

    fake = MagicMock()
    fake.history.return_value = _bars_frame(1200)
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        yf_gateway.daily_bars("NVDA", start=date.today() - timedelta(days=30))
        far = date.today() - timedelta(days=1000)
        yf_gateway.daily_bars("NVDA", start=far)
        yf_gateway.daily_bars("NVDA", start=far + timedelta(days=10))

    assert fake.history.call_count == 2
    assert fake.history.call_args.kwargs["start"] == far.isoformat()


def test_daily_bars_is_none_when_yahoo_has_nothing():
    from datetime import date

    import pandas as pd

    fake = MagicMock()
    fake.history.return_value = pd.DataFrame()
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        assert yf_gateway.daily_bars("NVDA", start=date.today()) is None
