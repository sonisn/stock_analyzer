"""Daily bars kept on disk between runs.

Adjusted prices are rewritten backwards by every dividend and split, so
the store may only append when nothing was rewritten — these tests pin
the cases where it must download the whole history instead.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stock_analyzer.data import bar_store, yf_gateway

TZ = "America/New_York"


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("YF_BARS_DIR", str(tmp_path / "bars"))
    yf_gateway.reset()
    yield
    yf_gateway.reset()


def _bars(first: date, last: date, *, close_from: float = 100.0, events: dict | None = None):
    idx = pd.date_range(first, last, freq="D", tz=TZ, name="Date")
    frame = pd.DataFrame(
        {
            "Close": [close_from + i for i in range(len(idx))],
            "Volume": 1000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )
    for day, (col, value) in (events or {}).items():
        frame.loc[pd.Timestamp(day, tz=TZ), col] = value
    return frame


# Synced four days ago: always before the last final close, whatever the
# day. A one-day age counted as current from Saturday afternoon to Monday
# afternoon, so these tests failed every weekend — and blocked Monday's
# 08:30 update, which runs the suite first.
_STALE_S = 4 * 86400


def _seed(symbol: str, frame: pd.DataFrame, requested_from: date, *, age_s: float = _STALE_S):
    bar_store.save(symbol, frame, requested_from, checked_at=time.time() - age_s)


TODAY = date.today()
FROM = TODAY - timedelta(days=800)


def _fake_history(frames: list[pd.DataFrame]):
    fake = MagicMock()
    fake.history.side_effect = frames
    return fake


def test_first_read_downloads_and_stores():
    fake = _fake_history([_bars(FROM, TODAY)])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        out = yf_gateway.daily_bars("NVDA", start=TODAY - timedelta(days=30))
    assert len(out) == 31
    stored = bar_store.load("NVDA")
    assert stored is not None and stored.requested_from <= TODAY - timedelta(days=730)
    assert len(stored.frame) == len(_bars(FROM, TODAY))


def test_a_later_run_downloads_only_the_new_days():
    _seed("NVDA", _bars(FROM, TODAY - timedelta(days=3)), FROM)
    since = TODAY - timedelta(days=3 + bar_store.OVERLAP_DAYS)
    delta = _bars(since, TODAY, close_from=100.0 + (since - FROM).days)
    fake = _fake_history([delta])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        out = yf_gateway.daily_bars("NVDA", start=FROM)

    assert fake.history.call_count == 1
    assert fake.history.call_args.kwargs["start"] == since.isoformat()
    expected = _bars(FROM, TODAY)
    assert list(out["Close"]) == list(expected["Close"])
    assert yf_gateway.stats()["bars_extended"] == 1


def test_a_recent_sync_is_served_without_asking_yahoo():
    _seed("NVDA", _bars(FROM, TODAY), FROM, age_s=10)
    fake = _fake_history([])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        out = yf_gateway.daily_bars("NVDA", start=TODAY - timedelta(days=10))
    assert fake.history.call_count == 0
    assert len(out) == 11


@pytest.mark.parametrize("event", ["Dividends", "Stock Splits"])
def test_a_new_dividend_or_split_downloads_the_whole_history(event):
    _seed("KO", _bars(FROM, TODAY - timedelta(days=3)), FROM)
    since = TODAY - timedelta(days=3 + bar_store.OVERLAP_DAYS)
    delta = _bars(
        since,
        TODAY,
        close_from=100.0 + (since - FROM).days,
        events={TODAY - timedelta(days=1): (event, 0.5)},
    )
    full = _bars(FROM, TODAY, close_from=90.0)  # everything before rescaled
    fake = _fake_history([delta, full])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        out = yf_gateway.daily_bars("KO", start=FROM)

    assert fake.history.call_count == 2
    assert fake.history.call_args.kwargs["start"] == FROM.isoformat()
    assert out["Close"].iloc[0] == 90.0
    assert bar_store.load("KO").frame["Close"].iloc[0] == 90.0


def test_an_event_already_stored_does_not_refetch_every_day():
    """A dividend inside the overlap was absorbed by the last full download."""
    paid = TODAY - timedelta(days=5)
    _seed("KO", _bars(FROM, TODAY - timedelta(days=3), events={paid: ("Dividends", 0.5)}), FROM)
    since = TODAY - timedelta(days=3 + bar_store.OVERLAP_DAYS)
    delta = _bars(
        since,
        TODAY,
        close_from=100.0 + (since - FROM).days,
        events={paid: ("Dividends", 0.5)},
    )
    fake = _fake_history([delta])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        yf_gateway.daily_bars("KO", start=FROM)
    assert fake.history.call_count == 1


def test_restated_closes_download_the_whole_history():
    _seed("AAPL", _bars(FROM, TODAY - timedelta(days=3)), FROM)
    since = TODAY - timedelta(days=3 + bar_store.OVERLAP_DAYS)
    restated = _bars(since, TODAY, close_from=99.0 + (since - FROM).days)
    fake = _fake_history([restated, _bars(FROM, TODAY)])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        yf_gateway.daily_bars("AAPL", start=FROM)
    assert fake.history.call_count == 2


def test_the_last_stored_bar_may_change():
    """It may have been written mid-session; a new close is not a restatement."""
    stored = _bars(FROM, TODAY - timedelta(days=3))
    _seed("AAPL", stored, FROM)
    since = TODAY - timedelta(days=3 + bar_store.OVERLAP_DAYS)
    delta = _bars(since, TODAY, close_from=100.0 + (since - FROM).days)
    delta.loc[stored.index[-1], "Close"] += 2.5
    fake = _fake_history([delta])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        out = yf_gateway.daily_bars("AAPL", start=FROM)
    assert fake.history.call_count == 1
    assert out.loc[stored.index[-1], "Close"] == delta.loc[stored.index[-1], "Close"]


def test_yahoo_down_serves_the_stored_bars():
    _seed("NVDA", _bars(FROM, TODAY - timedelta(days=3)), FROM)
    fake = _fake_history([pd.DataFrame()])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        out = yf_gateway.daily_bars("NVDA", start=TODAY - timedelta(days=30))
    assert out is not None and out.index[-1].date() == TODAY - timedelta(days=3)


def test_a_longer_window_than_stored_downloads_it():
    _seed("NVDA", _bars(FROM, TODAY), FROM)
    far = TODAY - timedelta(days=2000)
    fake = _fake_history([_bars(far, TODAY)])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        yf_gateway.daily_bars("NVDA", start=far)
    assert fake.history.call_args.kwargs["start"] == far.isoformat()
    assert bar_store.load("NVDA").requested_from == far


def test_store_off_means_no_files(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("YF_BARS_DIR", "off")
    fake = _fake_history([_bars(FROM, TODAY)])
    with patch.object(yf_gateway.yf, "Ticker", return_value=fake):
        yf_gateway.daily_bars("NVDA", start=FROM)
    assert bar_store.load("NVDA") is None
    assert not (tmp_path / "bars").exists()


def test_an_unreadable_file_is_a_cache_miss():
    path = bar_store.store_dir() / "NVDA.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not parquet")
    assert bar_store.load("NVDA") is None


# --- the batched path (the model's panel) ------------------------------------


def _download_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, axis=1)


def test_batch_extends_stored_symbols_in_one_request_and_downloads_new_ones():
    start = TODAY - timedelta(days=400)
    _seed("AAA", _bars(start, TODAY - timedelta(days=3)), start)
    since = TODAY - timedelta(days=3 + bar_store.OVERLAP_DAYS)
    delta = _bars(since, TODAY, close_from=100.0 + (since - start).days)
    calls: list[tuple[list[str], str]] = []

    def fake_download(symbols, **kwargs):
        calls.append((list(symbols), kwargs["start"]))
        if symbols == ["AAA"]:
            return _download_frame({"AAA": delta})
        return _download_frame({"BBB": _bars(start, TODAY)})

    with patch.object(yf_gateway.yf, "download", side_effect=fake_download):
        out = yf_gateway.daily_bars_many(["aaa", "BBB"], start=start)

    assert calls == [(["AAA"], since.isoformat()), (["BBB"], start.isoformat())]
    assert list(out["AAA"]["Close"]) == list(_bars(start, TODAY)["Close"])
    assert len(out["BBB"]) == len(_bars(start, TODAY))
    assert bar_store.load("BBB") is not None


def test_batch_refetches_a_symbol_with_a_new_split():
    start = TODAY - timedelta(days=400)
    _seed("AAA", _bars(start, TODAY - timedelta(days=3)), start)
    since = TODAY - timedelta(days=3 + bar_store.OVERLAP_DAYS)
    split = _bars(
        since,
        TODAY,
        close_from=100.0 + (since - start).days,
        events={TODAY: ("Stock Splits", 4.0)},
    )
    full = _bars(start, TODAY, close_from=25.0)
    responses = iter([_download_frame({"AAA": split}), _download_frame({"AAA": full})])
    with patch.object(yf_gateway.yf, "download", side_effect=lambda *a, **k: next(responses)):
        out = yf_gateway.daily_bars_many(["AAA"], start=start)
    assert out["AAA"]["Close"].iloc[0] == 25.0


# --- when stored bars are as new as a download ------------------------------

NY = bar_store._EXCHANGE_TZ


def _at(y, m, d, hh, mm):
    from datetime import datetime

    return datetime(y, m, d, hh, mm, tzinfo=NY).timestamp()


def _synced(ts: float) -> bar_store.StoredBars:
    return bar_store.StoredBars(frame=pd.DataFrame(), requested_from=FROM, checked_at=ts)


def test_synced_after_the_close_is_current_all_evening_and_weekend():
    friday_evening = _synced(_at(2026, 9, 25, 17, 0))
    assert bar_store.is_current(friday_evening, 1800, now=_at(2026, 9, 25, 23, 0))
    assert bar_store.is_current(friday_evening, 1800, now=_at(2026, 9, 27, 12, 0))  # Sunday
    assert bar_store.is_current(friday_evening, 1800, now=_at(2026, 9, 28, 8, 0))  # Mon pre-open
    assert not bar_store.is_current(friday_evening, 1800, now=_at(2026, 9, 28, 10, 0))


def test_synced_before_the_close_is_stale_after_it():
    afternoon = _synced(_at(2026, 9, 25, 16, 15))
    assert not bar_store.is_current(afternoon, 1800, now=_at(2026, 9, 25, 20, 0))


def test_during_the_session_only_the_ttl_counts():
    morning = _synced(_at(2026, 9, 25, 9, 40))
    assert bar_store.is_current(morning, 1800, now=_at(2026, 9, 25, 10, 0))
    assert not bar_store.is_current(morning, 1800, now=_at(2026, 9, 25, 10, 30))
