"""Per-ticker Yahoo answers reused across runs, and the nightly warm-up."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import polars as pl

from stock_analyzer.cli import earnings_watch
from stock_analyzer.config import Settings
from stock_analyzer.data import bar_store, eps_revisions, fetch_cache, fundamentals


def _counting(answers):
    asked: list[str] = []

    def fetch(todo):
        for t in todo:
            asked.append(t)
            yield t, answers.get(t)

    return fetch, asked


def test_a_second_run_reads_the_cache_and_misses_are_asked_again(tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    fetch, asked = _counting({"AAA": {"x": 1}, "BBB": {"x": 2}})  # CCC: Yahoo had nothing
    assert fetch_cache.fetch_many("k", ["AAA", "BBB", "CCC"], fetch) == {
        "AAA": {"x": 1},
        "BBB": {"x": 2},
    }
    assert fetch_cache.fetch_many("k", ["BBB", "CCC", "AAA"], fetch) == {
        "AAA": {"x": 1},
        "BBB": {"x": 2},
    }
    assert asked == ["AAA", "BBB", "CCC", "CCC"]


def test_answers_expire_and_are_dropped_from_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    path = tmp_path / "k.json"
    path.write_text(json.dumps({"OLD": {"at": 0, "value": {"x": 0}}}))
    fetch, asked = _counting({"OLD": {"x": 9}, "NEW": {"x": 1}})
    assert fetch_cache.fetch_many("k", ["OLD"], fetch) == {"OLD": {"x": 9}}
    fetch_cache.fetch_many("k", ["NEW"], fetch)
    assert asked == ["OLD", "NEW"]
    assert set(json.loads(path.read_text())) == {"OLD", "NEW"}
    assert json.loads(path.read_text())["OLD"]["value"] == {"x": 9}


def test_zero_hours_or_off_bypasses_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FETCH_CACHE_DAYS", "0")
    fetch, asked = _counting({"AAA": {"x": 1}})
    fetch_cache.fetch_many("k", ["AAA"], fetch)
    fetch_cache.fetch_many("k", ["AAA"], fetch)
    assert asked == ["AAA", "AAA"] and not list(tmp_path.iterdir())


def test_a_corrupt_file_is_a_cold_start(tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    (tmp_path / "k.json").write_text("{not json")
    fetch, _ = _counting({"AAA": {"x": 1}})
    assert fetch_cache.fetch_many("k", ["AAA"], fetch) == {"AAA": {"x": 1}}


def test_batch_fundamentals_caches_yahoo_not_the_filed_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    calls: list[str] = []

    def fake_fetch(t):
        calls.append(t)
        return {"ticker": t, "free_cash_flow": 1.0}

    def overlay(results, *, as_of=None):
        for row in results.values():
            row["free_cash_flow"] = 2.0

    monkeypatch.setattr(fundamentals, "fetch_fundamentals", fake_fetch)
    monkeypatch.setattr(fundamentals, "_overlay_filed_values", overlay)
    assert fundamentals.batch_fundamentals(["AAA"])["AAA"]["free_cash_flow"] == 2.0
    assert fundamentals.batch_fundamentals(["AAA"])["AAA"]["free_cash_flow"] == 2.0
    assert calls == ["AAA"]
    stored = json.loads((tmp_path / "fundamentals.json").read_text())
    assert stored["AAA"]["value"]["free_cash_flow"] == 1.0  # Yahoo's answer, as fetched


def test_nightly_warm_up_fetches_what_the_prescreen_would_pick(monkeypatch):
    techs = {
        "UP": {"dist_from_52w_high": -0.10},
        "NEAR": {"dist_from_52w_high": -0.20},
        "KNIFE": {"dist_from_52w_high": -0.60},  # fails the soft gate
        "MINE": {"dist_from_52w_high": -0.70},  # tracked: passes anyway
    }
    fetched: dict[str, list[str]] = {}
    monkeypatch.setattr(earnings_watch, "tracked_tickers", lambda db, today: ["MINE"])
    monkeypatch.setattr(earnings_watch, "load_base_universe", lambda: ("UP", "NEAR", "KNIFE"))
    monkeypatch.setattr(earnings_watch, "batch_technicals", lambda ts: techs)
    monkeypatch.setattr(
        earnings_watch, "batch_fundamentals", lambda ts, refresh: fetched.setdefault("f", list(ts))
    )
    monkeypatch.setattr(
        earnings_watch,
        "batch_eps_revisions",
        lambda ts, refresh: fetched.setdefault("e", list(ts)),
    )
    settings = Settings(discover_trend_gate="soft", discover_max_screen_candidates=2)
    assert earnings_watch.warm_screen_cache("db", settings) == 2
    assert fetched == {"f": ["UP", "MINE"], "e": ["UP", "MINE"]}
    assert eps_revisions.batch_eps_revisions([]) == {}


def _entry(days_ago: float, value: dict) -> dict:
    return {"at": time.time() - days_ago * 86400, "value": value}


def test_an_answer_expires_after_the_report_it_was_fetched_ahead_of(tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    today = date.today()
    (tmp_path / "k.json").write_text(
        json.dumps(
            {
                "REPORTED": _entry(3, {"next_earnings": (today - timedelta(days=1)).isoformat()}),
                "LATER": _entry(3, {"next_earnings": (today + timedelta(days=9)).isoformat()}),
                "TODAY": _entry(3, {"next_earnings": today.isoformat()}),  # numbers not out yet
                "WEEKOLD": _entry(8.5, {}),
                "NODATE": _entry(6, {}),
            }
        )
    )
    assert set(fetch_cache.entries("k")) == {"LATER", "TODAY", "NODATE"}


def test_the_night_renews_the_oldest_fifth_and_a_failed_renewal_keeps_the_old(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    names = [f"T{i}" for i in range(10)]
    (tmp_path / "k.json").write_text(
        json.dumps({t: _entry(i / 2, {"v": t}) for i, t in enumerate(names)})
    )
    assert fetch_cache.oldest("k", [*names, "NEW"]) == ["T9", "T8", "T7"]  # ceil(11 / 5)
    fetch, asked = _counting({"T9": {"v": "renewed"}})  # T8's renewal fails
    got = fetch_cache.fetch_many("k", names, fetch, refresh=["T9", "T8"])
    assert sorted(asked) == ["T8", "T9"]
    assert got["T9"] == {"v": "renewed"} and got["T8"] == {"v": "T8"} and len(got) == 10


def test_a_cached_answer_is_repriced_to_a_later_close():
    row = {
        "quote_price": 100.0,
        "quote_date": "2026-09-20",
        "market_cap": 10e9,
        "forward_pe": 20.0,
        "trailing_pe": 25.0,
        "peg_ratio": None,
        "fcf_yield": 0.05,
        "analyst_target_mean": 150.0,
        "analyst_target_upside_pct": 0.5,
        "revenue_growth_yoy": 0.3,
    }
    fundamentals.reprice(row, (date(2026, 9, 20), 90.0))  # same day: Yahoo's live price stands
    assert row["market_cap"] == 10e9
    fundamentals.reprice(row, (date(2026, 9, 25), 120.0))
    assert row["market_cap"] == 12e9 and row["forward_pe"] == 24.0 and row["trailing_pe"] == 30.0
    assert abs(row["fcf_yield"] - 0.05 / 1.2) < 1e-12 and row["peg_ratio"] is None
    assert row["analyst_target_upside_pct"] == 0.25 and row["revenue_growth_yoy"] == 0.3
    assert (row["quote_price"], row["quote_date"]) == (120.0, "2026-09-25")


def test_fundamentals_are_one_request_and_carry_the_next_report(monkeypatch):
    info = {
        "marketCap": 5e9,
        "currentPrice": 50.0,
        "operatingCashflow": 4e8,
        "freeCashflow": 2e8,
        "earningsTimestamp": 1793190600,
    }
    asked: list[str] = []

    def call(ticker, what, fn, **kw):
        asked.append(what)
        return info

    monkeypatch.setattr(fundamentals.yf_gateway, "ticker_call", call)
    row = fundamentals.fetch_fundamentals("X")
    assert asked == ["fundamentals.info"]
    assert row["operating_cash_flow"] == 4e8 and row["quote_price"] == 50.0
    assert row["next_earnings"] and row["next_earnings"].startswith("2026-10-2")


def test_revisions_are_stamped_with_the_next_report_from_fundamentals(tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    (tmp_path / "fundamentals.json").write_text(
        json.dumps({"AAA": _entry(1, {"next_earnings": "2099-01-30"})})
    )
    monkeypatch.setattr(eps_revisions, "fetch_eps_revisions", lambda t: {"ticker": t})
    assert eps_revisions.batch_eps_revisions(["AAA", "BBB"]) == {
        "AAA": {"ticker": "AAA", "next_earnings": "2099-01-30"},
        "BBB": {"ticker": "BBB", "next_earnings": None},
    }


def test_the_stored_close_comes_from_the_bar_store(monkeypatch):
    frame = pl.DataFrame({"date": [date(2026, 9, 24), date(2026, 9, 25)], "Close": [10.0, 11.0]})
    monkeypatch.setattr(
        bar_store, "load", lambda t: bar_store.StoredBars(frame, date(2026, 1, 1), 0.0)
    )
    assert fundamentals._stored_close("X") == (date(2026, 9, 25), 11.0)


def test_a_past_report_date_is_rolled_forward_to_the_next_quarter():
    from datetime import datetime

    today = date(2026, 9, 27)
    past = datetime(2026, 7, 29, 16, 0).timestamp()
    assert fundamentals._next_report(past, today) == "2026-10-28"
    future = datetime(2026, 11, 17, 16, 0).timestamp()
    assert fundamentals._next_report(future, today) == "2026-11-17"
    assert fundamentals._next_report(None, today) is None
