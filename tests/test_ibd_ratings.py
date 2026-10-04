"""IBD-style ratings, bases and market direction on synthetic bars."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stock_analyzer.data import industry_groups
from stock_analyzer.data.quarterly_eps import eps_growth, periods
from stock_analyzer.discover import ibd_ratings as ibd


def _bars(close, *, volume=None, spread=0.005, start=date(2024, 1, 2)):
    close = np.asarray(close, dtype=float)
    volume = np.full(len(close), 1e6) if volume is None else np.asarray(volume, dtype=float)
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(len(close))],
            "Open": close,
            "High": close * (1 + spread),
            "Low": close * (1 - spread),
            "Close": close,
            "Volume": volume,
        }
    )


def _trend(n, start, end):
    return np.geomspace(start, end, n)


# --- percentiles and the per-stock measures ----------------------------------


def test_percentile_ranks_run_from_1_to_99_and_skip_blanks():
    ranks = ibd.percentile_ranks({"a": 1.0, "b": 2.0, "c": 3.0, "d": None})
    assert ranks == {"a": 1, "b": 50, "c": 99}
    assert ibd.percentile_ranks({"x": 5.0, "y": 5.0}) == {"x": 50, "y": 50}


def test_the_latest_quarter_counts_twice_in_rs():
    """Same 12-month gain; the stock that made it lately is stronger."""
    early = np.concatenate([_trend(190, 100, 150), np.full(64, 150.0)])
    late = np.concatenate([np.full(190, 100.0), _trend(64, 100, 150)])
    assert ibd.rs_strength(late) > ibd.rs_strength(early)
    assert ibd.rs_strength(np.ones(100)) is None  # under a year


def test_buying_near_the_highs_grades_above_selling_near_the_lows():
    n = 70
    close = np.full(n, 100.0)
    at_high = ibd.accumulation(close * 1.01, close * 0.99, close * 1.009, np.ones(n))
    at_low = ibd.accumulation(close * 1.01, close * 0.99, close * 0.991, np.ones(n))
    assert at_high > 0.8 and at_low < -0.8
    assert ibd.ad_grade(99) == "A" and ibd.ad_grade(1) == "E" and ibd.ad_grade(None) is None


def test_eps_strength_needs_the_latest_quarter_and_clips_turnarounds():
    assert ibd.eps_strength({"q1_growth": None, "cagr_3y": 0.2}) is None
    assert ibd.eps_strength({"q1_growth": 50.0}) == pytest.approx(3.0)  # +5,000% counts as +300%
    assert ibd.eps_strength(
        {"q1_growth": 0.5, "q2_growth": None, "cagr_3y": 0.25}
    ) == pytest.approx((0.4 * 0.5 + 0.4 * 0.25) / 0.8)


# --- bases ---------------------------------------------------------------------


def _flat_base(weeks=8, depth=0.10, extra=None):
    run_up = _trend(200, 50, 100)  # +100% into the base
    days = weeks * 5
    body = 100 * (1 - depth / 2 + depth / 2 * np.cos(np.linspace(0, 2 * np.pi, days)))
    tail = [] if extra is None else extra
    return _bars(np.concatenate([run_up, body, tail]), spread=0.0)


def test_a_shallow_consolidation_after_a_run_up_is_a_flat_base():
    base = ibd.find_base(_flat_base(weeks=8, depth=0.10))
    assert base is not None and base.kind == "flat base"
    assert base.weeks == 8
    assert base.depth == pytest.approx(0.10, abs=0.01)
    assert base.pivot == pytest.approx(100.10, abs=0.02)


def test_no_run_up_means_no_base():
    flat = _bars(
        np.concatenate(
            [np.full(200, 100.0), 100 * (0.95 + 0.05 * np.cos(np.linspace(0, 6.28, 40)))]
        )
    )
    assert ibd.find_base(flat) is None


def test_price_against_the_pivot_sets_the_status():
    below = ibd.find_base(_flat_base(extra=[97.0]))
    assert below is not None and below.status == "below pivot"
    zone = ibd.find_base(_flat_base(extra=[101.0] * 3))
    assert zone is not None and zone.status in {"buy zone", "breakout"}
    far = ibd.find_base(_flat_base(extra=[112.0] * 3))
    assert far is not None and far.status == "extended"


def test_a_deeper_base_with_a_shallow_upper_pullback_is_a_cup_with_handle():
    run_up = _trend(200, 50, 100)
    left = np.linspace(100, 75, 25)
    right = np.linspace(75, 98, 25)
    handle = np.linspace(98, 93, 8)
    base = ibd.find_base(_bars(np.concatenate([run_up, left, right, handle]), spread=0.0))
    assert base is not None and base.kind == "cup with handle"
    assert base.pivot == pytest.approx(98.10, abs=0.05)


# --- the market ------------------------------------------------------------------


def test_a_down_day_on_rising_volume_is_distribution():
    close = np.full(40, 100.0)
    vol = np.full(40, 1e6)
    close[-3:] = [99.5, 99.5, 99.5]  # -0.5% on the first of them
    vol[-3] = 2e6
    days = ibd.distribution_days(_bars(close, volume=vol))
    assert days == [37]


def test_a_five_percent_rally_retires_a_distribution_day():
    close = np.concatenate([np.full(30, 100.0), [99.0], np.linspace(99, 105, 5)])
    vol = np.full(len(close), 1e6)
    vol[30] = 2e6
    assert ibd.distribution_days(_bars(close, volume=vol)) == []


def test_follow_through_is_day_four_or_later_on_higher_volume():
    close = np.array([100, 90, 91, 91.5, 92, 94.0])
    vol = np.array([1, 1, 1, 1, 1, 2.0]) * 1e6
    bars = _bars(close, volume=vol, spread=0.0)
    assert ibd.follow_through_since_low(bars, 1) == 5
    undercut = _bars(np.array([100, 90, 91, 89, 92, 94.0]), volume=vol, spread=0.0)
    assert ibd.follow_through_since_low(undercut, 1) is None  # day 2 after the new low


def test_market_status_from_drawdown_and_distribution():
    rising = _bars(_trend(300, 100, 130))
    assert ibd.market_direction({"SPY": rising})["status"] == "Confirmed uptrend"
    falling = _bars(np.concatenate([_trend(250, 100, 130), _trend(50, 130, 110)]))
    assert ibd.market_direction({"SPY": falling})["status"] == "Market in correction"
    assert ibd.market_direction({})["status"] == "unknown"


# --- the universe ------------------------------------------------------------------


def test_rate_universe_ranks_leaders_first_and_ranks_groups():
    n = 300
    bars = {
        "LEAD": _bars(_trend(n, 50, 150)),
        "MID": _bars(_trend(n, 100, 120)),
        "LAG": _bars(_trend(n, 100, 80)),
        "SEMI": _bars(_trend(n, 60, 140)),
    }
    rows = ibd.rate_universe(
        bars,
        spy=_bars(_trend(n, 100, 110)),
        industries={"LEAD": "Chips", "SEMI": "Chips", "MID": "Banks", "LAG": "Banks"},
        eps={"LEAD": {"q1_growth": 0.8}, "LAG": {"q1_growth": -0.3}},
    )
    by = {r["ticker"]: r for r in rows}
    assert rows[0]["ticker"] == "LEAD"
    assert by["LEAD"]["rs_rating"] == 99 and by["LAG"]["rs_rating"] == 1
    assert by["LEAD"]["eps_rating"] == 99 and by["MID"]["eps_rating"] is None
    assert by["LEAD"]["rs_line_high"] is True
    # Two members per group: below MIN_GROUP_SIZE, so no group rank.
    assert by["LEAD"]["group_rank"] is None


# --- SEC EPS ------------------------------------------------------------------------


def _fact(start, end, val, filed="2026-08-01"):
    return {"start": start, "end": end, "val": val, "filed": filed}


def test_q4_comes_from_the_year_less_its_three_quarters():
    facts = [
        _fact("2025-01-01", "2025-03-31", 1.0),
        _fact("2025-04-01", "2025-06-30", 1.0),
        _fact("2025-07-01", "2025-09-30", 1.0),
        _fact("2025-01-01", "2025-12-31", 5.0),
    ]
    quarters, years = periods(facts)
    assert quarters[-1] == (date(2025, 12, 31), 2.0)
    assert years == [(date(2025, 12, 31), 5.0)]


def test_eps_growth_over_the_same_quarter_a_year_before():
    q = [
        (date(2025, 3, 31), 1.0),
        (date(2025, 6, 30), 1.0),
        (date(2026, 3, 31), 1.5),
        (date(2026, 6, 30), 2.0),
    ]
    years = [(date(2022, 12, 31), 2.0), (date(2025, 12, 31), 4.0)]
    g = eps_growth(q, years, today=date(2026, 8, 15))
    assert g["q1_growth"] == pytest.approx(1.0)
    assert g["q2_growth"] == pytest.approx(0.5)
    assert g["cagr_3y"] == pytest.approx(2 ** (1 / 3) - 1)
    assert eps_growth(q, years, today=date(2027, 6, 1)) is None  # stopped filing


def test_growth_off_a_near_zero_base_is_not_a_number():
    q = [(date(2025, 6, 30), 0.001), (date(2026, 6, 30), 1.0)]
    assert eps_growth(q, [], today=date(2026, 8, 1))["q1_growth"] is None


def test_an_empty_concept_falls_back_to_the_latest_tag_in_company_facts(monkeypatch):
    from stock_analyzer.data import quarterly_eps as q

    old = [_fact("2020-01-01", "2020-03-31", 0.5)]
    now = [_fact("2026-04-01", "2026-06-30", 1.2)]
    company = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {"units": {"USD/shares": old}},
                "IncomeLossFromContinuingOperationsPerDilutedShare": {"units": {"USD/shares": now}},
            }
        }
    }
    asked = []

    def get_json(url):
        asked.append(url.rsplit("/", 1)[-1])
        return {"units": {"USD/shares": []}} if "companyconcept" in url else company

    monkeypatch.setattr(q, "load_ticker_cik_map", lambda: {"IQV": 1478242})
    monkeypatch.setattr(q._HTTP, "get_json", get_json)
    assert q.fetch_facts("IQV") == now
    assert asked == ["EarningsPerShareDiluted.json", "CIK0001478242.json"]


def test_a_stale_concept_falls_back_too(monkeypatch):
    from stock_analyzer.data import quarterly_eps as q

    stale = [_fact("2011-04-01", "2011-06-30", 0.5)]
    now = [
        _fact(str(date.today() - timedelta(days=120)), str(date.today() - timedelta(days=30)), 1.2)
    ]
    company = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {"units": {"USD/shares": stale}},
                "IncomeLossFromContinuingOperationsPerDilutedShare": {"units": {"USD/shares": now}},
            }
        }
    }
    monkeypatch.setattr(q, "load_ticker_cik_map", lambda: {"MNST": 865752})
    monkeypatch.setattr(
        q._HTTP,
        "get_json",
        lambda url: {"units": {"USD/shares": stale}} if "companyconcept" in url else company,
    )
    assert q.fetch_facts("MNST") == now


def test_yahoo_reports_are_dated_to_the_quarter_before_and_summed_into_years():
    from stock_analyzer.data.quarterly_eps import eps_growth, yahoo_periods

    # Reported in the month after each quarter; the three 2026 reports doubled.
    days = [date(y, m, 25) for y in range(2022, 2027) for m in (1, 4, 7, 10)]
    reports = [(d, 2.0 if d.year == 2026 else 1.0) for d in days if d <= date(2026, 7, 31)]
    quarters, years = yahoo_periods(reports)
    assert quarters[0][0] == date(2021, 12, 31) and quarters[-1] == (date(2026, 6, 30), 2.0)
    assert years[-1] == (date(2026, 6, 30), 7.0)  # Q3 2025 at 1.0 + three at 2.0
    g = eps_growth(quarters, years, today=date(2026, 8, 15))
    assert g["q1_growth"] == pytest.approx(1.0) and g["cagr_3y"] is not None
    # Half-yearly reporters get quarters but no summed years.
    assert (
        yahoo_periods(
            [
                (date(2025, 8, 1), 1.0),
                (date(2026, 2, 1), 1.0),
                (date(2026, 8, 1), 1.0),
                (date(2027, 2, 1), 1.0),
            ]
        )[1]
        == []
    )


def test_batch_eps_falls_back_to_yahoo_and_retries_when_yahoo_is_down(monkeypatch, tmp_path):
    from stock_analyzer.data import quarterly_eps as q

    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        q.yf_gateway, "map_symbols", lambda fn, todo, workers: ((t, fn(t)) for t in todo)
    )
    monkeypatch.setattr(
        q,
        "fetch_eps",
        lambda t, strict: {"q1_growth": 0.1, "source": "sec"} if t == "SEC" else None,
    )
    yahoo = {"TSM": [(date(2025, 7, 17), 1.0), (date(2026, 7, 16), 1.5)], "DOWN": None, "NONE": []}
    monkeypatch.setattr(q, "yahoo_reports", lambda t: yahoo[t])
    today = date.today()
    yahoo["TSM"] = [(today - timedelta(days=380), 1.0), (today - timedelta(days=15), 1.5)]
    got = q.batch_eps(["SEC", "TSM", "DOWN", "NONE"])
    assert got["SEC"]["source"] == "sec" and got["TSM"]["source"] == "yahoo"
    assert "DOWN" not in got and "NONE" not in got
    cached = json.loads((tmp_path / "quarterly_eps.json").read_text())
    assert "DOWN" not in cached  # asked again next run
    assert cached["NONE"]["value"] == {"untagged": True, "checked": True}


def test_answers_cached_as_none_before_the_fallbacks_are_asked_again(monkeypatch, tmp_path):
    import time

    from stock_analyzer.data import quarterly_eps as q

    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    now = time.time()
    (tmp_path / "quarterly_eps.json").write_text(
        json.dumps(
            {
                "OLD": {"at": now, "value": {"untagged": True}},
                "NEW": {"at": now, "value": {"untagged": True, "checked": True}},
            }
        )
    )
    asked = []
    monkeypatch.setattr(
        q.yf_gateway, "map_symbols", lambda fn, todo, workers: asked.extend(todo) or []
    )
    q.batch_eps(["OLD", "NEW"])
    assert asked == ["OLD"]


# --- industry map --------------------------------------------------------------------


def test_the_industry_map_is_scanned_weekly_and_a_failed_scan_keeps_the_old(monkeypatch, tmp_path):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    scans = []

    def rescan():
        scans.append(1)
        return (
            {}
            if len(scans) > 1
            else {"NVDA": {"sector": "Technology", "industry": "Semiconductors"}}
        )

    assert industry_groups.industry_map(now=1_000_000, rescan=rescan) == {"NVDA": "Semiconductors"}
    assert industry_groups.industry_map(now=1_000_000 + 86400, rescan=rescan) == {
        "NVDA": "Semiconductors"
    }
    assert len(scans) == 1  # still fresh
    later = 1_000_000 + 8 * 86400
    assert industry_groups.industry_map(now=later, rescan=rescan) == {"NVDA": "Semiconductors"}
    assert len(scans) == 2  # rescanned, came back empty, kept last week's
    assert json.loads((tmp_path / "industry_groups.json").read_text())["at"] == 1_000_000


def test_an_industry_map_that_cannot_be_read_is_not_rescanned_over(monkeypatch, tmp_path):
    monkeypatch.setenv("FETCH_CACHE_DIR", str(tmp_path))
    good = {"at": 1_000_000, "map": {"NVDA": {"sector": "Technology", "industry": "Semis"}}}
    (tmp_path / "industry_groups.json").write_text(json.dumps(good))

    def no_handles(self, *a, **k):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(type(tmp_path), "read_text", no_handles)
    scans = []
    got = industry_groups.industry_map(now=1_000_000, rescan=lambda: scans.append(1) or {})
    monkeypatch.undo()
    assert got == {} and not scans
    assert json.loads((tmp_path / "industry_groups.json").read_text()) == good


def test_the_scan_pages_through_each_industry_in_our_spelling():
    pages = {
        ("Semis", 0): {"quotes": [{"symbol": "NVDA"}] * 1 + [{"symbol": "BRK.B"}], "total": 2},
    }
    found = industry_groups.scan(
        lambda ind, off: pages[(ind, off)], {"Technology": ["Semis"]}, pause=0
    )
    assert found["BRK-B"] == {"sector": "Technology", "industry": "Semis"}


# --- the dashboard tab ----------------------------------------------------------------


def _store(db, n_leaders=5):
    from stock_analyzer.db.session import get_session
    from stock_analyzer.db.tables import IbdMarket, IbdRating

    with get_session(db) as s:
        for i in range(n_leaders):
            s.add(
                IbdRating(
                    ticker=f"L{i}",
                    as_of="2026-09-25",
                    composite=99 - i,
                    industry="Chips",
                    group_rank=1,
                )
            )
        s.add(IbdRating(ticker="HELD", as_of="2026-09-25", composite=10))
        s.add(IbdRating(ticker="ZONE", as_of="2026-09-25", composite=75, base_status="buy zone"))
        s.add(
            IbdRating(
                ticker="FAR",
                as_of="2026-09-25",
                composite=5,
                base_status="below pivot",
                vs_pivot=-0.2,
            )
        )
        s.add(
            IbdMarket(
                day="2026-09-25", status="Confirmed uptrend", detail="2 days", indexes='{"SPY": {}}'
            )
        )
        s.commit()


def test_the_tab_carries_leaders_holdings_picks_and_buy_zones_only(monkeypatch, tmp_path):
    from stock_analyzer.reporting import leaders

    db = str(tmp_path / "d.db")
    _store(db)
    monkeypatch.setattr(leaders, "LEADERS", 2)
    monkeypatch.setattr(leaders, "charts", lambda tickers: ([], {}))
    monkeypatch.setattr(
        leaders,
        "signals",
        lambda db, tickers, today: {"FAR": {"pick": {"rank": 1, "d": "2026-09-20"}}},
    )
    data = leaders.collect(db, held={"HELD"}, today=date(2026, 9, 27))
    kept = {r["t"] for r in data["rows"]}
    assert kept == {"L0", "L1", "HELD", "ZONE", "FAR"}  # FAR only because we picked it
    assert data["market"]["status"] == "Confirmed uptrend"
    assert data["groups"][0]["leaders"] == ["L0", "L1", "L2", "L3", "L4"]
    far = next(r for r in data["rows"] if r["t"] == "FAR")
    assert far["ours"]["pick"]["rank"] == 1
    assert next(r for r in data["rows"] if r["t"] == "HELD")["held"] is True


def test_our_signals_join_the_ratings(monkeypatch, tmp_path):
    from stock_analyzer.reporting import leaders

    monkeypatch.setattr(leaders, "_picks", lambda db, today: {"NVDA": {"rank": 2}})
    monkeypatch.setattr(leaders, "_screen_scores", lambda db: {"NVDA": 71.5})
    monkeypatch.setattr(
        leaders,
        "_cached",
        lambda kind, field: {"NVDA": 68.0} if kind == "contracted_book" else {"NVDA": "raising"},
    )
    monkeypatch.setattr(
        "stock_analyzer.data.hedge_funds_13f.changes",
        lambda db, tickers: {
            "NVDA": [
                {
                    "action": "added",
                    "fund": "F",
                    "shares_change_pct": 20,
                    "weight_pct": 3.0,
                    "weight_before_pct": 2.5,
                }
            ]
        },
    )
    monkeypatch.setattr("stock_analyzer.data.insider_buying.clusters", lambda db, today: [])
    monkeypatch.setattr(
        "stock_analyzer.discover.earnings_standouts.recent_standouts",
        lambda db, days, today: (_ for _ in ()).throw(RuntimeError("db locked")),
    )
    got = leaders.signals("db", ["NVDA", "AAPL"], date(2026, 9, 27))
    assert set(got) == {"NVDA"}
    n = got["NVDA"]
    assert n["pick"] == {"rank": 2} and n["score"] == 71.5 and n["book"] == 68.0
    assert n["rev"] == "raising" and n["funds_net"] == 1 and "F +20%" in n["funds"]
    assert "standout" not in n  # a failing source costs its signal, not the tab


def test_the_chart_is_six_months_on_spys_calendar_with_an_rs_line(monkeypatch):
    from types import SimpleNamespace

    from stock_analyzer.reporting import leaders

    spy = _bars(np.full(300, 100.0))
    stock = _bars(np.linspace(50, 100, 300))
    monkeypatch.setattr(
        leaders.bar_store,
        "load",
        lambda t: (
            SimpleNamespace(frame={"SPY": spy, "NVDA": stock}[t]) if t in ("SPY", "NVDA") else None
        ),
    )
    calendar, series = leaders.charts(["NVDA", "MISSING"])
    assert len(calendar) == leaders.CHART_DAYS and set(series) == {"NVDA"}
    s = series["NVDA"]
    assert len(s["c"]) == leaders.CHART_DAYS
    assert s["rs"][0] == 100.0 and s["rs"][-1] > 100  # rising against a flat SPY
    assert s["m50"][-1] is not None and s["m200"][-1] is not None
    assert s["v"][-1] == 1000  # volume in thousands


def test_the_page_has_both_tabs_and_says_when_ratings_are_missing():
    from stock_analyzer.dashboard_page import render_page

    base = dict(
        generated="2026-09-27",
        latest_run=1,
        holdings=[],
        history={},
        views={},
        runs=[],
        record={"rows": 0, "tickers": 0, "first": None, "last": None},
        suggestions=[],
        holdings_ok=True,
        reasoning={},
        reviews={},
    )
    page = render_page({**base, "ibd": {}})
    assert 'data-tab="leaders"' in page and 'id="tab-leaders"' in page
    assert "No ratings yet" in page


# --- history and signals, for comparing progress ------------------------------------------


def _r(t, comp, status=None):
    return {"ticker": t, "composite": comp, "base_status": status}


def test_daily_history_keeps_the_names_that_matter_and_weekly_keeps_all():
    rows = [_r("TOP", 95), _r("ZONE", 60, "buy zone"), _r("MINE", 20), _r("REST", 50)]
    daily = {r["ticker"] for r in ibd.history_rows(rows, tracked={"MINE"}, full=False)}
    assert daily == {"TOP", "ZONE", "MINE"}
    weekly = ibd.history_rows(rows, tracked=set(), full=True)
    assert {r["ticker"] for r in weekly} == {"TOP", "ZONE", "MINE", "REST"}
    assert all(r["full"] for r in weekly)


def test_a_signal_needs_composite_80_and_is_not_repeated_within_30_days():
    rows = [
        _r("A", 85, "buy zone"),
        _r("B", 85, "breakout"),
        _r("C", 70, "buy zone"),
        _r("D", 90, "extended"),
        _r("E", 90, "buy zone"),
    ]
    assert [r["ticker"] for r in ibd.new_signals(rows, recent={"E"})] == ["A", "B"]


def test_signals_are_graded_against_spy_over_the_same_sessions(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from stock_analyzer.db.session import get_session
    from stock_analyzer.db.tables import IbdSignal
    from stock_analyzer.reporting import leaders

    db = str(tmp_path / "d.db")
    spy = _bars(np.linspace(100, 110, 200))  # +10% over 199 sessions
    stock = _bars(np.linspace(100, 150, 200))
    day = str(spy["date"][0])
    young = str(spy["date"][190])
    with get_session(db) as s:
        s.add(IbdSignal(ticker="WIN", day=day, status="buy zone", price=100.0))
        s.add(IbdSignal(ticker="NEW", day=young, status="breakout", price=148.0))
        s.commit()
    monkeypatch.setattr(
        leaders.bar_store, "load", lambda t: SimpleNamespace(frame={"SPY": spy}.get(t, stock))
    )
    card = leaders.signal_scorecard(db)
    win = next(x for x in card["signals"] if x["t"] == "WIN")
    assert win["h"]["21"]["done"] and win["h"]["21"]["ret"] > win["h"]["21"]["spy"] > 0
    new = next(x for x in card["signals"] if x["t"] == "NEW")
    assert not new["h"]["21"]["done"]  # 9 sessions old: shown "so far", not averaged
    from stock_analyzer.serialization import dumps_compact

    dumps_compact(card)  # the page embeds it: every key must be a string
    one_month = card["summary"][0]
    assert one_month["n"] == 1 and one_month["beat"] == 100 and one_month["avg_excess"] > 0


def test_a_missing_close_is_skipped_not_divided():
    from stock_analyzer.reporting import leaders

    frame = _bars(np.linspace(100, 110, 30)).with_columns(
        pl.when(pl.int_range(pl.len()) >= 25).then(None).otherwise(pl.col("Close")).alias("Close")
    )
    day = str(frame["date"][0])
    ret, n = leaders._forward(frame, day, 29)  # the last 5 closes are missing
    assert n == 24 and ret == pytest.approx(frame["Close"][24] / 100 - 1)
    nan = _bars([100.0, float("nan")])
    assert leaders._forward(nan, str(nan["date"][0]), 1) == (None, 0)


def test_rating_history_comes_back_per_stock_in_date_order(tmp_path):
    from stock_analyzer.db.session import get_session
    from stock_analyzer.db.tables import IbdHistory
    from stock_analyzer.reporting import leaders

    db = str(tmp_path / "d.db")
    with get_session(db) as s:
        for d, c in (("2026-09-24", 90), ("2026-09-25", 93), ("2026-09-23", 88)):
            s.add(IbdHistory(day=d, ticker="NVDA", composite=c, rs_rating=c - 5))
        s.add(IbdHistory(day="2026-09-25", ticker="ONCE", composite=50))
        s.commit()
    got = leaders.rating_history(db, ["NVDA", "ONCE"], date(2026, 9, 27))
    assert got["NVDA"]["d"] == ["2026-09-23", "2026-09-24", "2026-09-25"]
    assert got["NVDA"]["c"] == [88, 90, 93]
    assert "ONCE" not in got  # one point is not a line yet


def test_the_morning_job_stores_ratings_history_and_signals(monkeypatch, tmp_path):
    from sqlalchemy import select

    from stock_analyzer.cli import ibd as job
    from stock_analyzer.config import Settings
    from stock_analyzer.db.session import get_session
    from stock_analyzer.db.tables import IbdHistory, IbdMarket, IbdRating, IbdSignal

    n = 300
    bars = {f"S{i}": _bars(_trend(n, 50 + i, 150 - i)) for i in range(6)}
    bars["SPY"] = bars["QQQ"] = _bars(_trend(n, 100, 120))
    monkeypatch.setattr(job, "all_us_2b", lambda: tuple(f"S{i}" for i in range(6)))
    monkeypatch.setattr(job, "tracked_tickers", lambda db, today: ["S5"])
    monkeypatch.setattr(job.yf_gateway, "daily_bars_many", lambda names, start, what: dict(bars))
    monkeypatch.setattr(job, "industry_map", lambda: {})
    monkeypatch.setattr(job, "batch_eps", lambda names, refresh: {})
    monkeypatch.setattr(job.fetch_cache, "oldest", lambda kind, names: [])

    leader = []

    def zone(rows):  # make the top-rated stock a buy-zone signal
        rows[0]["base_status"], rows[0]["base"] = "buy zone", "flat base"
        leader[:] = [rows[0]["ticker"]]
        return rows

    real = job.rate_universe
    monkeypatch.setattr(job, "rate_universe", lambda *a, **k: zone(real(*a, **k)))
    db = str(tmp_path / "d.db")
    settings = Settings(discover_db_path=db)
    job.run(settings, today=date(2026, 9, 27))
    with get_session(db) as s:
        assert len(s.scalars(select(IbdRating)).all()) == 6
        first = s.scalars(select(IbdHistory)).all()
        assert len(first) == 6 and all(h.full for h in first)  # the first run is a full snapshot
        assert [x.ticker for x in s.scalars(select(IbdSignal)).all()] == leader
        assert len(s.scalars(select(IbdMarket)).all()) == 1
    # The next session: a daily (partial) snapshot, and no repeat signal.
    for k, b in bars.items():
        bars[k] = pl.concat([b, b.tail(1).with_columns(pl.col("date") + timedelta(days=1))])
    job.run(settings, today=date(2026, 9, 28))
    with get_session(db) as s:
        days = {h.day for h in s.scalars(select(IbdHistory)).all()}
        second = [h for h in s.scalars(select(IbdHistory)).all() if h.day == max(days)]
        assert len(days) == 2 and not any(h.full for h in second)
        assert {h.ticker for h in second} >= {leader[0], "S5"}  # the signal's and the tracked one
        assert len(s.scalars(select(IbdSignal)).all()) == 1


def test_backfill_rebuilds_past_days_without_touching_live_ones(monkeypatch, tmp_path):
    from sqlalchemy import select

    from stock_analyzer.cli import ibd as job
    from stock_analyzer.config import Settings
    from stock_analyzer.data import quarterly_eps
    from stock_analyzer.db.session import get_session
    from stock_analyzer.db.tables import IbdHistory, IbdMarket

    n = 320
    bars = {f"S{i}": _bars(_trend(n, 50 + i, 150 - i)) for i in range(4)}
    bars["SPY"] = bars["QQQ"] = _bars(_trend(n, 100, 120))
    monkeypatch.setattr(job, "all_us_2b", lambda: tuple(f"S{i}" for i in range(4)))
    monkeypatch.setattr(job, "tracked_tickers", lambda db, today: [])
    monkeypatch.setattr(job.yf_gateway, "daily_bars_many", lambda names, start, what: dict(bars))
    monkeypatch.setattr(job, "industry_map", lambda: {})
    monkeypatch.setattr(job, "batch_eps", lambda names, refresh: {})
    monkeypatch.setattr(job.fetch_cache, "oldest", lambda kind, names: [])
    monkeypatch.setattr(quarterly_eps, "fetch_facts", lambda t: None)
    db = str(tmp_path / "d.db")
    settings = Settings(discover_db_path=db)
    job.run(settings, today=date(2026, 9, 27))  # the live day: the last bar
    live_day = str(bars["SPY"]["date"][-1])
    job.backfill(settings, sessions=10, today=date(2026, 9, 27))
    with get_session(db) as s:
        hist = s.scalars(select(IbdHistory)).all()
        by_day = {}
        for h in hist:
            by_day.setdefault(h.day, []).append(h)
        assert len(by_day) == 11  # ten rebuilt sessions and the live one
        assert not any(h.backfilled for h in by_day[live_day])
        rebuilt = [d for d in by_day if d != live_day]
        assert all(h.backfilled for d in rebuilt for h in by_day[d])
        assert sum(1 for d in rebuilt if all(h.full for h in by_day[d])) == 2  # weekly snapshots
        assert len(s.scalars(select(IbdMarket)).all()) == 11


def test_eps_as_of_uses_only_what_was_filed_by_then():
    facts = [
        _fact("2025-01-01", "2025-03-31", 1.0, filed="2025-05-01"),
        _fact("2026-01-01", "2026-03-31", 2.0, filed="2026-05-01"),
    ]
    from stock_analyzer.data.quarterly_eps import growth_as_of

    assert (
        growth_as_of(facts, date(2026, 4, 15)) is None
    )  # the 2026 quarter was not out yet (and 2025's is stale)
    assert growth_as_of(facts, date(2026, 5, 2))["q1_growth"] == pytest.approx(1.0)


# --- leaders into discover, and the view map ------------------------------------------


def test_leaders_for_discover_are_the_top_n_plus_strong_buy_zones(tmp_path):
    from stock_analyzer.cli.ibd import top_leaders
    from stock_analyzer.db.session import get_session
    from stock_analyzer.db.tables import IbdRating

    db = str(tmp_path / "d.db")
    with get_session(db) as s:
        for t, c, st in (
            ("A", 99, None),
            ("B", 97, None),
            ("C", 92, "buy zone"),
            ("D", 85, "buy zone"),
            ("E", 50, None),
        ):
            s.add(IbdRating(ticker=t, as_of="2026-09-25", composite=c, base_status=st))
        s.commit()
    assert top_leaders(db, 2, today=date(2026, 9, 27)) == ("A", "B", "C")  # C: 90+ in a buy zone
    assert top_leaders(db, 0, today=date(2026, 9, 27)) == ()
    assert top_leaders(db, 2, today=date(2026, 10, 9)) == ()  # stale ratings feed nothing


def test_leaders_join_the_universe_without_a_score_bonus(monkeypatch):
    from stock_analyzer.discover import universe as uni
    from stock_analyzer.discover.screen import _score_conviction

    monkeypatch.setattr(uni, "fetch_insider_trades", lambda **k: [])
    monkeypatch.setattr(uni, "fetch_hedge_fund_trades", lambda **k: [])
    monkeypatch.setattr(uni, "_drop_unlisted", lambda u: u)
    got = uni.build_universe(base_universe=("AAPL",), ibd_leaders=("DELL", "AAPL"))
    assert "ibd_leader" in got["DELL"]["sources"] and "ibd_leader" in got["AAPL"]["sources"]
    assert _score_conviction(got["DELL"])[0] == 0.0


def test_the_view_map_ranks_our_view_against_the_composite(monkeypatch):
    from types import SimpleNamespace

    from stock_analyzer.reporting import leaders

    good = {
        "revenue_growth_yoy": 0.3,
        "fcf_yield": 0.06,
        "operating_margin": 0.3,
        "debt_to_equity": 0.2,
    }
    weak = {
        "revenue_growth_yoy": 0.0,
        "fcf_yield": 0.0,
        "operating_margin": 0.02,
        "debt_to_equity": 1.9,
    }
    caches = {
        "fundamentals": {"AVGO": {"value": good}, "TSLA": {"value": weak}, "DELL": {"value": weak}},
        "contracted_book": {"AVGO": {"value": {"yoy_pct": 552.0}}},
        "eps_revisions": {},
    }
    monkeypatch.setattr(leaders.fetch_cache, "entries", lambda kind: caches[kind])
    rows = [
        SimpleNamespace(ticker=t, composite=c, industry=None)
        for t, c in (("AVGO", 68), ("TSLA", 14), ("DELL", 99), ("NOFUND", 90))
    ]
    pts = {p["t"]: p for p in leaders.view_map(rows, {"AVGO", "TSLA"}, {})}
    assert set(pts) == {"AVGO", "TSLA", "DELL"}  # no fundamentals, no point
    assert pts["AVGO"]["x"] > pts["DELL"]["x"] and pts["DELL"]["y"] > pts["AVGO"]["y"]
    assert pts["AVGO"]["held"] and not pts["DELL"]["held"]


# --- re-check prompts in the daily email -------------------------------------------------


def test_a_heavy_volume_200_day_break_is_found_and_a_quiet_one_is_not():
    from stock_analyzer.reporting.leaders import _two_hundred_day_break

    close = np.concatenate([np.full(250, 100.0), [80.0, 79.0]])
    loud = np.concatenate([np.full(250, 1e6), [3e6, 1e6]])
    brk = _two_hundred_day_break(_bars(close, volume=loud))
    assert brk is not None and brk["volume_x"] == 3.0
    assert _two_hundred_day_break(_bars(close)) is None  # same drop on normal volume
    assert _two_hundred_day_break(_bars(np.full(252, 100.0))) is None


def test_holdings_get_a_recheck_when_the_market_turns_on_them(monkeypatch, tmp_path):
    from stock_analyzer.db.session import get_session
    from stock_analyzer.db.tables import IbdHistory, IbdRating
    from stock_analyzer.reporting import leaders
    from stock_analyzer.reporting.health import PortfolioHealth, _market_checks_html

    db = str(tmp_path / "d.db")
    with get_session(db) as s:
        s.add(
            IbdRating(
                ticker="FELL", as_of="2026-09-25", composite=60, rs_rating=70, industry="Chips"
            )
        )
        s.add(
            IbdRating(
                ticker="WEAK", as_of="2026-09-25", composite=40, rs_rating=12, industry="Cars"
            )
        )
        s.add(
            IbdRating(
                ticker="FINE", as_of="2026-09-25", composite=95, rs_rating=90, industry="Chips"
            )
        )
        s.add(
            IbdRating(
                ticker="LEAD", as_of="2026-09-25", composite=99, rs_rating=99, industry="Chips"
            )
        )
        s.add(IbdHistory(day="2026-08-25", ticker="FELL", composite=92))
        s.add(IbdHistory(day="2026-08-25", ticker="FINE", composite=96))
        s.commit()
    monkeypatch.setattr(leaders.bar_store, "load", lambda t: None)
    checks = {
        c["ticker"]: c
        for c in leaders.holding_checks(db, ["FELL", "WEAK", "FINE"], today=date(2026, 9, 27))
    }
    assert set(checks) == {"FELL", "WEAK"}
    assert checks["FELL"]["reasons"] == ["Composite 92 → 60 in a month"]
    assert checks["FELL"]["alternative"]["ticker"] == "LEAD"  # FINE is held
    assert "RS Rating 12" in checks["WEAK"]["reasons"][0] and checks["WEAK"]["alternative"] is None
    html = _market_checks_html(PortfolioHealth(market_checks=list(checks.values())))
    assert "not a sell signal" in html and "LEAD (Composite 99, Chips)" in html


# --- sector direction, caps in the rebalancer, and the group chart -------------------------


def _srow(t, industry, above, before, comp=50, six=0.1):
    return {
        "ticker": t,
        "industry": industry,
        "above_50": above,
        "above_50_before": before,
        "composite": comp,
        "six_month": six,
    }


def test_sector_levels_follow_the_etf_and_breadth():
    healthy = [_srow(f"T{i}", "Software", True, True, comp=95, six=0.3) for i in range(5)]
    thinning = [
        _srow(f"H{i}", "Drugs", i < 2, True, six=0.05) for i in range(5)
    ]  # 40% now, 100% before
    broken = [_srow(f"U{i}", "Power", False, False, six=-0.1) for i in range(5)]
    up = _bars(_trend(260, 100, 130))
    down = _bars(np.concatenate([_trend(200, 100, 130), _trend(60, 130, 100)]))
    out = {
        x["sector"]: x
        for x in ibd.sector_direction(
            healthy + thinning + broken,
            sectors={
                **{r["ticker"]: "Technology" for r in healthy},
                **{r["ticker"]: "Healthcare" for r in thinning},
                **{r["ticker"]: "Utilities" for r in broken},
            },
            etfs={"XLK": up, "XLV": up, "XLU": down},
        )
    }
    assert out["Technology"]["status"] == "Leading" and out["Technology"]["rank"] == 1
    assert out["Technology"]["leaders"] == 5
    assert (
        out["Healthcare"]["status"] == "Caution"
        and "breadth fell" in out["Healthcare"]["reasons"][0]
    )
    assert out["Utilities"]["status"] == "Correction"


def test_semiconductors_get_their_own_line_inside_technology():
    semis = [_srow(f"S{i}", "Semiconductors", True, True, six=0.4) for i in range(4)]
    out = {
        x["sector"]: x
        for x in ibd.sector_direction(
            semis, sectors={r["ticker"]: "Technology" for r in semis}, etfs={}
        )
    }
    assert set(out) == {"Technology", "Semiconductors"}
    assert out["Semiconductors"]["rank"] is None and out["Semiconductors"]["status"] == "Leading"


def test_the_rebalancer_prompt_follows_the_caps_in_settings():
    from stock_analyzer.discover.rebalancer_prompt import (
        _EXAMPLE_OVER_CAP,
        _POSITION_PHRASES,
        _SECTOR_RULE,
        _build_rebalancer_instructions,
    )

    default = _build_rebalancer_instructions()
    assert all(p in default for p in _POSITION_PHRASES) and _EXAMPLE_OVER_CAP in default
    mine = _build_rebalancer_instructions(max_position_pct=35, max_sector_pct=100)
    assert not any(p in mine for p in _POSITION_PHRASES)
    assert "No single position should exceed ~35% of post-rebalance" in mine
    assert _SECTOR_RULE not in mine and "Sector WEIGHT alone is not a" in mine
    assert "MARKET LEADERSHIP" in mine
    capped = _build_rebalancer_instructions(max_sector_pct=45)
    assert "any sector >45% of portfolio" in capped


def test_a_held_sector_in_correction_is_a_decision_with_its_weakest_names():
    from stock_analyzer.reporting.health import PortfolioHealth, _sector_trends_html, decision_items

    h = PortfolioHealth(
        sector_trends=[
            {
                "sector": "Semiconductors",
                "status": "Correction",
                "reasons": "SMH below its 200-day average",
                "rank": None,
                "day": "2026-09-25",
                "pct": 55.0,
                "holdings": [
                    {"ticker": "MRVL", "composite": 30},
                    {"ticker": "AMD", "composite": 99},
                ],
            },
            {
                "sector": "Technology",
                "status": "Leading",
                "reasons": "",
                "rank": 1,
                "day": "2026-09-25",
                "pct": 70.0,
                "holdings": [],
            },
        ]
    )
    items = [i for i in decision_items(h) if i["label"].startswith("SECTOR")]
    assert len(items) == 1 and items[0]["priority"] == 2
    assert "MRVL (Composite 30)" in items[0]["text"] and "55% of your holdings" in items[0]["text"]
    line = _sector_trends_html(h)
    assert "Semiconductors Correction" in line and "Technology Leading (#1)" in line


def test_the_group_chart_is_an_equal_weight_index_against_spy(monkeypatch):
    from types import SimpleNamespace

    from stock_analyzer.reporting import leaders

    n = 300
    frames = {
        "SPY": _bars(np.full(n, 100.0)),
        "A": _bars(np.linspace(100, 200, n)),
        "B": _bars(np.full(n, 50.0)),
        "C": _bars(np.linspace(10, 5, n)),
    }
    monkeypatch.setattr(
        leaders.bar_store,
        "load",
        lambda t: SimpleNamespace(frame=frames[t]) if t in frames else None,
    )
    rows = [
        SimpleNamespace(ticker="A", industry="Chips", group_rank=1),
        SimpleNamespace(ticker="B", industry="Chips", group_rank=1),
        SimpleNamespace(ticker="C", industry="Retail", group_rank=40),
    ]
    gc = leaders.group_chart(rows, held={"C"})
    by = {g["name"]: g for g in gc["groups"]}
    assert len(gc["dates"]) == leaders.GROUP_CHART_SESSIONS + 1 and gc["spy"][-1] == 100.0
    assert by["Chips"]["top"] and by["Chips"]["series"][0] == 100.0
    assert 100 < by["Chips"]["series"][-1] < 200  # half of A's rise, B flat
    assert by["Retail"]["held"] and by["Retail"]["series"][-1] < 100


# --- company profile and top news ------------------------------------------------------------


def test_news_is_fetched_for_holdings_top_leaders_and_near_buy_points():
    from stock_analyzer.cli.ibd import NEWS_TOP, featured

    rows = [
        {
            "ticker": f"T{i}",
            "composite": 99 if i < 30 else 98,
            "base_status": None,
            "vs_pivot": None,
        }
        for i in range(60)
    ]
    rows.append(
        {"ticker": "NEAR", "composite": 75, "base_status": "below pivot", "vs_pivot": -0.03}
    )
    rows.append({"ticker": "FAR", "composite": 75, "base_status": "below pivot", "vs_pivot": -0.20})
    rows.append({"ticker": "MINE", "composite": 5, "base_status": None, "vs_pivot": None})
    got = featured(rows, {"MINE", "UNRATED"})
    assert got[0] == "MINE" and "UNRATED" not in got
    assert set(got) == {"MINE", "NEAR", *(f"T{i}" for i in range(NEWS_TOP))}


def test_dashboard_news_keeps_only_items_about_the_company():
    from stock_analyzer.data import ticker_news

    class FakeFinnhub:
        def company_news(self, symbol, _from, to):
            return [
                {
                    "headline": "Nvidia beats earnings, raises guidance",
                    "url": "https://a.com/1",
                    "datetime": 1790000000,
                    "summary": "Nvidia said...",
                    "source": "Reuters",
                },
                {
                    "headline": "5 stocks to watch this week",
                    "url": "https://b.com/2",
                    "datetime": 1790000000,
                    "summary": "Boeing, AbbVie...",
                    "source": "Yahoo",
                },
            ]

    got = ticker_news.dashboard_news(["NVDA"], {"NVDA": "NVIDIA Corporation"}, client=FakeFinnhub())
    assert [i["title"] for i in got["NVDA"]] == ["Nvidia beats earnings, raises guidance"]
    assert got["NVDA"][0]["source"] == "Reuters" and got["NVDA"][0]["published"]


def test_company_profiles_come_from_the_fundamentals_cache(monkeypatch):
    from stock_analyzer.reporting import leaders

    cache = {
        "NVDA": {
            "value": {
                "name": "NVIDIA",
                "summary": "Makes GPUs.",
                "market_cap": 4e12,
                "debt_to_equity": 0.1,
            }
        }
    }
    monkeypatch.setattr(leaders.fetch_cache, "entries", lambda kind: cache)
    got = leaders.company_profiles(["NVDA", "NONE"])
    assert got == {"NVDA": {"name": "NVIDIA", "summary": "Makes GPUs.", "market_cap": 4e12}}


def test_a_short_load_is_retried_then_fails_without_storing(monkeypatch, tmp_path):
    from sqlalchemy import select

    from stock_analyzer.cli import ibd as job
    from stock_analyzer.config import Settings
    from stock_analyzer.db.session import get_session
    from stock_analyzer.db.tables import IbdHistory, IbdRating

    n = 300
    bars = {f"S{i}": _bars(_trend(n, 50 + i, 150 - i)) for i in range(6)}
    bars["SPY"] = bars["QQQ"] = _bars(_trend(n, 100, 120))
    full = {f"S{i}": "Semiconductors" for i in range(6)}
    maps = [full]
    monkeypatch.setattr(job, "all_us_2b", lambda: tuple(f"S{i}" for i in range(6)))
    monkeypatch.setattr(job, "tracked_tickers", lambda db, today: [])
    monkeypatch.setattr(job.yf_gateway, "daily_bars_many", lambda names, start, what: dict(bars))
    monkeypatch.setattr(job, "industry_map", lambda: maps.pop(0))
    monkeypatch.setattr(job, "batch_eps", lambda names, refresh=(): {})
    monkeypatch.setattr(job.fetch_cache, "oldest", lambda kind, names: [])
    monkeypatch.setattr(job, "news_for", lambda db, tickers, today: 0)
    monkeypatch.setattr(job, "RETRY_WAIT_SECONDS", 0)
    db = str(tmp_path / "d.db")
    settings = Settings(discover_db_path=db)
    job.run(settings, today=date(2026, 9, 27))

    for k, b in bars.items():
        bars[k] = pl.concat([b, b.tail(1).with_columns(pl.col("date") + timedelta(days=1))])
    maps[:] = [{}, full]  # the industry map fails once, then loads
    job.run(settings, today=date(2026, 9, 28))
    kept = job.coverage_before(db)
    assert kept["with_industry"] == 6 and not maps

    for k, b in bars.items():
        bars[k] = pl.concat([b, b.tail(1).with_columns(pl.col("date") + timedelta(days=1))])
    maps[:] = [{}, {}]  # fails twice: nothing is stored
    with pytest.raises(RuntimeError, match="with_industry 0 vs 6"):
        job.run(settings, today=date(2026, 9, 29))
    with get_session(db) as s:
        assert {r.as_of for r in s.scalars(select(IbdRating)).all()} == {kept["as_of"]}
        assert len({h.day for h in s.scalars(select(IbdHistory)).all()}) == 2
