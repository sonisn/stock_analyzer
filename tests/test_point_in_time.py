"""Fundamentals as they were known on the day.

The bias this removes, measured against the live API on 2026-09-20: with
`asReported` off, `asof:2025-05-01` returns NVDA's quarter ending
2025-04-27 — filed 2025-05-28, four weeks after the as-of date. With it
on, the same request returns the quarter ending 2025-01-26, filed
2025-02-26.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_analyzer.model.fundamental_features import (
    FUNDAMENTAL_FEATURES,
    align_to_dates,
    fetch_history,
    fill_cross_section,
    month_ends,
    to_frame,
)


def test_as_reported_is_sent_for_point_in_time_requests(monkeypatch):
    from stock_analyzer.data import wisesheets

    seen = {}

    def fake_get(path, params):
        seen.update(params)
        return {"data": []}

    monkeypatch.setattr(wisesheets, "api_key", lambda: "wsh_test")
    monkeypatch.setattr(wisesheets, "_get", fake_get)
    wisesheets.fetch_point_in_time_ratios(["NVDA"], date(2025, 5, 1))
    assert seen["asReported"] == "true"
    assert seen["period"] == "asof:2025-05-01"


def test_ratios_come_from_one_filing(monkeypatch):
    from stock_analyzer.data import wisesheets

    monkeypatch.setattr(wisesheets, "api_key", lambda: "wsh_test")
    monkeypatch.setattr(
        wisesheets,
        "_get",
        lambda path, params: {
            "data": [
                {
                    "ticker": "NVDA",
                    "metric": "revenue",
                    "value": "130500000000",
                    "periodEnd": "2025-01-26",
                },
                {
                    "ticker": "NVDA",
                    "metric": "gross_profit",
                    "value": "97858000000",
                    "periodEnd": "2025-01-26",
                },
                {
                    "ticker": "NVDA",
                    "metric": "net_income",
                    "value": "72880000000",
                    "periodEnd": "2025-01-26",
                },
                # GOOGL tags no gross profit — a real gap, not an error.
                {
                    "ticker": "GOOGL",
                    "metric": "revenue",
                    "value": "350000000000",
                    "periodEnd": "2024-12-31",
                },
                {
                    "ticker": "GOOGL",
                    "metric": "net_income",
                    "value": "100118000000",
                    "periodEnd": "2024-12-31",
                },
            ]
        },
    )
    out, answered = wisesheets.fetch_point_in_time_ratios(["NVDA", "GOOGL"], date(2025, 5, 1))
    assert answered == ["NVDA", "GOOGL"]
    assert out["NVDA"]["gross_margin_pit"] == pytest.approx(0.7499, abs=1e-3)
    assert out["NVDA"]["net_margin_pit"] == pytest.approx(0.5584, abs=1e-3)
    assert "gross_margin_pit" not in out["GOOGL"]
    assert out["GOOGL"]["net_margin_pit"] == pytest.approx(0.2861, abs=1e-3)


def test_monthly_sampling_spans_the_period():
    dates = month_ends(date(2025, 1, 15), date(2025, 6, 30))
    assert dates[0] == date(2025, 1, 31) and dates[-1] == date(2025, 6, 30)
    assert len(dates) == 6


def test_the_cache_is_used_before_the_api(tmp_path, monkeypatch):
    from stock_analyzer.data import wisesheets

    calls = []
    monkeypatch.setattr(wisesheets, "is_configured", lambda: True)
    monkeypatch.setattr(wisesheets, "quota", lambda: {"monthly_remaining": 5000})
    monkeypatch.setattr(
        wisesheets,
        "fetch_point_in_time_ratios",
        lambda tickers, as_of: (
            calls.append((as_of, tuple(tickers)))
            or {t: {"return_on_equity_pit": 0.7} for t in tickers},
            list(tickers),
        ),
    )
    dates = [date(2025, 1, 31), date(2025, 2, 28)]
    first = fetch_history(["NVDA"], dates, cache_dir=str(tmp_path))
    assert len(calls) == 2 and len(first) == 2
    second = fetch_history(["NVDA"], dates, cache_dir=str(tmp_path))
    assert len(calls) == 2  # nothing re-fetched
    assert second == first


def test_a_wider_universe_fetches_only_the_names_the_cache_lacks(tmp_path, monkeypatch):
    # The defect this pins: keyed on the date alone, a file written for 20
    # tickers was served whole to a 500-ticker run, which then
    # median-filled 96% of its rows and measured nothing.
    from stock_analyzer.data import wisesheets

    calls = []
    monkeypatch.setattr(wisesheets, "is_configured", lambda: True)
    monkeypatch.setattr(wisesheets, "quota", lambda: {"monthly_remaining": 5000})
    monkeypatch.setattr(
        wisesheets,
        "fetch_point_in_time_ratios",
        lambda tickers, as_of: (
            calls.append(tuple(tickers)) or {t: {"return_on_equity_pit": 0.5} for t in tickers},
            list(tickers),
        ),
    )
    dates = [date(2025, 1, 31)]
    fetch_history(["AAA", "BBB"], dates, cache_dir=str(tmp_path))
    assert calls == [("AAA", "BBB")]

    out = fetch_history(["AAA", "BBB", "CCC", "DDD"], dates, cache_dir=str(tmp_path))
    assert calls[-1] == ("CCC", "DDD")  # only the new names
    assert sorted(out[date(2025, 1, 31)]) == ["AAA", "BBB", "CCC", "DDD"]

    fetch_history(["AAA", "CCC"], dates, cache_dir=str(tmp_path))
    assert len(calls) == 2  # everything already known


def test_a_ticker_the_api_does_not_cover_is_not_re_requested(tmp_path, monkeypatch):
    from stock_analyzer.data import wisesheets

    calls = []
    monkeypatch.setattr(wisesheets, "is_configured", lambda: True)
    monkeypatch.setattr(wisesheets, "quota", lambda: {"monthly_remaining": 5000})
    # TSM is a foreign private issuer: asked for, never returned.
    monkeypatch.setattr(
        wisesheets,
        "fetch_point_in_time_ratios",
        lambda tickers, as_of: (
            calls.append(tuple(tickers))
            or {t: {"return_on_equity_pit": 0.5} for t in tickers if t != "TSM"},
            list(tickers),
        ),
    )
    dates = [date(2025, 1, 31)]
    fetch_history(["NVDA", "TSM"], dates, cache_dir=str(tmp_path))
    out = fetch_history(["NVDA", "TSM"], dates, cache_dir=str(tmp_path))
    assert len(calls) == 1  # the absence is cached too
    assert "TSM" not in out[date(2025, 1, 31)]


def test_a_date_that_returns_nothing_is_retried_next_run(tmp_path, monkeypatch):
    # A denied date (the plan's 5-year window) or an outage must not be
    # cached as "asked and answered".
    from stock_analyzer.data import wisesheets

    calls = []
    monkeypatch.setattr(wisesheets, "is_configured", lambda: True)
    monkeypatch.setattr(wisesheets, "quota", lambda: {"monthly_remaining": 5000})
    monkeypatch.setattr(
        wisesheets,
        "fetch_point_in_time_ratios",
        lambda tickers, as_of: (calls.append(as_of) or {}, []),
    )
    dates = [date(2021, 10, 31)]
    assert fetch_history(["NVDA"], dates, cache_dir=str(tmp_path)) == {}
    fetch_history(["NVDA"], dates, cache_dir=str(tmp_path))
    assert len(calls) == 2


def test_a_thin_quota_stops_early_instead_of_draining_it(tmp_path, monkeypatch, caplog):
    from stock_analyzer.data import wisesheets

    calls = []
    monkeypatch.setattr(wisesheets, "is_configured", lambda: True)
    # 202 left, 200 reserved => 2 spare requests, 1 chunk each => 2 dates
    monkeypatch.setattr(wisesheets, "quota", lambda: {"monthly_remaining": 202})
    monkeypatch.setattr(
        wisesheets,
        "fetch_point_in_time_ratios",
        lambda tickers, as_of: (
            calls.append(as_of) or {"NVDA": {"return_on_equity_pit": 0.7}},
            list(tickers),
        ),
    )
    dates = [date(2025, m, 28) for m in range(1, 7)]
    fetch_history(["NVDA"], dates, cache_dir=str(tmp_path))
    assert len(calls) == 2
    assert calls == dates[-2:]  # the most recent dates, not the oldest


def test_values_are_carried_forward_never_backward():
    history = {
        date(2025, 1, 31): {"NVDA": {"return_on_equity_pit": 0.60}},
        date(2025, 3, 31): {"NVDA": {"return_on_equity_pit": 0.75}},
    }
    dates = pd.DatetimeIndex(["2025-01-10", "2025-02-14", "2025-03-14", "2025-04-11"])
    aligned = align_to_dates(to_frame(history), dates, ["NVDA"])
    values = aligned["return_on_equity_pit"].droplevel("ticker")
    # Before the first filing there is nothing to know.
    assert pd.isna(values.loc["2025-01-10"])
    # February reads January's filing; March still does; April reads March's.
    assert values.loc["2025-02-14"] == 0.60
    assert values.loc["2025-03-14"] == 0.60
    assert values.loc["2025-04-11"] == 0.75


def test_a_missing_ratio_becomes_the_days_median_not_a_dropped_row():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-03-14"), t) for t in ("A", "B", "GOOGL")],
        names=["date", "ticker"],
    )
    frame = pd.DataFrame({"gross_margin_pit": [0.40, 0.60, None]}, index=index)
    filled = fill_cross_section(frame)
    assert filled.loc[(pd.Timestamp("2025-03-14"), "GOOGL"), "gross_margin_pit"] == 0.50
    assert len(filled) == 3


def test_the_dataset_gains_the_columns_only_when_asked():
    from stock_analyzer.model.dataset import PricePanel, build_dataset

    dates = pd.bdate_range("2024-01-01", periods=320)
    frame = pd.DataFrame(
        {t: pd.Series(range(1, len(dates) + 1), index=dates, dtype=float) for t in ("AAA", "BBB")}
    )
    panel = PricePanel(frame, frame * 1.01, frame * 1000, frame["AAA"].copy())
    plain = build_dataset(panel)
    assert not [c for c in FUNDAMENTAL_FEATURES if c in plain.columns]

    fundamentals = pd.DataFrame(
        {"return_on_equity_pit": 0.5},
        index=pd.MultiIndex.from_product(
            [plain.index.levels[0], ["AAA", "BBB"]], names=["date", "ticker"]
        ),
    )
    joined = build_dataset(panel, fundamentals=fundamentals)
    assert "return_on_equity_pit" in joined.columns
    assert joined["return_on_equity_pit"].notna().all()


def test_leverage_and_equity_come_from_assets_when_equity_is_untagged(monkeypatch):
    # Measured at asof:2025-12-31: total_assets 10/10 and
    # total_liabilities 9/10, but total_equity only 3/10.
    from stock_analyzer.data import wisesheets

    monkeypatch.setattr(wisesheets, "api_key", lambda: "wsh_test")
    monkeypatch.setattr(
        wisesheets,
        "_get",
        lambda path, params: {
            "data": [
                {"ticker": "NVDA", "metric": "net_income", "value": "31910000000"},
                {"ticker": "NVDA", "metric": "total_assets", "value": "187000000000"},
                {"ticker": "NVDA", "metric": "total_liabilities", "value": "49000000000"},
            ]
        },
    )
    out = wisesheets.fetch_point_in_time_ratios(["NVDA"], date(2025, 12, 31))[0]["NVDA"]
    assert out["leverage_pit"] == pytest.approx(49 / 187, abs=1e-4)
    # equity = assets - liabilities when the filing never tagged equity
    assert out["return_on_equity_pit"] == pytest.approx(31.91 / (187 - 49), abs=1e-3)


def test_the_feature_names_are_the_ones_the_model_asks_for():
    from stock_analyzer.data.wisesheets import POINT_IN_TIME_RATIOS

    assert tuple(FUNDAMENTAL_FEATURES) == POINT_IN_TIME_RATIOS
    assert "leverage_pit" in FUNDAMENTAL_FEATURES


def test_a_chunk_lost_to_a_server_error_is_retried_not_cached_as_covered(tmp_path, monkeypatch):
    # 100 tickers x 6 metrics returns 503 in as-reported mode; 50 works.
    # A failed chunk must not be recorded as asked, or half a date
    # silently becomes a cross-section of medians.
    from stock_analyzer.data import wisesheets

    calls = []
    monkeypatch.setattr(wisesheets, "is_configured", lambda: True)
    monkeypatch.setattr(wisesheets, "quota", lambda: {"monthly_remaining": 5000})

    def half_fails(tickers, as_of):
        calls.append(tuple(tickers))
        answered = [t for t in tickers if t != "BBB"]  # BBB's chunk 503'd
        return {t: {"return_on_equity_pit": 0.5} for t in answered}, answered

    monkeypatch.setattr(wisesheets, "fetch_point_in_time_ratios", half_fails)
    dates = [date(2025, 1, 31)]
    fetch_history(["AAA", "BBB"], dates, cache_dir=str(tmp_path))
    fetch_history(["AAA", "BBB"], dates, cache_dir=str(tmp_path))
    assert calls[-1] == ("BBB",)  # asked again; AAA was not


def test_as_reported_requests_are_chunked_smaller(monkeypatch):
    from stock_analyzer.data import wisesheets

    sizes = []
    monkeypatch.setattr(wisesheets, "api_key", lambda: "wsh_test")
    monkeypatch.setattr(
        wisesheets,
        "_get",
        lambda path, params: sizes.append(len(params["tickers"].split(","))) or {"data": []},
    )
    many = [f"T{i:03d}" for i in range(120)]
    wisesheets.fetch_metrics(many, ["revenue"], period="asof:2025-01-31", as_reported=True)
    assert max(sizes) <= wisesheets.AS_REPORTED_MAX_TICKERS == 50
    sizes.clear()
    wisesheets.fetch_metrics(many, ["revenue"])  # the normal path keeps the full cap
    assert max(sizes) == 100
