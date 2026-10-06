"""Point-in-time SEC fundamentals and the factor studies — no network."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from stock_analyzer.model import factor_study as fs
from stock_analyzer.model import sec_history as sh


def _flow(start: str, end: str, val: float, filed: str) -> dict:
    return {"start": start, "end": end, "val": val, "filed": filed, "accn": filed}


def _body(revenue: list[dict], shares: list[dict] | None = None, **tags) -> dict:
    gaap = {"Revenues": {"units": {"USD": revenue}}}
    for tag, rows in tags.items():
        gaap[tag] = {"units": {"USD": rows}}
    dei = {}
    if shares is not None:
        dei["EntityCommonStockSharesOutstanding"] = {"units": {"shares": shares}}
    return {"facts": {"us-gaap": gaap, "dei": dei}}


# A calendar-year company: FY2023 revenue 400, H1 2023 180, H1 2024 220.
REVENUE = [
    _flow("2023-01-01", "2023-12-31", 400.0, "2024-02-01"),
    _flow("2023-01-01", "2023-06-30", 180.0, "2023-08-01"),
    _flow("2024-01-01", "2024-06-30", 220.0, "2024-08-01"),
    _flow("2022-01-01", "2022-12-31", 300.0, "2023-02-01"),
    _flow("2022-01-01", "2022-06-30", 140.0, "2022-08-01"),
]


def test_ttm_from_year_to_date_and_only_what_was_filed():
    known = sh.Known(sh._facts(_body(REVENUE), "us-gaap", ("Revenues",), "USD"))
    known.advance(date(2024, 6, 15))  # the H1 2024 10-Q is not out yet
    end, now, before = sh.latest_ttm(known, date(2024, 6, 15))
    assert (end, now) == (date(2023, 12, 31), 400.0)
    assert before == 300.0
    # Seven months on with nothing newer filed, the year is stale.
    assert sh.latest_ttm(known, date(2024, 7, 31)) is None
    known.advance(date(2024, 8, 31))
    end, now, before = sh.latest_ttm(known, date(2024, 8, 31))
    # 220 this half + 400 last year - 180 the same half a year earlier.
    assert (end, now, before) == (date(2024, 6, 30), 440.0, 340.0)


def test_a_restatement_counts_only_once_filed():
    rows = [*REVENUE, _flow("2023-01-01", "2023-12-31", 500.0, "2024-10-01")]
    known = sh.Known(sh._facts(_body(rows), "us-gaap", ("Revenues",), "USD"))
    known.advance(date(2024, 9, 1))
    assert sh.ttm(known.periods, date(2023, 12, 31)) == 400.0
    known.advance(date(2024, 10, 2))
    assert sh.ttm(known.periods, date(2023, 12, 31)) == 500.0


def test_raw_close_undoes_splits_and_dividends():
    # Traded 100 then 102, a $2 dividend, then a 2-for-1 split (50, 51).
    raw = [100.0, 102.0, 100.0, 50.0, 51.0]
    divs = [0.0, 0.0, 2.0, 0.0, 0.0]
    splits = [0.0, 0.0, 0.0, 2.0, 0.0]
    # Adjusted the way Yahoo does: split first, then the dividend factor.
    factor = (1 - 2.0 / 102.0) / 2
    adj = [raw[0] * factor, raw[1] * factor, raw[2] / 2, raw[3], raw[4]]
    bars = pl.DataFrame(
        {
            "date": [date(2024, 1, d) for d in range(1, 6)],
            "Close": adj,
            "Dividends": divs,
            "Stock Splits": splits,
        }
    )
    got = sh.raw_close(bars)["close"].to_list()
    assert np.allclose(got, raw)
    assert sh.splits_of(bars) == [(date(2024, 1, 4), 2.0)]


def test_share_count_is_carried_through_a_split_and_summed_across_classes():
    body = _body(
        REVENUE,
        shares=[
            {"end": "2024-07-15", "val": 100.0, "filed": "2024-08-01", "accn": "a"},
            {"end": "2024-07-15", "val": 30.0, "filed": "2024-08-01", "accn": "a"},  # class B
            {"end": "2024-07-15", "val": 30.0, "filed": "2024-08-01", "accn": "a"},  # repeated
        ],
    )
    shares = sh.Shares(body, [(date(2024, 8, 31), 4.0)])
    assert shares.at(date(2024, 7, 31)) is None
    assert shares.at(date(2024, 8, 30)) == 130.0
    assert shares.at(date(2024, 9, 30)) == 520.0


def test_sue_scales_the_latest_change_by_the_past_spread():
    quarters = [(date(2020 + i // 4, 3 * (i % 4) + 3, 28), 1.0 + 0.1 * i) for i in range(16)]
    quarters[-1] = (quarters[-1][0], quarters[-1][1] + 1.0)
    got = sh.sue(quarters, date(2023, 12, 31))
    # Every earlier change is +0.4 (zero spread) — the spread guard returns None.
    assert got is None
    quarters[5] = (quarters[5][0], quarters[5][1] + 0.2)
    got = sh.sue(quarters, date(2023, 12, 31))
    assert got is not None and got > 5
    assert sh.sue(quarters, date(2025, 1, 1)) is None  # stale


def test_newey_west_matches_plain_t_without_lags():
    x = np.array([0.1, 0.2, -0.1, 0.3, 0.0, 0.15])
    plain = x.mean() / (x.std() / math.sqrt(len(x)))
    assert math.isclose(fs.newey_west_t(x, 0), plain, rel_tol=1e-9)
    # Positively autocorrelated series: overlap inflates the plain t.
    y = np.repeat([0.1, -0.05, 0.2, 0.05], 6)
    assert fs.newey_west_t(y, 6) < fs.newey_west_t(y, 0)


def test_a_measure_that_ranks_outcomes_works_and_noise_does_not():
    rng = np.random.default_rng(0)
    rows = []
    for m in range(60):
        day = date(2015 + m // 12, m % 12 + 1, 1)
        for i in range(50):
            signal = rng.normal()
            rows.append(
                {
                    "date": day,
                    "ticker": f"T{i}",
                    "good": signal,
                    "noise": rng.normal(),
                    "fwd_126": 0.5 * signal + rng.normal(),
                }
            )
    frame = pl.DataFrame(rows)
    good = fs.evaluate(frame, "good", "fwd_126", 126)
    assert good.verdict == "works" and good.mean_ic > 0.3 and good.spread > 0
    assert fs.evaluate(frame, "noise", "fwd_126", 126).verdict == "no edge"
    flipped = frame.with_columns((-pl.col("good")).alias("bad"))
    assert fs.evaluate(flipped, "bad", "fwd_126", 126).verdict == "WRONG WAY"


def test_screen_measures_use_the_screens_own_rules():
    fund = pl.DataFrame(
        {
            "date": [date(2024, 1, 31)] * 2,
            "ticker": ["GOOD", "WEAK"],
            "market_cap": [5e10, 5e10],
            "revenue_growth": [0.30, 0.02],
            "operating_margin": [0.30, 0.05],
            "fcf_yield": [0.06, 0.01],
            "free_cash_flow": [1e9, 1e8],
            "operating_cash_flow": [2e9, 2e8],
            "debt_to_equity": [0.3, 1.0],
            "roe": [0.25, 0.05],
            "gross_profitability": [0.4, 0.1],
            "sue": [1.0, 0.0],
        }
    )
    out = fs.screen_measures(fund)
    assert out["passes_rules"].to_list() == [True, False]
    assert out["screen_points"][0] == 45.0  # full marks on all four parts
    assert out["low_debt"].to_list() == [-0.3, -1.0]


def test_a_verdict_that_fails_among_large_companies_is_flagged():
    rng = np.random.default_rng(1)
    rows = []
    for m in range(60):
        day = date(2015 + m // 12, m % 12 + 1, 1)
        for i in range(80):
            small = i < 40
            signal = rng.normal()
            # Only the small names' outcomes follow the measure.
            outcome = (0.5 * signal if small else 0.0) + rng.normal()
            rows.append(
                {"date": day, "ticker": f"T{i}", "m": signal, "fwd_126": outcome, "small": small}
            )
    frame = pl.DataFrame(rows)
    r = fs.evaluate(frame, "m", "fwd_126", 126)
    r.large = fs.evaluate(frame.filter(~pl.col("small")), "m", "fwd_126", 126)
    assert r.own_verdict == "works" and r.verdict == "survivorship?"
    r.large = fs.evaluate(frame, "m", "fwd_126", 126)
    assert r.verdict == "works"
