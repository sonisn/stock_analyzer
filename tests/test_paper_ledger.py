"""Paper-trading ledger: follow every run's picks with a fixed tranche and
compare with SPY. Temp SQLite DB + fake price history; no network."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import text

from stock_analyzer.db.repository import insert_pick, insert_run, insert_run_outputs
from stock_analyzer.db.session import get_session
from stock_analyzer.discover.paper_ledger import (
    Tranche,
    build_ledger,
    ledger_report_data,
    load_tranches,
    parse_weights,
)
from stock_analyzer.discover.report_sections import append_paper_ledger_section

D0 = date(2026, 6, 1)
TODAY = date(2026, 6, 30)


def _series(start_px: float, end_px: float, start: date = D0) -> pd.DataFrame:
    """Linear price path from `start` to TODAY; flat before `start`."""
    idx = pd.date_range(start - timedelta(days=10), TODAY, freq="D")
    span = (TODAY - start).days
    closes = [
        start_px
        if ts.date() <= start
        else start_px + (end_px - start_px) * (ts.date() - start).days / span
        for ts in idx
    ]
    return pd.DataFrame({"Close": closes}, index=idx)


def _fetch(frames):
    return lambda ticker, start, end: frames.get(ticker)


# --- weights ---------------------------------------------------------------


def test_percent_allocations_normalized():
    text_ = "---\nTICKER: A\nAllocation: 30% of new capital\n---\nTICKER: B\nAllocation: 10%\n"
    assert parse_weights(text_, ["A", "B"]) == {"A": 0.75, "B": 0.25}


def test_dollar_allocations_work_as_relative_weights():
    text_ = "TICKER: A\nAllocation: $1,500\nTICKER: B\nAllocation: $500\n"
    assert parse_weights(text_, ["A", "B"]) == {"A": 0.75, "B": 0.25}


def test_unparseable_or_partial_falls_back_to_equal_weight():
    assert parse_weights("", ["A", "B"]) == {"A": 0.5, "B": 0.5}
    partial = "TICKER: A\nAllocation: 30%\nTICKER: B\nRationale: x\n"
    assert parse_weights(partial, ["A", "B"]) == {"A": 0.5, "B": 0.5}


# --- loading tranches ----------------------------------------------------------


def _seed_run(db, run_at: str, tickers: list[str], sizer: str = "") -> None:
    with get_session(db) as session:
        run_id = insert_run(
            session,
            universe_size=1,
            survivors=1,
            picks=len(tickers),
            opus_model="o",
            sonnet_model="s",
            cash_budget=None,
        )
        session.exec(
            text("UPDATE runs SET run_at = :r WHERE id = :i"), params={"r": run_at, "i": run_id}
        )
        for rank, t in enumerate(tickers, start=1):
            insert_pick(
                session,
                run_id,
                rank=rank,
                ticker=t,
            )
        insert_run_outputs(
            session, run_id, ranker_full="", redteam_full="", sizer_full=sizer, holdings_summary=""
        )


def test_one_tranche_per_day_latest_run_wins(tmp_path):
    db = str(tmp_path / "t.db")
    _seed_run(db, "2026-06-01T01:00:00", ["OLD"])
    _seed_run(db, "2026-06-01T09:00:00", ["NEW"])
    _seed_run(
        db,
        "2026-06-02T09:00:00",
        ["X", "Y"],
        "TICKER: X\nAllocation: 60%\nTICKER: Y\nAllocation: 40%\n",
    )
    tranches = load_tranches(db)
    assert [(t.run_date, t.weights) for t in tranches] == [
        (date(2026, 6, 1), {"NEW": 1.0}),
        (date(2026, 6, 2), {"X": 0.6, "Y": 0.4}),
    ]


def test_runs_without_picks_are_ignored(tmp_path):
    db = str(tmp_path / "t.db")
    _seed_run(db, "2026-06-01T01:00:00", [])
    assert load_tranches(db) == []


# --- simulation ---------------------------------------------------------------


def test_single_tranche_returns_vs_spy():
    tranches = [Tranche(1, D0, {"A": 0.5, "B": 0.5})]
    frames = {"SPY": _series(100, 105), "A": _series(50, 60), "B": _series(20, 20)}
    report = build_ledger(tranches, today=TODAY, fetch=_fetch(frames))
    # A +20% on half, B flat on half -> +10%; SPY +5%
    assert report.strategy_return_pct == pytest.approx(10.0)
    assert report.spy_return_pct == pytest.approx(5.0)
    assert report.invested == 1000.0
    (t,) = report.tranches
    assert t.strategy_return_pct == pytest.approx(10.0)


def test_second_tranche_adds_capital_from_its_own_entry_date():
    later = D0 + timedelta(days=14)
    tranches = [Tranche(1, D0, {"A": 1.0}), Tranche(2, later, {"A": 1.0})]
    frames = {"SPY": _series(100, 100), "A": _series(100, 130)}
    report = build_ledger(tranches, today=TODAY, fetch=_fetch(frames))
    assert report.invested == 2000.0
    # Before the second entry only $1,000 is in the book.
    before = [
        inv for d, inv in zip(report.curve_dates, report.invested_values, strict=True) if d < later
    ]
    assert before and set(before) == {1000.0}
    # Second tranche bought A at a higher price, so it earned less.
    first, second = report.tranches
    assert first.strategy_return_pct > second.strategy_return_pct > 0


def test_ticker_without_history_is_dropped_and_weights_renormalized():
    tranches = [Tranche(1, D0, {"A": 0.5, "GONE": 0.5})]
    frames = {"SPY": _series(100, 100), "A": _series(10, 11)}
    report = build_ledger(tranches, today=TODAY, fetch=_fetch(frames))
    assert report.skipped_tickers == ["GONE"]
    assert report.strategy_return_pct == pytest.approx(10.0)
    assert report.tranches[0].tickers == ["A"]


def test_run_after_last_close_is_not_entered():
    tranches = [Tranche(1, D0, {"A": 1.0}), Tranche(2, TODAY + timedelta(days=1), {"A": 1.0})]
    frames = {"SPY": _series(100, 100), "A": _series(10, 10)}
    report = build_ledger(tranches, today=TODAY, fetch=_fetch(frames))
    assert len(report.tranches) == 1
    assert report.invested == 1000.0


def test_no_spy_history_returns_empty_report():
    report = build_ledger([Tranche(1, D0, {"A": 1.0})], today=TODAY, fetch=_fetch({}))
    assert ledger_report_data(report) is None


def test_curve_is_downsampled_but_ends_on_latest_close():
    start = TODAY - timedelta(days=400)
    frames = {"SPY": _series(100, 110, start), "A": _series(10, 12, start)}
    report = build_ledger([Tranche(1, start, {"A": 1.0})], today=TODAY, fetch=_fetch(frames))
    assert len(report.curve_dates) <= 122
    assert report.curve_dates[-1] == TODAY


# --- report section ------------------------------------------------------------


def test_section_headline_says_behind_when_picks_lag_spy():
    frames = {"SPY": _series(100, 110), "A": _series(10, 10)}
    data = ledger_report_data(
        build_ledger([Tranche(1, D0, {"A": 1.0})], today=TODAY, fetch=_fetch(frames))
    )
    sections: list = []
    append_paper_ledger_section(sections, data)
    assert [s.kind for s in sections] == ["heading", "para", "equity_curve", "table"]
    assert "10.0 points behind" in sections[1].text
    assert sections[3].table_rows[0][-1] == "-10.0"


def test_section_skipped_without_data():
    sections: list = []
    append_paper_ledger_section(sections, None)
    assert sections == []


# --- chart model -----------------------------------------------------------


def test_nice_ticks_are_round_and_cover_the_range():
    from stock_analyzer.discover.report_html import _nice_ticks

    ticks = _nice_ticks(-12.4, 4.9)
    assert ticks == [-15.0, -10.0, -5.0, 0.0, 5.0]


def test_chart_model_plots_return_on_invested_capital():
    from stock_analyzer.discover.report_html import ledger_chart_model

    model = ledger_chart_model(
        {
            "dates": ["2026-06-01", "2026-06-15"],
            "strategy": [1000.0, 2200.0],
            "benchmark": [1000.0, 1900.0],
            "invested": [1000.0, 2000.0],
        }
    )
    (picks_name, picks, _), (spy_name, spy, _) = model["series"]
    assert (picks_name, spy_name) == ("Picks", "SPY")
    assert picks == pytest.approx([0.0, 10.0])  # $2,200 on $2,000 in, not +120%
    assert spy == pytest.approx([0.0, -5.0])
    assert model["lo"] <= -5.0 and model["hi"] >= 10.0


def test_chart_model_rejects_mismatched_series():
    from stock_analyzer.discover.report_html import ledger_chart_model

    assert (
        ledger_chart_model({"dates": ["a", "b"], "strategy": [1.0], "invested": [1.0, 1.0]}) is None
    )
