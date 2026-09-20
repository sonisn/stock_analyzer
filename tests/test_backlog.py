"""Signed orders not yet delivered — the one forward number that isn't a forecast.

On 2026-09-20 the daily email called AVGO's thesis broken while its
contracted book had gone $45.0B → $164.6B → $179.2B over two quarters,
and offered POWL as a tax-loss sale with its book up 71% over a year.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.data import backlog as bl

CIK = 1730168


def _facts(rows):
    return {"units": {"USD": rows}}


def _fact(end, val, filed, form="10-Q"):
    return {"end": end, "val": val, "filed": filed, "form": form}


AVGO_FACTS = _facts(
    [
        _fact("2025-05-04", 24.7e9, "2025-06-11"),
        _fact("2025-08-03", 27.5e9, "2025-09-10"),
        _fact("2025-11-02", 33.3e9, "2025-12-18", "10-K"),
        _fact("2026-02-01", 45.0e9, "2026-03-11"),
        _fact("2026-05-03", 164.6e9, "2026-06-09"),
        _fact("2026-08-02", 179.2e9, "2026-09-10"),
    ]
)


@pytest.fixture
def sec(monkeypatch):
    def _install(body):
        monkeypatch.setattr(bl, "load_ticker_cik_map", lambda: {"AVGO": CIK, "X": 1})
        monkeypatch.setattr(bl._HTTP, "get_json", lambda url: body)

    return _install


def test_the_latest_book_and_its_growth(sec):
    sec(AVGO_FACTS)
    rec = bl.fetch_rpo("AVGO", as_of=date(2026, 9, 20))
    assert rec["value"] == 179.2e9
    assert rec["period_end"] == "2026-08-02" and rec["filed"] == "2026-09-10"
    assert round(rec["qoq_pct"]) == 9  # 164.6 -> 179.2
    assert round(rec["yoy_pct"]) == 552  # against 27.5 a year earlier


def test_nothing_filed_after_the_as_of_date_is_used(sec):
    # The book AVGO disclosed on 2026-09-10 did not exist for anyone on
    # 2026-06-01; using it in a backtest would be look-ahead.
    sec(AVGO_FACTS)
    rec = bl.fetch_rpo("AVGO", as_of=date(2026, 6, 1))
    assert rec["value"] == 45.0e9
    assert rec["filed"] == "2026-03-11"


def test_an_amended_filing_replaces_the_quarter_it_restates(sec):
    sec(
        _facts(
            [
                _fact("2026-05-03", 164.6e9, "2026-06-09"),
                _fact("2026-05-03", 160.0e9, "2026-07-01", "10-Q/A"),
            ]
        )
    )
    rec = bl.fetch_rpo("AVGO", as_of=date(2026, 9, 20))
    assert rec["value"] == 160.0e9  # the later filing wins


def test_a_company_that_stopped_tagging_is_not_reported(sec):
    # ANET's last RPO fact is from 2022: stale, not a current book.
    sec(_facts([_fact("2022-12-31", 1.3e9, "2023-02-14", "10-K")]))
    assert bl.fetch_rpo("AVGO", as_of=date(2026, 9, 20)) is None


def test_a_company_with_no_concept_at_all_is_silent(monkeypatch):
    monkeypatch.setattr(bl, "load_ticker_cik_map", lambda: {"BE": 2})

    def boom(url):
        raise RuntimeError("404")

    monkeypatch.setattr(bl._HTTP, "get_json", boom)
    assert bl.fetch_rpo("BE") is None


def test_an_unknown_ticker_needs_no_request(monkeypatch):
    monkeypatch.setattr(bl, "load_ticker_cik_map", lambda: {})
    monkeypatch.setattr(bl._HTTP, "get_json", lambda url: pytest.fail("should not be called"))
    assert bl.fetch_rpo("NOPE") is None


def test_coverage_is_the_book_in_years_of_revenue():
    assert bl.coverage_years(179.2e9, 89.1e9) == pytest.approx(2.01, abs=0.01)
    assert bl.coverage_years(None, 89.1e9) is None
    assert bl.coverage_years(179.2e9, 0) is None


def test_the_note_dates_itself():
    rec = {
        "value": 2.4e9,
        "period_end": "2026-06-30",
        "filed": "2026-08-04",
        "yoy_pct": 71.0,
        "qoq_pct": 33.0,
    }
    note = bl.backlog_note(rec)
    assert "$2.4B" in note and "+71% over a year" in note
    assert "as of 2026-06-30" in note and "filed 2026-08-04" in note
    assert "1.5x trailing revenue" in bl.backlog_note(rec, revenue_ttm=1.6e9)
    assert bl.backlog_note(None) == ""


# --- what it says next to a sale suggestion ---------------------------------------


def test_a_growing_book_argues_against_a_broken_thesis():
    from stock_analyzer.reporting.health import build_portfolio_health, decision_items

    health = build_portfolio_health(
        {"IRA": [{"ticker": "AVGO", "units": 162, "price": 357.61}]},
        backlog={
            "AVGO": {
                "value": 179.2e9,
                "period_end": "2026-08-02",
                "filed": "2026-09-10",
                "yoy_pct": 552.0,
                "qoq_pct": 9.0,
            }
        },
        held_thesis_checks=lambda held: [
            {
                "ticker": "AVGO",
                "status": "BROKEN",
                "return_pct": -17.0,
                "signals": [{"text": "analysts cutting estimates", "severity": "warning"}],
            }
        ],
    )
    text = next(i for i in decision_items(health) if i["label"] == "BROKEN")["text"]
    assert "Against that, the order book is growing" in text
    assert "$179.2B" in text and "filed 2026-09-10" in text


def test_a_shrinking_book_confirms_the_sale_instead():
    from stock_analyzer.reporting.health import build_portfolio_health, decision_items

    health = build_portfolio_health(
        {"IRA": [{"ticker": "X", "units": 100, "price": 50.0}]},
        backlog={
            "X": {
                "value": 1.0e9,
                "period_end": "2026-06-30",
                "filed": "2026-08-01",
                "yoy_pct": -35.0,
                "qoq_pct": -10.0,
            },
        },
        held_thesis_checks=lambda held: [
            {
                "ticker": "X",
                "status": "BROKEN",
                "return_pct": -40.0,
                "signals": [{"text": "estimates cut", "severity": "warning"}],
            }
        ],
    )
    text = next(i for i in decision_items(health) if i["label"] == "BROKEN")["text"]
    assert "The order book agrees: it is shrinking" in text


def test_a_book_that_barely_moved_says_nothing():
    from stock_analyzer.reporting.health import backlog_clause, build_portfolio_health

    health = build_portfolio_health(
        {},
        backlog={
            "X": {"value": 1e9, "period_end": "2026-06-30", "filed": "2026-08-01", "yoy_pct": 2.0}
        },
    )
    assert backlog_clause(health, "X") == ""
    assert backlog_clause(health, "NOTHELD") == ""


def test_the_table_ranks_by_the_fastest_growing_book():
    from stock_analyzer.reporting.health import build_portfolio_health, render_backlog_html

    health = build_portfolio_health(
        {},
        backlog={
            "POWL": {
                "value": 2.4e9,
                "period_end": "2026-06-30",
                "filed": "2026-08-04",
                "yoy_pct": 71.0,
                "qoq_pct": 33.0,
            },
            "AVGO": {
                "value": 179.2e9,
                "period_end": "2026-08-02",
                "filed": "2026-09-10",
                "yoy_pct": 552.0,
                "qoq_pct": 9.0,
            },
        },
    )
    html = render_backlog_html(health)
    assert "Contracted book (order backlog)" in html
    assert html.index("AVGO") < html.index("POWL")
    assert "$179.2B" in html and "+552%" in html


# --- the discover pipeline sees it too --------------------------------------------


class _FakeWorkflow:
    """Just enough of the discover workflow to exercise the step."""

    def __init__(self, survivors):
        self.state = {"survivors": [{"ticker": t} for t in survivors]}

    step_contracted_book = None  # bound below


def _step(monkeypatch, survivors, books, *, boom=False):
    from stock_analyzer.cli.discover import DiscoverPipeline
    from stock_analyzer.data import backlog

    def fake_batch(tickers, as_of=None):
        if boom:
            raise RuntimeError("SEC down")
        return {t: books[t] for t in tickers if t in books}

    monkeypatch.setattr(backlog, "batch_rpo", fake_batch)
    wf = _FakeWorkflow(survivors)
    return DiscoverPipeline.step_contracted_book(wf, None), wf


def test_the_step_fetches_books_for_survivors(monkeypatch):
    books = {
        "AVGO": {"value": 179.2e9, "yoy_pct": 552.0},
        "TSLA": {"value": 10.1e9, "yoy_pct": -3.0},
    }
    out, wf = _step(monkeypatch, ["AVGO", "TSLA", "BE"], books)
    assert set(wf.state["contracted_book"]) == {"AVGO", "TSLA"}
    assert "2/3 tag one" in out.content
    assert "1 growing" in out.content  # TSLA's is shrinking


def test_no_survivors_needs_no_request(monkeypatch):
    out, wf = _step(monkeypatch, [], {})
    assert wf.state["contracted_book"] == {}
    assert "no survivors" in out.content


def test_a_failed_fetch_does_not_stop_the_run(monkeypatch):
    out, wf = _step(monkeypatch, ["AVGO"], {}, boom=True)
    assert wf.state["contracted_book"] == {}
    assert "0/1" in out.content


def test_the_analyst_is_told_what_a_missing_book_means():
    from stock_analyzer.discover.analyst import ANALYST_INSTRUCTIONS

    text = ANALYST_INSTRUCTIONS
    assert "contracted_book" in text
    assert "not a forecast" in text
    # the trap: absence must never read as bad news
    assert "never evidence against" in text
