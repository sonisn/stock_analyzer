"""Facts that fill a block when the headlines are all syndication."""

from __future__ import annotations

from datetime import date

from stock_analyzer.discover.stock_facts import (
    company_facts,
    estimates_line,
    filings_line,
    insider_line,
)

FILINGS = [
    {"form": "10-Q", "filed_on": "2026-09-10", "url": "https://sec/10q"},
    {"form": "8-K", "filed_on": "2026-09-02", "url": "https://sec/8k"},
    {"form": "8-K", "filed_on": "2026-08-17", "url": "https://sec/8k-old"},
]


def test_filings_name_the_form_and_link_it():
    line = filings_line(FILINGS)
    assert line.startswith("10-Q filed Sep 10 — quarterly report (https://sec/10q)")
    assert "8-K filed Sep 02 — material event (https://sec/8k)" in line
    assert "8k-old" not in line  # only the two newest


def test_no_filings_no_line():
    assert filings_line([]) is None


def test_estimates_read_the_next_year_number():
    assert estimates_line({"next_year_up_30d": 6, "next_year_down_30d": 1}) == (
        "Next-year EPS: 6 up / 1 down in 30 days — analysts raising"
    )
    assert "cutting" in estimates_line({"next_year_up_30d": 1, "next_year_down_30d": 9})
    assert "holding" in estimates_line({"next_year_up_30d": 2, "next_year_down_30d": 2})
    assert estimates_line({"next_year_up_30d": 0, "next_year_down_30d": 0}) is None
    assert estimates_line(None) is None


def test_insider_line_only_when_there_was_activity():
    assert insider_line({"n_buys": 0, "n_sells": 0}) is None
    line = insider_line({"n_buys": 1, "n_sells": 3, "sell_value_usd": 2_100_000})
    assert line == "Form 4s, last 90 days: 1 buy(s), 3 sale(s) ($2.1M sold)"


def test_company_facts_skips_what_it_does_not_have():
    facts = company_facts(filings=FILINGS, revisions=None, insider=None)
    assert list(facts) == ["Filings"]
    assert company_facts() == {}


def test_a_bad_filing_date_does_not_break_the_block():
    assert filings_line([{"form": "8-K", "filed_on": None, "url": "https://sec/x"}]) is None


def test_labels_start_with_a_letter():
    """The email's field parser only starts a row on a letter."""
    facts = company_facts(
        filings=FILINGS,
        revisions={"next_year_up_30d": 3, "next_year_down_30d": 0},
        insider={"n_buys": 2, "n_sells": 0, "buy_value_usd": 500_000},
    )
    assert all(label[0].isalpha() for label in facts)
    assert set(facts) == {"Filings", "Estimates", "Insiders"}


def test_dates_are_read_as_dates_not_strings():
    assert "Sep 02" in filings_line([{"form": "8-K", "filed_on": "2026-09-02", "url": "u"}])
    assert date.fromisoformat("2026-09-02").strftime("%b %d") == "Sep 02"
