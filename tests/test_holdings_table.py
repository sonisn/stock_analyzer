"""_aggregate_holdings / _holdings_table_rows: same aggregation the LLM
prompt text (_holdings_summary) already relied on, reshaped as table rows
for the report instead of a monospace bullet list."""

from __future__ import annotations

from stock_analyzer.cli.discover import (
    _aggregate_holdings,
    _holdings_summary,
    _holdings_table_rows,
)


def _holdings(*positions: tuple[str, str, float, float]) -> dict:
    """positions: (account, ticker, units, avg_price)."""
    out: dict[str, list[dict]] = {}
    for account, ticker, units, avg in positions:
        out.setdefault(account, []).append(
            {"ticker": ticker, "units": units, "average_purchase_price": avg}
        )
    return out


def test_aggregates_across_accounts():
    holdings = _holdings(
        ("IRA", "NVDA", 10, 100.0),
        ("Taxable", "NVDA", 5, 200.0),
    )
    agg = _aggregate_holdings(holdings)
    assert agg["NVDA"]["units"] == 15
    assert agg["NVDA"]["cost"] == 10 * 100.0 + 5 * 200.0


def test_table_rows_match_summary_text_values():
    holdings = _holdings(("IRA", "NVDA", 10, 100.0))
    rows = _holdings_table_rows(holdings)
    assert rows == [["NVDA", "10", "$100.00", "$1,000"]]
    # Same underlying numbers as the LLM-prompt text form.
    assert "NVDA: 10 shares @ avg $100.00" in _holdings_summary(holdings)


def test_zero_units_position_skipped():
    holdings = _holdings(("IRA", "DEAD", 0, 50.0))
    assert _aggregate_holdings(holdings) == {}
    assert _holdings_table_rows(holdings) == []


def test_missing_ticker_skipped():
    holdings = {"IRA": [{"ticker": None, "units": 10, "average_purchase_price": 5.0}]}
    assert _holdings_table_rows(holdings) == []


def test_empty_holdings_returns_empty():
    assert _holdings_table_rows({}) == []
    assert _holdings_summary({}) == ""


def test_rows_sorted_by_ticker():
    holdings = _holdings(("IRA", "TSLA", 1, 400.0), ("IRA", "AMD", 1, 300.0))
    rows = _holdings_table_rows(holdings)
    assert [r[0] for r in rows] == ["AMD", "TSLA"]
