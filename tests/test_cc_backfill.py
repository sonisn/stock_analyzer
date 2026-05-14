"""Tests for OptionWrite backfill from WRITE_CALL sizing strings."""
from __future__ import annotations

from datetime import date, datetime

from stock_analyzer.data.options_chain import OptionChain, OptionQuote
from stock_analyzer.discover.cc_backfill import (
    _parse_sizing,
    backfill_option_writes,
)
from stock_analyzer.discover.rebalance_schema import (
    OptionWrite,
    RebalanceAction,
    RebalancePlan,
)


def test_parse_sizing_canonical():
    assert _parse_sizing("1 contract $450C expiring 2026-06-18") == (
        1, 450.0, "2026-06-18"
    )


def test_parse_sizing_no_expiring_keyword():
    assert _parse_sizing("3 contracts $260C 2026-06-20") == (
        3, 260.0, "2026-06-20"
    )


def test_parse_sizing_decimal_strike():
    assert _parse_sizing("2 contracts $230.00C 2026-06-20") == (
        2, 230.0, "2026-06-20"
    )


def test_parse_sizing_with_comma():
    # Sometimes Opus formats large strikes with thousand separators.
    assert _parse_sizing("1 contract $1,250C 2026-07-18") == (
        1, 1250.0, "2026-07-18"
    )


def test_parse_sizing_rejects_garbage():
    assert _parse_sizing("not a sizing string") is None
    assert _parse_sizing("") is None
    assert _parse_sizing(None) is None  # type: ignore[arg-type]


def _chain(ticker: str, strike: float, expiry: str,
           bid: float = 3.0, ask: float = 3.2,
           delta: float = 0.36, iv: float = 0.30) -> OptionChain:
    return OptionChain(
        ticker=ticker, spot=100.0, asof=datetime.now(),
        calls=[OptionQuote(
            strike=strike, expiry=date.fromisoformat(expiry),
            bid=bid, ask=ask, iv=iv, delta=delta,
            open_interest=1000, volume=500,
        )],
        source="yfinance",
    )


def test_backfill_synthesizes_optionwrite_for_orphan_write_call():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[
            RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                            sizing="3 contracts $260C 2026-06-20"),
        ],
        option_writes=[],  # orphan
        full_text="…",
    )
    chains = {"NVDA": _chain("NVDA", 260.0, "2026-06-20",
                              bid=2.20, ask=2.40, delta=0.36, iv=0.29)}
    out = backfill_option_writes(plan, chains=chains)
    assert len(out.option_writes) == 1
    ow = out.option_writes[0]
    assert ow.ticker == "NVDA"
    assert ow.strike == 260.0
    assert ow.expiry == "2026-06-20"
    assert ow.contracts == 3
    assert ow.est_premium_per_share == 2.30  # mid of 2.20/2.40
    assert ow.delta == 0.36
    assert ow.assignment_probability == 0.36
    assert "backfilled" in ow.notes


def test_backfill_skips_when_already_present():
    """If Opus did populate option_writes for a ticker, don't double-write."""
    existing = OptionWrite(
        ticker="NVDA", strike=260.0, expiry="2026-06-20",
        contracts=3, est_premium_per_share=2.40,
        delta=0.36, assignment_probability=0.36, notes="from Opus",
    )
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                                  sizing="3 contracts $260C 2026-06-20")],
        option_writes=[existing],
        full_text="…",
    )
    chains = {"NVDA": _chain("NVDA", 260.0, "2026-06-20")}
    out = backfill_option_writes(plan, chains=chains)
    assert out.option_writes == [existing]
    assert out.option_writes[0].notes == "from Opus"  # not overwritten


def test_backfill_skips_unparseable_sizing():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                                  sizing="some opaque sizing description")],
        option_writes=[],
        full_text="…",
    )
    out = backfill_option_writes(plan, chains={"NVDA": _chain("NVDA", 260.0, "2026-06-20")})
    assert out.option_writes == []


def test_backfill_skips_when_no_chain_match():
    """Opus picked a strike that's not in our chain (off-cycle expiry,
    hallucinated strike, etc.). Backfill should NOT invent data."""
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                                  sizing="1 contract $999C 2026-06-20")],
        option_writes=[],
        full_text="…",
    )
    out = backfill_option_writes(plan, chains={"NVDA": _chain("NVDA", 260.0, "2026-06-20")})
    assert out.option_writes == []


def test_backfill_skips_when_no_chain_for_ticker():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[RebalanceAction(action="WRITE_CALL", ticker="MYSTERY",
                                  sizing="1 contract $100C 2026-06-20")],
        option_writes=[],
        full_text="…",
    )
    out = backfill_option_writes(plan, chains={})
    assert out.option_writes == []


def test_backfill_handles_five_orphan_write_calls():
    """Real production scenario: Opus emitted 5 WRITE_CALL actions but
    zero option_writes. Backfill should synthesize all 5."""
    tickers = [
        ("NVDA", 260.0), ("AVGO", 450.0), ("BE", 350.0),
        ("GOOGL", 420.0), ("TSLA", 460.0),
    ]
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[
            RebalanceAction(
                action="WRITE_CALL", ticker=t,
                sizing=f"1 contract ${k:.0f}C expiring 2026-06-20",
            )
            for t, k in tickers
        ],
        option_writes=[],
        full_text="…",
    )
    chains = {t: _chain(t, k, "2026-06-20") for t, k in tickers}
    out = backfill_option_writes(plan, chains=chains)
    assert len(out.option_writes) == 5
    assert {ow.ticker for ow in out.option_writes} == {t for t, _ in tickers}
