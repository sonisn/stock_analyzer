"""Cash-secured puts (the wheel's front half): eligibility, Greeks
fallback, prompt block, post-LLM backfill/validation, report data, and
the data plumbing (put chains, open short puts, recent picks)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stock_analyzer.data.options_chain import TradierChain, YFinanceChain
from stock_analyzer.discover.cc_eligibility import apply_earnings_filter
from stock_analyzer.discover.csp_eligibility import (
    build_csp_context_block,
    eligible_csp_tickers,
    fill_put_deltas,
    put_delta,
    puts_near_band,
)
from stock_analyzer.discover.csp_validation import backfill_csp_writes, validate_csp_writes
from stock_analyzer.discover.rebalance_csp import csp_report_data
from stock_analyzer.discover.rebalance_sections import append_csp_section
from stock_analyzer.discover.rebalancer import REBALANCER_INSTRUCTIONS
from stock_analyzer.discover.report_html import render_html_email
from stock_analyzer.discover.report_pdf import render_pdf
from stock_analyzer.models.market import OptionChain, OptionQuote
from stock_analyzer.models.portfolio import CspCandidate
from stock_analyzer.models.rebalance import CashSecuredPut, RebalanceAction, RebalancePlan

TODAY = date(2026, 9, 18)
EXP = TODAY + timedelta(days=35)  # inside the 30-45 DTE band


def _q(strike: float, delta: float | None, *, expiry: date = EXP, iv: float | None = 0.35):
    return OptionQuote(
        strike=strike,
        expiry=expiry,
        bid=1.90,
        ask=2.10,
        iv=iv,
        delta=delta,
        open_interest=500,
        volume=50,
    )


def _chain(ticker: str = "NVDA", spot: float = 160.0, puts=None) -> OptionChain:
    return OptionChain(
        ticker=ticker,
        spot=spot,
        asof=datetime(2026, 9, 18, 10),
        puts=puts if puts is not None else [_q(145.0, -0.15), _q(140.0, -0.10)],
        source="tradier",
    )


def _cand(ticker: str = "NVDA", max_cash: float = 25_000.0) -> CspCandidate:
    return CspCandidate(
        ticker=ticker,
        last_pick_run_at="2026-09-17",
        last_pick_rank=1,
        shares_held=0,
        max_csp_cash=max_cash,
    )


def _put(ticker="NVDA", strike=145.0, contracts=1, delta=-0.15, expiry=EXP) -> CashSecuredPut:
    return CashSecuredPut(
        ticker=ticker,
        strike=strike,
        expiry=expiry.isoformat(),
        contracts=contracts,
        est_premium_per_share=2.0,
        delta=delta,
    )


def _plan(puts, actions=None) -> RebalancePlan:
    if actions is None:
        actions = [RebalanceAction(action="SELL_PUT", ticker=p.ticker, sizing="x") for p in puts]
    return RebalancePlan(
        status="ACTION",
        aggressiveness_applied="balanced",
        actions=actions,
        full_text="plan",
        csp_writes=puts,
    )


def _validate(plan, *, eligible=None, chains=None, budget=100_000.0):
    return validate_csp_writes(
        plan,
        eligible=eligible if eligible is not None else {"NVDA": _cand()},
        chains=chains if chains is not None else {"NVDA": _chain()},
        cash_budget=budget,
        delta_min=0.10,
        delta_max=0.25,
        dte_min=30,
        dte_max=45,
        max_pct_total=0.80,
        today=TODAY,
    )


# --- schema ---------------------------------------------------------------


def test_cash_secured_put_math_and_delta_sign():
    cp = _put(strike=145.0, contracts=2, delta=0.15)  # LLM sent a positive delta
    assert cp.delta == -0.15
    assert cp.cash_reserved == 29_000.0
    assert cp.premium_usd == 400.0


def test_plan_accepts_sell_put_and_defaults_empty_csp_writes():
    plan = RebalancePlan(
        status="ACTION",
        aggressiveness_applied="balanced",
        actions=[RebalanceAction(action="SELL_PUT", ticker="NVDA", sizing="1 contract $145P")],
        full_text="x",
    )
    assert plan.csp_writes == []


# --- eligibility ----------------------------------------------------------


def test_eligibility_filters_and_dedups():
    picks = [
        ("NVDA", 3, "2026-09-10T00:00:00"),
        ("NVDA", 1, "2026-09-17T00:00:00"),  # newer — wins
        ("AAPL", 2, "2026-09-17T00:00:00"),  # round lot held → covered-call side
        ("TSLA", 4, "2026-09-17T00:00:00"),  # denylisted
        ("AMD", 5, "2026-09-17T00:00:00"),  # put already open
        ("INTC", 2, "2026-09-10T00:00:00"),  # thesis broken
        ("MSFT", 3, "2026-09-10T00:00:00"),  # 40 shares held — still a candidate
    ]
    out = eligible_csp_tickers(
        picks,
        positions={"AAPL": {"units": 150}, "MSFT": {"units": 40}},
        cash_budget=100_000.0,
        denylist=("tsla",),
        open_short_puts={"AMD": {"contracts": 1, "collateral_usd": 15_000.0}},
        thesis_status={"INTC": "BROKEN", "MSFT": "INTACT"},
    )
    assert list(out) == ["NVDA", "MSFT"]  # newest run first
    assert out["NVDA"].last_pick_rank == 1
    assert out["NVDA"].last_pick_run_at == "2026-09-17"
    assert out["NVDA"].max_csp_cash == 25_000.0
    assert out["MSFT"].shares_held == 40
    assert out["MSFT"].thesis_status == "INTACT"


def test_eligibility_needs_cash_and_caps_count():
    picks = [(f"T{i}", i, "2026-09-17") for i in range(1, 12)]
    assert eligible_csp_tickers(picks, positions={}, cash_budget=0.0, denylist=()) == {}
    out = eligible_csp_tickers(
        picks, positions={}, cash_budget=10_000.0, denylist=(), max_candidates=3
    )
    assert list(out) == ["T1", "T2", "T3"]


# --- Greeks fallback + chain filtering ------------------------------------


def test_put_delta_black_scholes():
    atm = put_delta(spot=100.0, strike=100.0, iv=0.30, days=35)
    far = put_delta(spot=100.0, strike=80.0, iv=0.30, days=35)
    assert atm is not None and -0.5 < atm < -0.4
    assert far is not None and -0.05 < far < 0.0
    assert put_delta(spot=100.0, strike=90.0, iv=0.0, days=35) is None


def test_fill_put_deltas_only_fills_missing():
    chain = _chain(puts=[_q(145.0, None), _q(140.0, -0.12)])
    filled = fill_put_deltas(chain, today=TODAY)
    assert filled.puts[0].delta is not None and filled.puts[0].delta < 0
    assert filled.puts[1].delta == -0.12


def test_puts_near_band_keeps_usable_rows():
    later = EXP + timedelta(days=7)
    chain = _chain(
        puts=[
            _q(155.0, -0.40),  # out of band
            _q(150.0, -0.25),
            _q(148.0, -0.20),
            _q(146.0, -0.18),
            _q(145.0, -0.15),  # 4th row of this expiry — capped
            _q(120.0, -0.02),  # out of band
            _q(130.0, None),  # no delta
            _q(144.0, -0.12, expiry=later),
        ]
    )
    rows = puts_near_band(chain, delta_min=0.10, delta_max=0.25)
    assert [(q.strike, q.expiry) for q in rows] == [
        (150.0, EXP),
        (148.0, EXP),
        (146.0, EXP),
        (144.0, later),
    ]
    affordable = puts_near_band(chain, delta_min=0.10, delta_max=0.25, max_collateral=14_600.0)
    assert [q.strike for q in affordable] == [146.0, 145.0, 144.0]


def test_earnings_filter_applies_to_puts():
    chain = _chain(puts=[_q(145.0, -0.15), _q(145.0, -0.15, expiry=EXP + timedelta(days=30))])
    filtered, _ = apply_earnings_filter(chain, earnings_date=EXP)
    assert [q.expiry for q in filtered.puts] == [EXP + timedelta(days=30)]


def test_context_block():
    block = build_csp_context_block(
        candidates={"NVDA": _cand(), "NOCHAIN": _cand("NOCHAIN")},
        chains={"NVDA": _chain()},
        earnings={},
        cash_budget=100_000.0,
        open_put_collateral=15_000.0,
        delta_min=0.10,
        delta_max=0.25,
        max_pct_per_put=0.25,
        max_pct_total=0.80,
    )
    assert "CASH-SECURED PUT CONTEXT" in block
    assert "TICKER: NVDA" in block and "NOCHAIN" not in block
    assert "$145.00 strike" in block
    assert "Already reserved by open short puts: $15,000" in block
    assert "Total collateral cap:    $80,000" in block
    empty = build_csp_context_block(
        candidates={"NOCHAIN": _cand("NOCHAIN")},
        chains={},
        earnings={},
        cash_budget=1.0,
        open_put_collateral=0.0,
        delta_min=0.1,
        delta_max=0.25,
        max_pct_per_put=0.25,
        max_pct_total=0.8,
    )
    assert empty == ""


def test_rebalancer_prompt_describes_puts():
    assert "CASH-SECURED PUTS" in REBALANCER_INSTRUCTIONS
    assert "|Δ| 0.10-0.25" in REBALANCER_INSTRUCTIONS
    assert "no options" not in REBALANCER_INSTRUCTIONS
    assert "LIQUIDITY GUARD (puts" in REBALANCER_INSTRUCTIONS


# --- validation -----------------------------------------------------------


def test_valid_put_kept_with_chain_values_and_canonical_sizing():
    cleaned, warnings = _validate(
        _plan([_put(contracts=2, delta=-0.2)]), eligible={"NVDA": _cand(max_cash=40_000.0)}
    )
    assert warnings == []
    (cp,) = cleaned.csp_writes
    assert cp.contracts == 2
    assert cp.delta == -0.15  # re-read from the chain row
    assert cp.est_premium_per_share == pytest.approx(2.0)  # chain mid
    assert cleaned.actions[0].sizing == f"2 contracts $145.00P {EXP.isoformat()}"


LATE = TODAY + timedelta(days=60)


@pytest.mark.parametrize(
    ("put", "chains", "reason"),
    [
        (_put(ticker="AAPL"), None, "not a put candidate"),
        (_put(strike=147.0), None, "not in the chain"),
        (_put(expiry=LATE), {"NVDA": _chain(puts=[_q(145.0, -0.15, expiry=LATE)])}, "60 days"),
        (_put(), {"NVDA": _chain(spot=140.0)}, "not below spot"),
        (_put(), {"NVDA": _chain(puts=[_q(145.0, -0.40)])}, "outside 0.10-0.25"),
    ],
)
def test_invalid_puts_dropped(put, chains, reason):
    cleaned, warnings = _validate(_plan([put]), chains=chains)
    assert cleaned.csp_writes == []
    assert not any(a.action == "SELL_PUT" for a in cleaned.actions)
    assert any(reason in w for w in warnings), warnings


def test_put_without_action_and_action_without_put_dropped():
    plan = _plan(
        [_put()],
        actions=[
            RebalanceAction(action="SELL_PUT", ticker="AMD", sizing="garbage"),
            RebalanceAction(action="BUY", ticker="MSFT", sizing="$1,000"),
        ],
    )
    cleaned, warnings = _validate(plan)
    assert cleaned.csp_writes == []
    assert [a.action for a in cleaned.actions] == ["BUY"]
    assert any("no matching SELL_PUT" in w for w in warnings)
    assert any("SELL_PUT on AMD dropped" in w for w in warnings)


def test_contracts_clamped_to_caps():
    # Per-put cap $25k: 3 × $14.5k = $43.5k → 1 contract.
    cleaned, warnings = _validate(_plan([_put(contracts=3)]))
    assert cleaned.csp_writes[0].contracts == 1
    assert any("cut from 3 to 1" in w for w in warnings)

    # Total cap 80% of $40k = $32k; per-put cap is generous. First put
    # takes 2 contracts ($29k), second can't fit one ($14k left of $3k).
    eligible = {"NVDA": _cand(max_cash=40_000.0), "AMD": _cand("AMD", max_cash=40_000.0)}
    chains = {"NVDA": _chain(), "AMD": _chain("AMD")}
    plan = _plan([_put(contracts=2), _put(ticker="AMD", contracts=1)])
    cleaned, warnings = _validate(plan, eligible=eligible, chains=chains, budget=40_000.0)
    assert [(c.ticker, c.contracts) for c in cleaned.csp_writes] == [("NVDA", 2)]
    assert any("put on AMD dropped: one contract needs" in w for w in warnings)


def test_backfill_from_sizing_then_validates():
    for sizing in (
        f"1 contract $145P {EXP.isoformat()}",
        f"1 contracts at $145 strike, exp {EXP.isoformat()}",
    ):
        plan = _plan([], actions=[RebalanceAction(action="SELL_PUT", ticker="NVDA", sizing=sizing)])
        filled = backfill_csp_writes(plan, chains={"NVDA": _chain()})
        assert [(c.ticker, c.strike, c.delta) for c in filled.csp_writes] == [
            ("NVDA", 145.0, -0.15)
        ]
        cleaned, _ = _validate(filled)
        assert len(cleaned.csp_writes) == 1

    no_row = _plan(
        [], actions=[RebalanceAction(action="SELL_PUT", ticker="NVDA", sizing="1 contract $99P")]
    )
    assert backfill_csp_writes(no_row, chains={"NVDA": _chain()}).csp_writes == []


# --- report ---------------------------------------------------------------


def test_report_data_and_section_render():
    cleaned, _ = _validate(_plan([_put(contracts=1)]))
    data = csp_report_data(cleaned, cash_budget=100_000.0)
    assert data is not None
    assert data["total_cash_reserved"] == 14_500.0
    assert data["pct_of_budget"] == pytest.approx(14.5)
    assert data["rows"][0]["net_cost_if_assigned"] == pytest.approx(143.0)
    assert csp_report_data(_plan([]), cash_budget=1.0) is None

    sections: list = []
    append_csp_section(sections, data, ["put on AMD dropped: x"])
    html = render_html_email(sections, {})
    assert "Cash-secured puts" in html and "$14,500" in html and "put on AMD dropped" in html
    assert render_pdf(sections, {}).startswith(b"%PDF")


# --- data plumbing --------------------------------------------------------


def test_yfinance_puts_are_otm_only():
    e = (date.today() + timedelta(days=35)).isoformat()
    cols = ["strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"]
    puts = pd.DataFrame(
        [(220.0, 2.0, 2.2, 0.3, 100, 10), (240.0, 7.0, 7.2, 0.3, 100, 10)], columns=cols
    )
    calls = pd.DataFrame([(250.0, 3.0, 3.2, 0.3, 100, 10)], columns=cols)
    t = MagicMock()
    t.fast_info = MagicMock(last_price=235.0)
    t.options = (e,)
    t.option_chain.side_effect = lambda _e: MagicMock(calls=calls, puts=puts)
    with patch("stock_analyzer.data.yf_gateway.yf.Ticker", return_value=t):
        chain = YFinanceChain().fetch("PUTX", dte_min=30, dte_max=45, kind="puts")
    assert chain is not None
    assert chain.calls == []
    assert [q.strike for q in chain.puts] == [220.0]


def test_tradier_both_sides(monkeypatch):
    from stock_analyzer.http_client import HttpClient

    monkeypatch.setenv("TRADIER_API_KEY", "fake")
    e = (date.today() + timedelta(days=35)).isoformat()
    rows = [
        {"strike": 260, "bid": 2, "ask": 2.2, "option_type": "call", "greeks": {"delta": 0.3}},
        {"strike": 220, "bid": 1, "ask": 1.1, "option_type": "put", "greeks": {"delta": -0.2}},
        {"strike": 250, "bid": 9, "ask": 9.2, "option_type": "put", "greeks": {"delta": -0.7}},
    ]

    def _json(url, **_kw):
        if "expirations" in url:
            return {"expirations": {"date": [e]}}
        if "chains" in url:
            return {"options": {"option": rows}}
        return {"quotes": {"quote": {"last": 235.0}}}

    with patch.object(HttpClient, "get_json", side_effect=_json):
        both = TradierChain().fetch("X", dte_min=30, dte_max=45, kind="both")
        calls = TradierChain().fetch("X", dte_min=30, dte_max=45)
    assert [q.strike for q in both.calls] == [260.0]
    assert [(q.strike, q.delta) for q in both.puts] == [(220.0, -0.2)]
    assert calls.puts == []


def test_fetch_open_short_puts_sums_collateral(monkeypatch):
    from stock_analyzer.data import brokerage

    monkeypatch.setattr(brokerage, "_credentials", lambda: ("u", "s"))
    client = MagicMock()
    client.account_information.list_user_accounts.return_value = [
        {"id": "a", "name": "IRA"},
        {"id": "b", "name": "Taxable"},
    ]
    positions = {
        "a": [{"symbol": "NVDA  261016P00145000", "units": -2}],
        "b": [
            {"symbol": "NVDA  261016P00140000", "units": -1},
            {"symbol": "AMD   261016P00150000", "units": 1},  # long put — ignored
            {"symbol": "AAPL  261016C00250000", "units": -1},  # short call — ignored
        ],
    }
    client.account_information.get_all_account_positions.side_effect = lambda **kw: positions[
        kw["account_id"]
    ]
    monkeypatch.setattr(brokerage, "_client", lambda: client)
    assert brokerage.fetch_open_short_puts() == {
        "NVDA": {
            "contracts": 3,
            "collateral_usd": 43_000.0,
            "by_account": {"IRA": 29_000.0, "Taxable": 14_000.0},
        }
    }


def test_fetch_recent_picks(tmp_path: Path):
    from stock_analyzer.db.repository import fetch_recent_picks, insert_pick, insert_run
    from stock_analyzer.db.session import get_session

    with get_session(str(tmp_path / "p.db")) as s:
        ids = []
        for n in range(3):
            run_id = insert_run(
                s,
                universe_size=1,
                survivors=1,
                picks=2,
                opus_model="o",
                sonnet_model="s",
                cash_budget=None,
            )
            ids.append(run_id)
            insert_pick(s, run_id, rank=1, ticker=f"A{n}")
            insert_pick(s, run_id, rank=2, ticker=f"B{n}")
        # A run with no picks doesn't use up the lookback.
        insert_run(
            s,
            universe_size=1,
            survivors=0,
            picks=0,
            opus_model="o",
            sonnet_model="s",
            cash_budget=None,
            kind="rebalance",
        )
        s.commit()
        out = fetch_recent_picks(s, n_runs=2)
    assert [(t, r) for t, r, _ in out] == [("A2", 1), ("B2", 2), ("A1", 1), ("B1", 2)]


def test_csp_data_pipeline_offline():
    from stock_analyzer.config import Settings
    from stock_analyzer.discover import rebalance_csp

    state = {
        "cash_balance": 110_000.0,
        "picks": [(1, "NVDA", "one-liner")],
        "holdings_positions": {"AAPL": {"units": 200}},
        "thesis_checks": [{"ticker": "INTC", "status": "BROKEN"}],
        "finnhub_signals": {"NVDA": {"next_earnings_date": "2027-01-01"}},
    }
    recent = [("AAPL", 1, "2026-09-17"), ("INTC", 2, "2026-09-17"), ("AMD", 3, "2026-09-17")]
    chains = {"NVDA": _chain(), "AMD": _chain("AMD", puts=[_q(140.0, None)])}
    with (
        patch(
            "stock_analyzer.data.brokerage.fetch_open_short_puts",
            return_value={"XOM": {"contracts": 1, "collateral_usd": 10_000.0}},
        ),
        patch("stock_analyzer.data.options_chain.fetch_chains", return_value=chains) as fc,
        patch("stock_analyzer.data.earnings_calendar.next_earnings_date", return_value=None),
    ):
        result = rebalance_csp.run_csp_data_pipeline(state, Settings(), recent)
    assert fc.call_args.kwargs["kind"] == "puts"
    assert sorted(result.eligibility) == ["AMD", "NVDA"]
    assert result.cash_budget == 100_000.0  # open put collateral excluded
    assert "TICKER: NVDA" in result.context_block
    assert result.chains["AMD"].puts[0].delta is not None  # filled from IV


def test_rebalance_workflow_has_csp_step():
    from stock_analyzer.cli.rebalance import RebalancePipeline
    from stock_analyzer.config import Settings

    names = [getattr(s, "name", None) for s in RebalancePipeline(Settings()).build_workflow().steps]
    assert names.index("cc_data") < names.index("csp_data") < names.index("rebalance")


def test_split_shares_stay_put_candidates_when_cc_side_known():
    picks = [("SPLT", 1, "2026-09-17"), ("CC", 2, "2026-09-17")]
    positions = {"SPLT": {"units": 110}, "CC": {"units": 200}}
    out = eligible_csp_tickers(
        picks,
        positions=positions,
        cash_budget=100_000.0,
        denylist=(),
        covered_call_tickers={"CC"},
    )
    assert list(out) == ["SPLT"]
    # Without the covered-call set, 100+ shares in total still routes away.
    assert eligible_csp_tickers(picks, positions=positions, cash_budget=1e5, denylist=()) == {}


def test_estimate_action_dollars():
    from stock_analyzer.discover.csp_validation import estimate_action_dollars

    def est(action, sizing, units=100.0, price=50.0):
        a = RebalanceAction(action=action, ticker="X", sizing=sizing)
        return estimate_action_dollars(a, units=units, price=price)

    assert est("BUY", "~$3,400 in Traditional IRA") == 3400
    assert est("ADD", "$2.5k") == 2500
    assert est("BUY", "100 shares (1 lot)") == 5000
    assert est("SELL", "full position") == 5000
    assert est("TRIM", "25%") == 1250
    assert est("TRIM", "full position") == 5000
    assert est("SELL", "50 shares") == 2500
    assert est("SELL", "") == 5000
    assert est("BUY", "starter position") is None


def test_puts_fit_the_cash_left_after_the_plans_buys():
    from stock_analyzer.discover.csp_validation import cash_left_for_puts

    plan = RebalancePlan(
        status="ACTION",
        aggressiveness_applied="balanced",
        actions=[
            RebalanceAction(action="BUY", ticker="A", sizing="~$10,000 in IRA"),
            RebalanceAction(action="ADD", ticker="B", sizing="some more"),
            RebalanceAction(action="SELL", ticker="C", sizing="full position"),
            RebalanceAction(action="SELL_PUT", ticker="NVDA", sizing="x"),
        ],
        full_text="x",
        csp_writes=[_put(contracts=2)],
    )
    budget, room, notes = cash_left_for_puts(
        plan,
        cash_budget=40_000.0,
        account_room={"IRA": 30_000.0, "Taxable": 10_000.0},
        units={"C": 10.0},
        prices={"C": 100.0},
    )
    assert budget == 40_000 - 10_000 + 1_000
    assert room == {"IRA": 20_000.0, "Taxable": 10_000.0}
    assert notes == ["couldn't size ADD B ('some more')"]

    # $20k left in the IRA holds one $14.5k contract; the put moves there.
    cleaned, warnings = validate_csp_writes(
        plan,
        eligible={"NVDA": _cand(max_cash=40_000.0)},
        chains={"NVDA": _chain()},
        cash_budget=budget,
        delta_min=0.10,
        delta_max=0.25,
        dte_min=30,
        dte_max=45,
        max_pct_total=0.80,
        account_room=room,
        today=TODAY,
    )
    (cp,) = cleaned.csp_writes
    assert (cp.account, cp.contracts) == ("IRA", 1)
    assert cleaned.actions[-1].sizing.endswith(" in IRA")
    assert any("cut from 2 to 1" in w for w in warnings)


def test_unnamed_buys_cannot_be_double_spent_per_account():
    """A BUY that names no account still spends cash, so no account may
    claim more put room than the whole plan leaves."""
    from stock_analyzer.discover.csp_validation import cash_left_for_puts
    from stock_analyzer.models.rebalance import RebalanceAction

    plan = _plan([], [RebalanceAction(ticker="NVDA", action="BUY", sizing="$18,000")])
    budget, room, notes = cash_left_for_puts(
        plan,
        cash_budget=20_000.0,
        account_room={"Traditional IRA": 20_000.0},
        units={},
        prices={},
    )
    assert budget == 2_000.0
    assert room["Traditional IRA"] == 2_000.0  # not the untouched 20k
    assert any("named no account" in n for n in notes)


def test_a_buy_that_names_its_account_still_only_charges_that_account():
    from stock_analyzer.discover.csp_validation import cash_left_for_puts
    from stock_analyzer.models.rebalance import RebalanceAction

    plan = _plan(
        [], [RebalanceAction(ticker="NVDA", action="BUY", sizing="$5,000 in Traditional IRA")]
    )
    budget, room, notes = cash_left_for_puts(
        plan,
        cash_budget=20_000.0,
        account_room={"Traditional IRA": 12_000.0, "HSA": 8_000.0},
        units={},
        prices={},
    )
    assert budget == 15_000.0
    assert room == {"Traditional IRA": 7_000.0, "HSA": 8_000.0}
    assert notes == []
