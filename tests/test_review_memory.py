"""The Reviewer's memory of its last verdict per holding."""

from __future__ import annotations

from sqlalchemy import text

from stock_analyzer.db.repository import insert_holdings_review, insert_run
from stock_analyzer.db.session import exec_sql, get_session
from stock_analyzer.discover.rebalance_holdings import build_holding_review_payloads
from stock_analyzer.discover.review_memory import previous_reviews


def _review(db, run_at, ticker, verdict, confidence, text_):
    with get_session(db) as s:
        run_id = insert_run(
            s,
            universe_size=1,
            survivors=1,
            picks=0,
            opus_model="o",
            sonnet_model="s",
            cash_budget=None,
            kind="rebalance",
        )
        exec_sql(s, text("UPDATE runs SET run_at = :t WHERE id = :i"), {"t": run_at, "i": run_id})
        insert_holdings_review(
            s, run_id, ticker, verdict=verdict, confidence=confidence, review_text=text_
        )


def test_the_latest_review_is_remembered_from_its_reasoning(tmp_path):
    db = str(tmp_path / "m.db")
    _review(db, "2026-09-20T10:00:00", "TSLA", "SELL", 8, "Forward outlook: old view")
    _review(
        db,
        "2026-09-27T10:00:00",
        "TSLA",
        "TRIM",
        7,
        "TICKER: TSLA Verdict: TRIM Position context: 200 shares. Forward outlook: "
        + "estimates falling " * 60,
    )
    memory = previous_reviews(db, ["tsla", "NVDA"])
    assert set(memory) == {"TSLA"}
    tsla = memory["TSLA"]
    assert (tsla["date"], tsla["verdict"], tsla["confidence"]) == ("2026-09-27", "TRIM", 7)
    assert tsla["excerpt"].startswith("estimates falling")
    assert tsla["excerpt"].endswith("...") and len(tsla["excerpt"]) <= 404


def test_no_table_means_no_memory(tmp_path):
    assert previous_reviews(str(tmp_path / "missing" / "x.db"), ["TSLA"]) == {}


def test_the_payload_carries_it():
    payloads = build_holding_review_payloads(
        positions={"TSLA": {"avg_buy_price": 400.0, "units": 10, "cost_basis": 4000.0}},
        fund={},
        tech={"TSLA": {"price": 370.0}},
        rfs={},
        insider_selling={},
        finnhub_signals={},
        eps_revisions={},
        position_splits={},
        account_meta={},
        tax_lots_raw={},
        share_trades={},
        holdings_quarterly_mda={},
        holdings_peers={},
        holdings_transcripts={},
        news={},
        risk_factors_chars=10,
        quarterly_mda_chars=10,
        transcript_chars=10,
        previous={"TSLA": {"verdict": "TRIM"}},
    )
    assert payloads["TSLA"]["previous_review"] == {"verdict": "TRIM"}
