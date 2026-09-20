"""Every sale comes with a destination for the money (discover/reinvest.py,
the daily email's decisions and the rebalance report's fallback)."""

from __future__ import annotations

from pathlib import Path

from stock_analyzer.db.repository import insert_candidate, insert_pick, insert_run
from stock_analyzer.db.session import get_session
from stock_analyzer.discover.rebalance_sections import append_reinvest_section
from stock_analyzer.discover.reinvest import (
    format_idea,
    load_pick_pool,
    reinvest_ideas,
    sector_peers,
    unfunded_sales,
)
from stock_analyzer.discover.report_html import render_html_email
from stock_analyzer.models.rebalance import RebalanceAction, RebalancePlan
from stock_analyzer.reporting.health import (
    build_portfolio_health,
    decision_items,
    render_health_html,
)

IDEA_A = {"ticker": "ANET", "rank": 1, "pick_date": "2026-09-17", "sector": "Technology"}
IDEA_B = {"ticker": "AMP", "rank": 3, "pick_date": "2026-09-17", "sector": "Financials"}


def _seed(db: str) -> None:
    with get_session(db) as s:
        for picks, cands in (
            ([("OLD", 1)], [("OLD", "Energy", 50.0, True)]),
            (
                [("ANET", 1), ("KO", 2)],
                [
                    ("ANET", "Technology", 80.0, True),
                    ("KO", "Staples", 60.0, True),
                    ("AMD", "Technology", 70.0, True),
                    ("DELL", "Technology", 90.0, True),
                    ("FAIL", "Technology", 99.0, False),
                    ("MSFT", "Technology", 85.0, True),
                ],
            ),
        ):
            run_id = insert_run(
                s,
                universe_size=1,
                survivors=1,
                picks=len(picks),
                opus_model="o",
                sonnet_model="s",
                cash_budget=None,
            )
            for ticker, sector, score, passed in cands:
                insert_candidate(
                    s,
                    run_id,
                    ticker,
                    passed_filter=passed,
                    fail_reasons=[],
                    score=score,
                    score_components={},
                    score_breakdown={},
                    sources=[],
                    conviction=7,
                    sector=sector,
                    price=100.0,
                )
            for ticker, rank in picks:
                insert_pick(s, run_id, rank=rank, ticker=ticker, conviction=8)
        s.commit()


def test_pool_ideas_and_peers(tmp_path: Path):
    db = str(tmp_path / "r.db")
    _seed(db)
    pool = load_pick_pool(db, n_runs=1)
    assert [(i["ticker"], i["rank"], i["sector"]) for i in pool] == [
        ("ANET", 1, "Technology"),
        ("KO", 2, "Staples"),
    ]
    assert [i["ticker"] for i in load_pick_pool(db)] == ["ANET", "KO", "OLD"]
    assert [i["ticker"] for i in reinvest_ideas(pool, held={"KO"})] == ["ANET"]
    assert reinvest_ideas(pool, held=set(), avoid_sectors={"Technology", "Staples"}) == []
    assert [i["ticker"] for i in reinvest_ideas(pool, held=set(), exclude={"ANET"})] == ["KO"]
    # Best score first; failed-filter and held names skipped; not itself.
    assert sector_peers(db, ["ANET", "XOM"], held={"MSFT"}) == {"ANET": {"peers": ["DELL", "AMD"]}}


def test_unfunded_sales_and_report_section():
    def plan(*actions):
        return RebalancePlan(
            status="ACTION",
            aggressiveness_applied="balanced",
            actions=[RebalanceAction(action=a, ticker=t, sizing="x") for a, t in actions],
            full_text="x",
        )

    assert unfunded_sales(plan(("SELL", "MRVL"), ("TRIM", "X"))) == ["MRVL", "X"]
    assert unfunded_sales(plan(("SELL", "MRVL"), ("ADD", "GOOGL"))) == []
    assert unfunded_sales(plan(("SELL_PUT", "A"), ("SELL", "B"))) == []

    sections: list = []
    append_reinvest_section(sections, {"sold": ["MRVL"], "ideas": [IDEA_A]})
    html = render_html_email(sections, {})
    assert "Where the sale proceeds could go" in html and "ANET" in html
    empty: list = []
    append_reinvest_section(empty, {"sold": ["MRVL"], "ideas": []})
    assert empty == []


def test_daily_sell_lines_name_a_destination():
    holdings = {
        "Brokerage": [
            {"ticker": "BAD", "units": 10, "average_purchase_price": 100.0, "price": 90.0},
            {"ticker": "DOWN", "units": 10, "average_purchase_price": 100.0, "price": 70.0},
            {"ticker": "LOSS", "units": 100, "average_purchase_price": 50.0, "price": 40.0},
        ]
    }
    asked = {}

    def reinvest(held, over_cap, n):
        asked.update(held=held, n=n)
        return [IDEA_A, IDEA_B]

    h = build_portfolio_health(
        holdings,
        held_thesis_checks=lambda held: [
            {
                "ticker": "BAD",
                "status": "BROKEN",
                "return_pct": -10.0,
                "excess_pct": -20.0,
                "signals": [{"severity": "broken", "text": "estimates cut"}],
            }
        ],
        harvest=lambda: [
            {
                "ticker": "LOSS",
                "account": "Brokerage",
                "units": 100,
                "price": 40.0,
                "loss_usd": -1000.0,
                "loss_pct": -20.0,
                "est_tax_saving_usd": 300.0,
                "wash_sale_until": None,
                "rebuy_ok_after": "2026-10-19",
                "swap_candidates": ["PEER"],
            }
        ],
        reinvest=reinvest,
    )
    assert asked == {"held": {"BAD", "DOWN", "LOSS"}, "n": 2}  # the swap covers LOSS
    # One ticker can raise more than one decision (a tax-loss sale and a
    # covered-call note, say), so keep the sale-shaped line for each.
    sale_labels = {"BROKEN", "DRAWDOWN", "TAX LOSS", "TARGET HIT"}
    texts = {i["ticker"]: i["text"] for i in decision_items(h) if i["label"] in sale_labels}
    assert texts["BAD"].endswith(f"Reinvest the ~$900 in {format_idea(IDEA_A)}.")
    assert f"If you do sell, reinvest the ~$700 in {format_idea(IDEA_B)}." in texts["DOWN"]
    assert "buy PEER (same sector, not the same stock) with the ~$4,000" in texts["LOSS"]
    assert "Where sale proceeds could go" in render_health_html(h)


def test_no_sales_means_no_reinvest_lookup():
    calls = []
    h = build_portfolio_health(
        {"A": [{"ticker": "OK", "units": 1, "average_purchase_price": 1.0, "price": 2.0}]},
        reinvest=lambda *a: calls.append(a) or [],
    )
    assert calls == [] and h.reinvest == []
