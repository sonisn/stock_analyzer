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


# --- why this stock ----------------------------------------------------------------


RANKER_TEXT = """\
PICK 1: ANET — Arista offers best-in-class 45.4% operating margins and a raised
$3.25B AI-networking revenue target.

Bull thesis:
Something long here.
---

PICK 2: A — Agilent provides defensive life-sciences exposure with accelerating
estimates and a newly expanded diagnostics footprint via the Biocare acquisition.

Why this over alternatives:
I chose this over DXCM.
"""


def test_the_rankers_own_sentence_is_read_back():
    from stock_analyzer.discover.reinvest import pick_headline

    reason = pick_headline(RANKER_TEXT, "A")
    assert reason.startswith("Agilent provides defensive life-sciences exposure")
    # not the neighbouring pick
    assert "Arista" not in reason
    assert pick_headline(RANKER_TEXT, "ANET").startswith("Arista offers best-in-class")


def test_a_ticker_with_no_stored_reason_is_silent_not_invented():
    from stock_analyzer.discover.reinvest import pick_headline

    assert pick_headline(RANKER_TEXT, "NVDA") == ""
    assert pick_headline(None, "A") == ""
    assert pick_headline("", "A") == ""


def test_the_idea_line_carries_the_reason_and_the_sector_standing():
    from stock_analyzer.discover.reinvest import format_idea

    idea = {
        "ticker": "A",
        "rank": 2,
        "pick_date": "2026-09-17",
        "sector": "Healthcare",
        "reason": "Agilent provides defensive life-sciences exposure.",
        "sector_bias": "leader",
    }
    line = format_idea(idea)
    assert line.startswith("A (pick #2, 2026-09-17, Healthcare, sector leading)")
    assert line.endswith("— Agilent provides defensive life-sciences exposure.")


def test_the_old_bare_label_still_works_without_either():
    from stock_analyzer.discover.reinvest import format_idea

    idea = {"ticker": "ANET", "rank": 1, "pick_date": "2026-09-17", "sector": "Technology"}
    assert format_idea(idea) == "ANET (pick #1, 2026-09-17, Technology)"


def test_sector_bias_is_tagged_from_the_rotation_summary():
    from stock_analyzer.discover.reinvest import with_sector_bias

    summary = {
        "leaders": ["Technology", "Healthcare", "Financial Services"],
        "laggards": ["Utilities", "Communication Services", "Consumer Cyclical"],
    }
    ideas = [
        {"ticker": "A", "sector": "Healthcare"},
        {"ticker": "NEE", "sector": "Utilities"},
        {"ticker": "XOM", "sector": "Energy"},
    ]
    tagged = {i["ticker"]: i["sector_bias"] for i in with_sector_bias(ideas, summary)}
    assert tagged == {"A": "leader", "NEE": "laggard", "XOM": "neutral"}


def test_without_a_rotation_summary_nothing_is_claimed():
    from stock_analyzer.discover.reinvest import with_sector_bias

    ideas = [{"ticker": "A", "sector": "Healthcare"}]
    assert "sector_bias" not in with_sector_bias(ideas, None)[0]


# --- a suggested stock gets the same look as a held one ---------------------------


def test_suggested_tickers_are_collected_in_suggestion_order():
    from stock_analyzer.reporting.health import build_portfolio_health, suggested_tickers

    health = build_portfolio_health(
        {
            "Brokerage": [
                {"ticker": "HELD", "units": 10, "average_purchase_price": 100.0, "price": 70.0}
            ]
        },
        reinvest=lambda held, over_cap, n: [
            {"ticker": "A", "rank": 2, "pick_date": "2026-09-17", "sector": "Healthcare"},
            {"ticker": "AMP", "rank": 3, "pick_date": "2026-09-17", "sector": "Financial"},
        ],
    )
    out = suggested_tickers(health)
    assert out[0] == "A"  # the drawdown line's destination comes first
    assert "AMP" in out
    assert "HELD" not in out  # never suggest what is already owned


def test_the_idea_block_shows_a_chart_and_the_numbers():
    from stock_analyzer.reporting.health import build_portfolio_health, render_idea_details_html

    health = build_portfolio_health({})
    health.idea_details["A"] = {
        "name": "Agilent Technologies",
        "price": "$153.69",
        "pct_today": "+0.8%",
        "range_52w": "$96.00 - $160.00",
        "pe": "29.7",
        "analyst_target": "$175.00",
        "trend_1mo": "up",
        "trend_6mo": "up",
        "reason": "Agilent provides defensive life-sciences exposure.",
        "chart_cid": "chart-A",
    }
    html = render_idea_details_html(health)
    assert "Ideas for new money" in html
    assert "A — Agilent Technologies" in html
    assert "Agilent provides defensive life-sciences exposure." in html
    assert "Price $153.69" in html and "Analyst target $175.00" in html
    assert "Trend: 1mo up" in html
    assert 'src="cid:chart-A"' in html


def test_an_idea_without_a_chart_still_renders():
    from stock_analyzer.reporting.health import build_portfolio_health, render_idea_details_html

    health = build_portfolio_health({})
    health.idea_details["AMP"] = {"name": "Ameriprise", "price": "$540.86"}
    html = render_idea_details_html(health)
    assert "AMP — Ameriprise" in html and "cid:" not in html


def test_no_ideas_render_nothing():
    from stock_analyzer.reporting.health import build_portfolio_health, render_idea_details_html

    assert render_idea_details_html(build_portfolio_health({})) == ""


def test_new_ideas_can_be_switched_off_without_losing_holding_actions():
    # "Drop the new suggestions, keep the action on current holdings."
    from stock_analyzer.reporting.health import (
        build_portfolio_health,
        decision_items,
        render_idea_details_html,
    )

    holdings = {
        "Brokerage": [
            {"ticker": "DOWN", "units": 10, "average_purchase_price": 100.0, "price": 70.0}
        ]
    }
    # reinvest=None is what the CLI passes when the switch is off.
    off = build_portfolio_health(holdings, reinvest=None)
    drawdown = next(i for i in decision_items(off) if i["label"] == "DRAWDOWN")
    assert "DOWN" in drawdown["text"]  # the holding action survives
    assert "einvest" not in drawdown["text"]  # nothing new is proposed
    assert drawdown["reinvest_into"] is None
    assert render_idea_details_html(off) == ""

    on = build_portfolio_health(
        holdings,
        reinvest=lambda held, over_cap, n: [
            {"ticker": "A", "rank": 2, "pick_date": "2026-09-17", "sector": "Healthcare"}
        ],
    )
    on_item = next(i for i in decision_items(on) if i["label"] == "DRAWDOWN")
    assert "reinvest" in on_item["text"] and on_item["reinvest_into"] == "A"
