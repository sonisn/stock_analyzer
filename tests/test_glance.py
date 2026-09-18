"""'At a glance' blocks that open the discover and rebalance reports."""

from __future__ import annotations

from stock_analyzer.discover.rebalance_sections import append_rebalance_glance
from stock_analyzer.discover.report_sections import build_sections
from stock_analyzer.models.llm import Allocation, RankerOutput, RankerPick, SizerOutput
from stock_analyzer.models.rebalance import RebalanceAction, RebalancePlan


def _pick(rank: int, ticker: str, agreement: float | None, voters: list[str] | None) -> RankerPick:
    return RankerPick.model_construct(
        rank=rank,
        ticker=ticker,
        one_liner=f"{ticker} thesis in one line.",
        why_over_alternatives="x",
        conviction=8,
        time_horizon="6-12 months",
        sector_concentration_check="x",
        bull_thesis="x",
        what_youre_betting_on="x",
        scenarios=[],
        agreement_ratio=agreement,
        voting_providers=voters,
    )


def test_discover_report_opens_with_the_glance_and_ends_with_history():
    ranker = RankerOutput.model_construct(
        picks=[_pick(1, "ANET", 1.0, ["claude", "gemini", "openai"]), _pick(2, "AMP", None, None)],
        pairs_not_to_hold_together=[],
        full_text="",
    )
    sizer = SizerOutput.model_construct(
        allocations=[
            Allocation.model_construct(
                ticker="ANET", allocation_pct=30.0, allocation_usd=None, rationale="r"
            ),
            Allocation.model_construct(
                ticker="AMP", allocation_pct=None, allocation_usd=4000.0, rationale="r"
            ),
        ],
        concentration_warnings=["SECTOR CAP: Technology picks (ANET) totaled 60% — scaled to 50%"],
        full_text="",
    )
    thesis = [
        {
            "ticker": "MSFT",
            "status": "BROKEN",
            "return_pct": -16.4,
            "signals": [],
            "pick_date": "2026-06-01",
            "excess_pct": -19.0,
            "bear_target_pct": -15.0,
            "bull_target_pct": 30.0,
        },
    ]
    sections = build_sections(
        ranker_text="",
        redteam_text="",
        sizer_text="",
        candidates=[],
        universe_size=10,
        holdings_summary="",
        ranker_output=ranker,
        sizer_output=sizer,
        thesis_checks=thesis,
        paper_ledger=None,
        usage={"rows": [], "budget": {"cap_usd": 2, "spent_usd": 1, "notes": ["skipped a round"]}},
    )
    kinds = [(x.kind, x.text) for x in sections]
    glance_at = kinds.index(("heading", "At a glance"))
    assert glance_at == 2  # right after the title and the funnel line
    first_card = next(i for i, x in enumerate(sections) if x.kind == "pick_card")
    assert kinds.index(("heading", "Open picks: thesis check")) > first_card
    glance = sections[glance_at + 1 : glance_at + 8]
    table = next(x for x in glance if x.kind == "table")
    assert table.table_rows[0][:4] == ["ANET", "30%", "8/10", "3/3"]
    assert table.table_rows[1][:4] == ["AMP", "$4,000", "8/10", "—"]
    para = " ".join(x.text for x in glance if x.kind == "para")
    assert (
        "MSFT: thesis broken" in para and "SECTOR CAP: Technology picks (ANET) totaled 60%" in para
    )
    assert "Cost cap: skipped a round." in para


def test_rebalance_glance_lists_actions_then_other_decisions():
    plan = RebalancePlan.model_construct(
        status="ACTION",
        aggressiveness_applied="balanced",
        actions=[RebalanceAction.model_construct(action="TRIM", ticker="MSFT", sizing="25%")],
        summary="s",
        full_text="t",
    )
    sections = []
    append_rebalance_glance(
        sections,
        rebalance_plan=plan,
        thesis_checks=[{"ticker": "APH", "status": "TARGET HIT", "return_pct": 31.2}],
        harvest_candidates=[{"loss_usd": -1000.0, "est_tax_saving_usd": 320.0}],
        stop_loss_warnings=["STOP-LOSS: PYPL -22% from cost — HOLD escalated to TRIM 25%"],
    )
    assert sections[1].table_rows == [["TRIM", "MSFT", "25%"]]
    text = " ".join(x.text for x in sections[2:])
    assert "APH: past its bull target" in text and "STOP-LOSS: PYPL" in text
    assert "$1,000 of losses (~$320 tax)" in text

    empty = []
    append_rebalance_glance(
        empty,
        rebalance_plan=None,
        thesis_checks=None,
        harvest_candidates=None,
        stop_loss_warnings=None,
    )
    assert empty == []
