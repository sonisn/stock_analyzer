"""A plan that fails to parse must never read as a decision to hold.

On 2026-09-20 the rebalancer hit its 16,000-token output ceiling, the JSON
came back truncated mid-string, and the run emailed a cheerful report with
no action list. The premortem logged "skipped (NO_ACTION plan)" because it
could not tell an absent plan from one that recommended nothing.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.cli.rebalance import build_email_subject
from stock_analyzer.config import Settings
from stock_analyzer.discover.rebalance_csp import _blocked_note
from stock_analyzer.discover.rebalance_sections import build_rebalance_sections
from stock_analyzer.discover.rebalancer import RebalancePlanUnparseable, _looks_truncated

TRUNCATED = '{"status":"ACTION","aggressiveness_applied":"balanced","summary":"sell MRVL'


def _sections(**kw):
    base = dict(
        rebalance_text="",
        holdings_reviews={},
        ranker_text="",
        redteam_text="",
        sizer_text="",
        candidates=[],
        cash_balance=20878.0,
        macro_summary="",
        sector_rotation=None,
        holdings_positions={},
        holdings_technicals={},
        holdings_fundamentals={},
    )
    return build_rebalance_sections(**(base | kw))


def _banner(sections) -> str:
    return next(s.text for s in sections if s.kind == "status_banner")


def test_the_error_carries_the_text_the_run_paid_for():
    e = RebalancePlanUnparseable("bad json", raw_text=TRUNCATED, truncated=True)
    assert e.raw_text == TRUNCATED
    assert e.truncated is True
    # Default is the conservative one: don't claim truncation you didn't see.
    assert RebalancePlanUnparseable("bad json").truncated is False


def test_truncation_is_read_off_the_provider_not_guessed():
    class _M:
        stop_reason = "max_tokens"

    class _Run:
        metrics = _M()

    class _Ended:
        class metrics:  # noqa: N801
            stop_reason = "end_turn"

    assert _looks_truncated(_Run()) is True
    assert _looks_truncated(_Ended()) is False
    assert _looks_truncated(object()) is False


def test_a_lost_plan_is_not_a_hold_recommendation():
    sections = _sections(plan_failure="The rebalancer's plan was cut off before it finished")
    banner = _banner(sections)
    assert "PLAN INCOMPLETE" in banner
    assert "NO ACTION RECOMMENDED" not in banner
    text = " ".join(s.text or "" for s in sections)
    assert "cut off before it finished" in text
    assert "not because the rebalancer decided to hold" in text


def test_surviving_plan_text_is_shown_but_not_as_instructions():
    sections = _sections(
        plan_failure="The rebalancer's plan was cut off before it finished",
        rebalance_text=TRUNCATED,
    )
    text = " ".join(s.text or "" for s in sections)
    assert TRUNCATED in text
    assert "read it as notes, not as instructions" in text


def test_a_genuine_no_action_plan_still_reads_normally():
    banner = _banner(_sections(rebalance_text="STATUS: NO_ACTION"))
    assert "PLAN INCOMPLETE" not in banner


def test_the_subject_line_says_a_re_run_is_needed():
    assert "PLAN INCOMPLETE" in build_email_subject(
        action_count=0, gross_premium_usd=0.0, plan_failed=True
    )
    normal = build_email_subject(action_count=3, gross_premium_usd=0.0)
    assert "PLAN INCOMPLETE" not in normal
    assert date.today().strftime("%b-%d") in normal


class _Chain:
    def __init__(self, strikes):
        self.puts = [type("Q", (), {"strike": k})() for k in strikes]


def test_the_per_put_cap_explains_itself():
    """The live case: $20,847 at a 25% cap reaches a $52 strike, and the
    cheapest put on offer needs $11,601."""
    s = Settings()
    note = _blocked_note({"COP": object()}, {"COP": _Chain([116.0])}, 20847.0, s)
    assert "$5,211" in note or "$5,212" in note
    assert "$52" in note
    assert "COP at $11,600" in note
    assert "CSP_MAX_PCT_PER_PUT" in note


def test_a_missing_chain_is_reported_as_such():
    note = _blocked_note({"COP": object()}, {}, 20847.0, Settings())
    assert "No put chain came back" in note


def test_a_fitting_strike_blames_the_delta_band_instead():
    note = _blocked_note({"X": object()}, {"X": _Chain([20.0])}, 20847.0, Settings())
    assert "delta band" in note
    assert "CSP_MAX_PCT_PER_PUT" not in note


def test_the_put_side_now_has_the_same_vol_floor_as_the_call_side():
    s = Settings()
    assert s.csp_min_iv_hv_ratio == pytest.approx(s.cc_min_iv_hv_ratio)


def _ranker_and_sizer():
    from stock_analyzer.models.llm import Allocation, RankerOutput, RankerPick, SizerOutput

    pick = RankerPick(
        rank=1,
        ticker="MSFT",
        one_liner="A $684B contracted book growing 82% YoY at a 20.9x forward multiple.",
        why_over_alternatives="",
        conviction=8,
        time_horizon="3-5 years",
        sector_concentration_check="",
        bull_thesis="",
        what_youre_betting_on="",
        scenarios=[],
        agreement_ratio=1.0,
        voting_providers=["claude", "gemini", "openai"],
    )
    ranker = RankerOutput(picks=[pick], pairs_not_to_hold_together=[], full_text="PICK 1: MSFT")
    sizer = SizerOutput(
        allocations=[Allocation(ticker="MSFT", allocation_pct=26.0, rationale="")],
        concentration_warnings=[],
        full_text="",
    )
    return ranker, sizer


def test_the_rebalance_appendix_keeps_its_at_a_glance_numbers():
    """The appendix used to pass only the prose, so `build_sections` fell
    back to parsing it and every column rendered an em dash."""
    ranker, sizer = _ranker_and_sizer()
    sections = _sections(
        ranker_text=ranker.full_text,
        ranker_output=ranker,
        sizer_output=sizer,
    )
    rows = [r for s in sections if s.kind == "table" for r in (s.table_rows or [])]
    glance = next(r for r in rows if r and r[0] == "MSFT")
    assert glance[1] == "26%"
    assert glance[2] == "8/10"
    assert glance[3] == "3/3"
    assert "contracted book" in glance[4]
    assert "—" not in glance[1:4]


def test_the_output_ceiling_stays_inside_what_the_sdk_will_send():
    """32,000 was tried on 2026-09-20 and the SDK refused the request
    outright — it rejects a non-streaming call whose max_tokens implies
    more than ten minutes. Raising this again means streaming first."""
    from stock_analyzer.discover import rebalancer as r

    configured = r.REBALANCER_MAX_OUTPUT_TOKENS
    assert configured <= r.MAX_NONSTREAMING_OUTPUT_TOKENS, (
        f"max_tokens={configured} exceeds the SDK's non-streaming ceiling "
        f"({r.MAX_NONSTREAMING_OUTPUT_TOKENS}); the request will be refused before it is sent"
    )
    # And it must still be above the value that demonstrably truncated.
    assert configured > 16000
