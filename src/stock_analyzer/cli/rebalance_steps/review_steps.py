"""Rebalance steps that review each holding and prepare the options and tax
inputs the plan needs."""

from __future__ import annotations

from typing import Any

from agno.workflow.types import StepInput, StepOutput

from ...db.repository import fetch_recent_picks
from ...db.session import get_session
from ...discover.catalysts import repair_catalysts
from ...discover.rebalance_cc import (
    cc_empty_state,
    run_cc_data_pipeline,
)
from ...discover.rebalance_csp import (
    csp_empty_state,
    run_csp_data_pipeline,
)
from ...discover.rebalance_holdings import (
    build_holding_review_payloads,
    flag_drawdown_reviews,
)
from ...discover.reviewer import REVIEWER_INSTRUCTIONS, Reviewer, review_batch
from ...discover.tax_harvest import (
    find_harvest_candidates,
    format_harvest_block,
)
from ...logging import get_logger
from ...serialization import dumps_pretty
from ...usage import BUDGET, estimate_cost
from ..discover_steps.helpers import (
    _QUARTERLY_MDA_CHARS,
    _RISK_FACTORS_CHARS,
    _TRANSCRIPT_CHARS,
)

logger = get_logger("stock_analyzer.cli.rebalance")


class RebalanceReviewSteps:
    def step_review_holdings(self, step_input: StepInput) -> StepOutput:
        # `holdings_positions` deliberately carries everything, including
        # symbols no market data exists for — they are still valued and
        # taxed. Reviewing them is a different matter: a revoked CUSIP has
        # no price, no fundamentals and no news, so the reviewer spends an
        # LLM call to write "no data available" and the report prints it
        # beside real holdings. step_holdings_fetch already worked out
        # which tickers are analyzable; this is the list to review.
        analyzable = set(self.state.get("holdings_tickers") or self.state["holdings_positions"])
        payloads = build_holding_review_payloads(
            positions={
                t: p for t, p in self.state["holdings_positions"].items() if t in analyzable
            },
            fund=self.state["holdings_fundamentals"],
            tech=self.state["holdings_technicals"],
            rfs=self.state["holdings_risk_factors"],
            insider_selling=self.state.get("insider_selling", {}),
            finnhub_signals=self.state.get("finnhub_signals", {}),
            eps_revisions=self.state.get("eps_revisions", {}),
            position_splits=self.state.get("position_splits", {}),
            account_meta=self.state.get("account_meta", {}),
            tax_lots_raw=self.state.get("tax_lots", {}),
            share_trades=self.state.get("share_trades", {}),
            holdings_quarterly_mda=self.state.get("holdings_quarterly_mda", {}),
            holdings_peers=self.state.get("holdings_peers", {}),
            holdings_transcripts=self.state.get("holdings_transcripts", {}),
            news=self.state.get("news") or {},
            recent_news=self.state.get("recent_news") or {},
            thesis_checks=self.state.get("thesis_checks") or [],
            risk_factors_chars=_RISK_FACTORS_CHARS,
            quarterly_mda_chars=_QUARTERLY_MDA_CHARS,
            transcript_chars=_TRANSCRIPT_CHARS,
        )
        reviewer = Reviewer(
            "claude",
            self._reviewer_model(payloads),
            fallback=(
                self.settings.discover_fallback_provider,
                self.settings.resolve_fallback_model(),
            ),
        )
        reviews, _ = repair_catalysts(
            review_batch(reviewer, payloads), self.state.get("recent_news") or {}
        )
        drawdown_notes = flag_drawdown_reviews(
            reviews,
            self.state["holdings_positions"],
            self.state["holdings_technicals"],
        )
        if drawdown_notes:
            self.state["stop_loss_warnings"] = drawdown_notes
        self.state["holdings_reviews"] = reviews
        return StepOutput(content=f"Reviewed {len(self.state['holdings_reviews'])} holdings")

    def _reviewer_model(self, payloads: dict[str, dict[str, Any]]) -> str:
        """Sonnet, unless reviewing every holding on it would not fit the cost
        cap's remaining room (after the final-stage reserve) — then Haiku."""
        sonnet = self.settings.discover_sonnet_model
        haiku = self.settings.discover_haiku_model
        available = BUDGET.available_for()
        if available is None or not haiku:
            return sonnet
        est = sum(
            estimate_cost(sonnet, len(dumps_pretty(p)) + len(REVIEWER_INSTRUCTIONS), 2000) or 0.0
            for p in payloads.values()
        )
        if est <= available:
            return sonnet
        BUDGET.note(
            f"reviewed {len(payloads)} holdings on {haiku} instead of {sonnet} "
            f"(estimated ${est:.2f} vs ${available:.2f} available)"
        )
        return haiku

    def step_cc_data(self, step_input: StepInput) -> StepOutput:
        """Build the COVERED-CALL CONTEXT block consumed by the rebalancer."""
        if not self.settings.cc_enabled:
            self.state.update(cc_empty_state())
            return StepOutput(content="cc_data: disabled via CC_ENABLED=0")

        self.state.update(cc_empty_state())
        try:
            result = run_cc_data_pipeline(self.state, self.settings)
            self.state["cc_context_block"] = result.context_block
            self.state["cc_eligibility"] = result.eligibility
            self.state["cc_round_lot_coverage"] = result.coverage
            self.state["cc_stub_pool_total_usd"] = result.stub_pool
            self.state["cc_chains"] = result.chains
            self.state["cc_iv_hv_regimes"] = result.iv_hv_regimes
            self.state["stub_income_block"] = result.stub_income_block
            # Holdings where the premium is too cheap to be worth the cap:
            # not silence, a reason.
            self.state["cc_cheap_premium"] = result.cheap_premium
            return StepOutput(content=result.content)
        except Exception as e:
            logger.error(
                "step_cc_data crashed (%s) — rebalance will run WITHOUT "
                "CC context. Investigate the traceback below.",
                e,
                exc_info=True,
            )
            return StepOutput(
                content=f"cc_data: failed ({type(e).__name__}); CC disabled for this run"
            )

    def step_csp_data(self, step_input: StepInput) -> StepOutput:
        """Build the CASH-SECURED PUT CONTEXT block consumed by the rebalancer."""
        self.state.update(csp_empty_state())
        if not self.settings.csp_enabled:
            return StepOutput(content="csp_data: disabled via CSP_ENABLED=0")
        try:
            with get_session(self.settings.discover_db_path) as session:
                recent = fetch_recent_picks(session, n_runs=self.settings.csp_pick_lookback_runs)
            result = run_csp_data_pipeline(self.state, self.settings, recent)
        except Exception as e:
            logger.error(
                "step_csp_data crashed (%s) — rebalance will run WITHOUT put ideas.",
                e,
                exc_info=True,
            )
            return StepOutput(
                content=f"csp_data: failed ({type(e).__name__}); puts disabled for this run"
            )
        self.state["csp_context_block"] = result.context_block
        self.state["csp_eligibility"] = result.eligibility
        self.state["csp_chains"] = result.chains
        self.state["csp_cash_budget"] = result.cash_budget
        self.state["csp_account_room"] = result.account_room
        # A put that isn't offered still owes the reader a reason.
        self.state["csp_blocked_note"] = result.blocked_note
        self.state["csp_cheap_premium"] = result.cheap_premium
        return StepOutput(content=result.content)

    def step_tax_harvest(self, step_input: StepInput) -> StepOutput:
        """Deterministic tax-loss harvesting candidates (no LLM), fed to the
        Rebalancer and shown in the report."""
        try:
            prices = {
                t: (v or {}).get("price")
                for t, v in (self.state.get("holdings_technicals") or {}).items()
            }
            candidates = find_harvest_candidates(
                self.state.get("position_splits") or {},
                prices,
                self.state.get("tax_lots") or {},
                self.state.get("holdings_peers") or {},
                min_loss_usd=self.settings.harvest_min_loss_usd,
                min_loss_pct=self.settings.harvest_min_loss_pct,
                # Shares backing a short call cannot be sold without
                # buying it back, so they are not harvestable.
                covered_calls=self.state.get("covered_call_obligations") or {},
            )
        except Exception as e:
            logger.warning("tax-loss harvest scan failed (%s) — skipping", e)
            candidates = []
        self.state["harvest_candidates_obj"] = candidates
        self.state["harvest_block"] = format_harvest_block(candidates)
        total = sum(-c.loss_usd for c in candidates)
        return StepOutput(
            content=f"Tax-loss harvest: {len(candidates)} candidates, ${total:,.0f} of losses"
        )
