"""Rebalance steps that write the plan: the Opus rebalance, its pre-mortem, and
the suggestions ledger."""

from __future__ import annotations

from datetime import date
from typing import Any

from agno.workflow.types import StepInput, StepOutput

from ...db.session import get_session
from ...discover.premortem import PreMortemAgent
from ...discover.rebalance_cc import (
    apply_cc_plan_validation,
    log_rebalancer_input_estimate,
)
from ...discover.rebalance_csp import (
    apply_csp_plan_validation,
)
from ...discover.rebalancer import (
    RebalancePlanUnparseable,
    Rebalancer,
    format_accounts_block,
)
from ...discover.sale_validation import covered_call_block, validate_sales
from ...discover.tax_harvest import (
    flag_plan_conflicts,
    harvest_report_data,
)
from ...logging import get_logger
from ...usage import BudgetExceededError
from ..pipeline_base import PipelineBase
from .helpers import (
    _build_history_block,
)

logger = get_logger("stock_analyzer.cli.rebalance")


class RebalancePlanSteps(PipelineBase):
    def step_rebalance(self, step_input: StepInput) -> StepOutput:
        history_block = _build_history_block(self.settings.discover_db_path)
        if history_block:
            logger.info(
                "Cross-run context: %d holdings have prior decisions",
                history_block.count("\n"),
            )
        ranker_text = self.state.get("ranker_text") or ""
        if not ranker_text:
            logger.warning(
                "Rebalance: no ranker_text in state (Ranker step likely "
                "failed). Producing holdings-only plan from reviews."
            )
        rebalancer = self._build_rebalancer()
        log_rebalancer_input_estimate(
            self.state,
            ranker_text=ranker_text,
            history_block=history_block,
        )
        try:
            plan = self._request_plan(rebalancer, ranker_text, history_block)
        except Exception as e:  # noqa: BLE001 — see _record_lost_plan
            return self._record_lost_plan(e)
        plan = self._validate_covered_calls(plan)
        plan = self._validate_puts(plan)
        # Last, and deterministic: the prompt block above asks the model to
        # plan around promised shares; this checks that it did. A sale of
        # shares backing a short call cannot be executed at all.
        plan, sale_warnings = validate_sales(
            plan,
            positions=self.state.get("holdings_positions") or {},
            obligations=self.state.get("covered_call_obligations") or {},
        )
        if sale_warnings:
            self.state["sale_warnings"] = sale_warnings
        self.state["rebalance_plan"] = plan
        self.state["rebalance_text"] = plan.full_text
        self.state["harvest_candidates"] = harvest_report_data(
            flag_plan_conflicts(self.state.get("harvest_candidates_obj") or [], plan)
        )
        return StepOutput(
            content=(
                f"Rebalance plan generated "
                f"(status={plan.status}, "
                f"aggressiveness={plan.aggressiveness_applied}, "
                f"actions={len(plan.actions)})"
            )
        )

    def _build_rebalancer(self) -> Rebalancer:
        return Rebalancer(
            "claude",
            self.settings.discover_opus_model,
            cc_target_delta_min=self.settings.cc_target_delta_min,
            cc_target_delta_max=self.settings.cc_target_delta_max,
            cc_dte_min=self.settings.cc_dte_min,
            cc_dte_max=self.settings.cc_dte_max,
            cc_min_premium_usd=self.settings.cc_min_premium_usd,
            cc_slippage_buffer=self.settings.cc_slippage_buffer,
            cc_min_stub_usd=self.settings.cc_min_stub_usd,
            cc_stub_optimization=self.settings.cc_stub_optimization,
            csp_target_delta_min=self.settings.csp_target_delta_min,
            csp_target_delta_max=self.settings.csp_target_delta_max,
            csp_dte_min=self.settings.csp_dte_min,
            csp_dte_max=self.settings.csp_dte_max,
            csp_max_pct_per_put=self.settings.csp_max_pct_per_put,
            csp_max_pct_total=self.settings.csp_max_pct_total,
        )

    def _request_plan(self, rebalancer: Rebalancer, ranker_text: str, history_block: str) -> Any:
        return rebalancer.decide(
            self.state.get("holdings_reviews", {}),
            ranker_text,
            self.state.get("cash_balance"),
            self.state.get("macro_summary", ""),
            aggressiveness=self.settings.discover_rebalance_aggressiveness,
            history_block=history_block,
            market_themes_block=self.state.get("market_themes_block", ""),
            cc_context_block=self.state.get("cc_context_block", ""),
            harvest_block=self.state.get("harvest_block", ""),
            csp_context_block=self.state.get("csp_context_block", ""),
            accounts_block=format_accounts_block(
                self.state.get("account_cash") or {}, self.state.get("account_meta") or {}
            ),
            add_on_block=self._add_on_block(),
            backlog_block=self.state.get("backlog_block") or "",
            stub_income_block=self.state.get("stub_income_block") or "",
            obligations_block=covered_call_block(
                self.state.get("holdings_positions") or {},
                self.state.get("covered_call_obligations") or {},
            ),
        )

    def _record_lost_plan(self, e: Exception) -> StepOutput:
        # Every way the plan call can fail has to land here, not just bad
        # JSON. On 2026-09-20 the second attempt died on a ValueError
        # from the SDK (max_tokens too high to run unstreamed), which
        # the narrower `except RebalancePlanUnparseable` missed — so
        # the run reported the failure internally and still sent an
        # ordinary-looking email with an ordinary subject line.
        unparseable = e if isinstance(e, RebalancePlanUnparseable) else None
        # Do NOT let this pass as "no plan". Everything downstream —
        # the premortem, the report, the database — reads an absent
        # plan as a decision not to trade, which is the opposite of
        # what happened.
        logger.error(
            "The rebalance plan was not produced (%s: %s). The report will "
            "say so rather than render an empty action list.",
            type(e).__name__,
            e,
            exc_info=unparseable is None,
        )
        self.state["rebalance_plan"] = None
        self.state["rebalance_text"] = unparseable.raw_text if unparseable else ""
        if unparseable is None:
            note = f"The rebalancer did not return a plan ({type(e).__name__})"
        elif unparseable.truncated:
            note = "The rebalancer's plan was cut off before it finished"
        else:
            note = "The rebalancer's plan could not be read"
        self.state["rebalance_failed"] = note
        self.state["harvest_candidates"] = harvest_report_data(
            self.state.get("harvest_candidates_obj") or []
        )
        return StepOutput(content=f"rebalance: PLAN LOST ({type(e).__name__}: {e})")

    def _validate_covered_calls(self, plan: Any) -> Any:
        """Check WRITE_CALLs against the fetched chains. A crash keeps the
        unvalidated plan and says so."""
        try:
            plan, cc_warnings = apply_cc_plan_validation(
                plan,
                chains=self.state.get("cc_chains") or {},
                eligibility=self.state.get("cc_eligibility") or {},
                cc_context_block=self.state.get("cc_context_block") or "",
                settings=self.settings,
                spots={
                    t: px
                    for t, v in (self.state.get("holdings_technicals") or {}).items()
                    if (px := (v or {}).get("price"))
                },
            )
            if cc_warnings:
                self.state["cc_warnings"] = cc_warnings
        except Exception as e:
            logger.error(
                "CC validation crashed (%s) — using unvalidated plan. "
                "WRITE_CALL orphans / oversized contracts may slip through.",
                e,
                exc_info=True,
            )
            self.state["cc_warnings"] = [f"validation crashed: {e}"]
        return plan

    def _validate_puts(self, plan: Any) -> Any:
        """Check SELL_PUTs against the chains and cash caps. A crash drops
        every put: unvalidated puts could over-commit cash."""
        try:
            plan, csp_warnings = apply_csp_plan_validation(
                plan,
                chains=self.state.get("csp_chains") or {},
                eligibility=self.state.get("csp_eligibility") or {},
                cash_budget=self.state.get("csp_cash_budget") or 0.0,
                settings=self.settings,
                account_room=self.state.get("csp_account_room") or {},
                units={
                    t: float(p.get("units") or 0)
                    for t, p in (self.state.get("holdings_positions") or {}).items()
                },
                prices={
                    **{
                        t: (v or {}).get("price")
                        for t, v in (self.state.get("technicals") or {}).items()
                    },
                    **{
                        t: (v or {}).get("price")
                        for t, v in (self.state.get("holdings_technicals") or {}).items()
                    },
                },
            )
        except Exception as e:
            # Unvalidated puts could over-commit cash — drop them all.
            logger.error("CSP validation crashed (%s) — dropping all puts.", e, exc_info=True)
            plan = plan.model_copy(
                update={
                    "actions": [a for a in plan.actions if a.action != "SELL_PUT"],
                    "csp_writes": [],
                }
            )
            csp_warnings = [f"put validation crashed ({e}); all puts dropped"]
        if csp_warnings:
            self.state["csp_warnings"] = csp_warnings
        return plan

    def step_premortem(self, step_input: StepInput) -> StepOutput:
        """Adversarial hindsight on the rebalance plan: imagine reading the
        news 6 months from now where this plan went wrong, and write the
        post-mortem from that future. Skips on NO_ACTION (nothing to
        pre-mortem)."""
        plan = self.state.get("rebalance_plan")
        if plan is None:
            # An absent plan is a failure, not a decision. Saying
            # "NO_ACTION" here is what hid a lost plan for a whole run.
            self.state["premortem"] = None
            return StepOutput(content="premortem: skipped (no plan — the rebalance step failed)")
        if getattr(plan, "status", None) != "ACTION":
            self.state["premortem"] = None
            return StepOutput(content="premortem: skipped (plan recommends no action)")
        # Format the holdings_reviews into a single text blob for the agent.
        from ...models.llm import HoldingReview

        reviews_text = "\n\n".join(
            f"=== {ticker} ===\n{r.full_text if isinstance(r, HoldingReview) else r}"
            for ticker, r in self.state.get("holdings_reviews", {}).items()
        )
        agent = PreMortemAgent("claude", self.settings.discover_opus_model)
        try:
            premortem = agent.run(
                rebalance_plan_text=plan.full_text,
                ranker_text=self.state.get("ranker_text", ""),
                holdings_reviews_text=reviews_text,
            )
        except BudgetExceededError:
            self.state["premortem"] = None
            return StepOutput(content="premortem: skipped (cost cap)")
        self.state["premortem"] = premortem
        if premortem is None:
            return StepOutput(content="premortem: agent returned no content")
        return StepOutput(
            content=(
                f"Pre-mortem: verdict={premortem.overall_verdict}, "
                f"{len(premortem.failures)} failure mode(s)"
            )
        )

    def _add_on_block(self) -> str:
        from ...discover.add_on import format_add_on_block

        technicals = self.state.get("holdings_technicals") or {}
        values = {
            t: float(p.get("units") or 0) * float((technicals.get(t) or {}).get("price") or 0)
            for t, p in (self.state.get("holdings_positions") or {}).items()
        }
        return format_add_on_block(self.state.get("holdings_reviews") or {}, technicals, values)

    def _record_plan_suggestions(self, run_id: int) -> None:
        """Keep the plan's actions for the quarterly review."""
        from ...db.repository import record_suggestions

        plan = self.state.get("rebalance_plan")
        if plan is None or not plan.actions:
            return
        positions = self.state.get("holdings_positions") or {}
        prices = {
            **{t: (v or {}).get("price") for t, v in (self.state.get("technicals") or {}).items()},
            **{
                t: (v or {}).get("price")
                for t, v in (self.state.get("holdings_technicals") or {}).items()
            },
        }
        today = date.today().isoformat()
        # Where a sale's proceeds were meant to go. A sell is only good or
        # bad relative to what replaced it, so the pair has to be recorded
        # at the moment the advice is given — afterwards there is no way to
        # know which buy the sale was funding. The plan's own BUY/ADD lines
        # are the destinations, largest first.
        destinations = [a.ticker for a in plan.actions if a.action in {"BUY", "ADD"}]
        reinvest_into = destinations[0] if destinations else None
        rows = [
            {
                "suggested_on": today,
                "source": "rebalance",
                "action": a.action,
                "ticker": a.ticker,
                "detail": a.sizing,
                "price": prices.get(a.ticker),
                "units_held": (positions.get(a.ticker) or {}).get("units", 0.0),
                "run_id": run_id,
                "reinvest_into": (
                    reinvest_into
                    if a.action in {"SELL", "TRIM"} and reinvest_into != a.ticker
                    else None
                ),
            }
            for a in plan.actions
        ]
        try:
            with get_session(self.settings.discover_db_path) as session:
                record_suggestions(session, rows)
        except Exception as e:
            logger.warning("Could not record the plan's suggestions (%s)", e)

    def _reinvest_for_unfunded_sales(self) -> dict[str, Any] | None:
        """Ideas for proceeds the plan leaves without a destination (runs
        after persistence, so this run's picks are in the pool)."""
        from ...discover.reinvest import load_pick_pool, reinvest_ideas, unfunded_sales

        sold = unfunded_sales(self.state.get("rebalance_plan"))
        if not sold:
            return None
        broken = {
            c["ticker"] for c in self.state.get("thesis_checks") or [] if c["status"] == "BROKEN"
        }
        try:
            ideas = reinvest_ideas(
                load_pick_pool(self.settings.discover_db_path),
                held=set(self.state.get("holdings_positions") or {}),
                exclude=broken,
                n=3,
            )
        except Exception as e:
            logger.warning("reinvestment ideas failed (%s) — report goes without them", e)
            return None
        return {"sold": sold, "ideas": ideas}
