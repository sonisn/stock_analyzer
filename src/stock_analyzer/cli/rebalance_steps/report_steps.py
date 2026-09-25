"""The rebalance run's last step: persist, render and email the plan."""

from __future__ import annotations

from datetime import date
from typing import Any

from agno.workflow.types import StepInput, StepOutput

from ...db.session import get_session
from ...discover.rebalance_csp import (
    csp_report_data,
)
from ...discover.rebalance_persist import (
    deliver_rebalance_email,
    fetch_pick_charts,
    gross_premium_from_plan,
    log_full_analysis,
    persist_rebalance_run,
    print_rebalance_terminal,
)
from ...discover.report import (
    build_rebalance_sections,
    render_html_email,
    render_pdf,
)
from ...logging import get_logger
from ...usage import TRACKER
from ..pipeline_base import PipelineBase
from .helpers import (
    build_email_subject,
)

logger = get_logger("stock_analyzer.cli.rebalance")


class RebalanceReportSteps(PipelineBase):
    def step_persist_and_email_rebalance(self, step_input: StepInput) -> StepOutput:
        candidates = self.state.get("candidates") or []
        survivors = self.state.get("survivors") or []
        picks = self.state.get("picks") or []
        analyses = self.state.get("analyses") or {}
        ranker_text = self.state.get("ranker_text") or ""
        redteam_text = self.state.get("redteam_text") or ""
        sizer_text = self.state.get("sizer_text") or ""

        with get_session(self.settings.discover_db_path) as session:
            run_id = persist_rebalance_run(
                session,
                state=self.state,
                settings=self.settings,
                candidates=candidates,
                survivors=survivors,
                picks=picks,
                analyses=analyses,
                ranker_text=ranker_text,
                redteam_text=redteam_text,
                sizer_text=sizer_text,
            )

        self._record_plan_suggestions(run_id)
        self._record_pick_suggestions(run_id)
        charts, chart_cids = fetch_pick_charts(picks)
        reinvest = self._reinvest_for_unfunded_sales()
        sections = self._rebalance_report_sections(
            candidates=candidates,
            ranker_text=ranker_text,
            redteam_text=redteam_text,
            sizer_text=sizer_text,
            reinvest=reinvest,
        )
        html_body = render_html_email(sections, chart_cids)
        pdf_bytes = render_pdf(sections, charts)

        today = date.today()
        plan = self.state.get("rebalance_plan")
        action_count, gross_premium = _log_premium_summary(plan)

        subject = build_email_subject(
            action_count=action_count,
            gross_premium_usd=gross_premium,
            plan_failed=bool(self.state.get("rebalance_failed")),
        )
        pdf_filename = f"rebalance-{today.isoformat()}.pdf"
        delivered, delivery_error, local_pdf_path = deliver_rebalance_email(
            self.settings,
            subject=subject,
            html_body=html_body,
            charts=charts,
            chart_cids=chart_cids,
            pdf_bytes=pdf_bytes,
            pdf_filename=pdf_filename,
        )
        log_full_analysis(
            delivered=delivered,
            delivery_error=delivery_error,
            local_pdf_path=str(local_pdf_path),
            rebalance_text=self.state.get("rebalance_text", "") or "",
            ranker_text=ranker_text,
            redteam_text=redteam_text,
            sizer_text=sizer_text,
            holdings_reviews=self.state.get("holdings_reviews", {}),
        )
        print_rebalance_terminal(
            plan=plan,
            cc_block=self.state.get("cc_context_block") or "",
            ranker_text=ranker_text,
            sizer_text=sizer_text,
            rebalance_text=self.state.get("rebalance_text", "") or "",
            local_pdf_path=local_pdf_path,
        )

        self.state["run_id"] = run_id
        self.state["pdf_bytes"] = pdf_bytes
        self.state["local_pdf_path"] = str(local_pdf_path)
        status = "emailed" if delivered else "persisted (no email)"
        return StepOutput(
            content=(
                f"Rebalance run #{run_id} {status}; PDF {len(pdf_bytes)} bytes "
                f"(saved to {local_pdf_path})"
            )
        )

    def _rebalance_report_sections(
        self,
        *,
        candidates: list[dict[str, Any]],
        ranker_text: str,
        redteam_text: str,
        sizer_text: str,
        reinvest: dict[str, Any] | None,
    ) -> list[Any]:
        """The rebalance report's section list, from the run's state."""
        return build_rebalance_sections(
            rebalance_text=self.state.get("rebalance_text", "") or "",
            holdings_reviews=self.state.get("holdings_reviews", {}),
            ranker_text=ranker_text,
            redteam_text=redteam_text,
            sizer_text=sizer_text,
            candidates=candidates,
            cash_balance=self.state.get("cash_balance"),
            macro_summary=self.state.get("macro_summary", ""),
            sector_rotation=self.state.get("sector_rotation"),
            holdings_positions=self.state.get("holdings_positions", {}),
            holdings_technicals=self.state.get("holdings_technicals", {}),
            holdings_fundamentals=self.state.get("holdings_fundamentals", {}),
            track_record_block=self.state.get("track_record_block", ""),
            track_record=self.state.get("track_record"),
            thesis_checks=self.state.get("thesis_checks"),
            harvest_candidates=self.state.get("harvest_candidates"),
            rebalance_plan=self.state.get("rebalance_plan"),
            market_themes=self.state.get("market_themes"),
            premortem=self.state.get("premortem"),
            holdings_news=self.state.get("news"),
            cc_eligibility=self.state.get("cc_eligibility") or {},
            cc_round_lot_coverage=self.state.get("cc_round_lot_coverage") or {},
            cc_stub_pool_total_usd=self.state.get("cc_stub_pool_total_usd") or 0.0,
            cc_warnings=(self.state.get("cc_warnings") or [])
            + (self.state.get("sale_warnings") or [])
            + sorted((self.state.get("cc_cheap_premium") or {}).values()),
            cc_slippage_buffer=self.settings.cc_slippage_buffer,
            csp_summary=csp_report_data(
                self.state.get("rebalance_plan"),
                cash_budget=self.state.get("csp_cash_budget") or 0.0,
            ),
            csp_warnings=(
                ([self.state["csp_blocked_note"]] if self.state.get("csp_blocked_note") else [])
                + sorted((self.state.get("csp_cheap_premium") or {}).values())
                + (self.state.get("csp_warnings") or [])
            ),
            reinvest=reinvest,
            stop_loss_warnings=self.state.get("stop_loss_warnings") or [],
            stale_accounts=self.state.get("stale_accounts") or [],
            usage=TRACKER.report_data(),
            plan_failure=self.state.get("rebalance_failed"),
            ranker_output=self.state.get("ranker_output"),
            redteam_output=self.state.get("redteam_output"),
            sizer_output=self.state.get("sizer_output"),
        )


def _log_premium_summary(plan: Any) -> tuple[int, float]:
    """(action count, gross premium), logging the covered-call summary."""
    action_count, gross_premium = gross_premium_from_plan(plan)
    if gross_premium > 0:
        logger.info(
            "CC summary at email time: %d WRITE_CALL(s), $%s gross premium "
            "across %d total action(s)",
            sum(1 for a in plan.actions if a.action == "WRITE_CALL"),
            f"{gross_premium:,.0f}",
            action_count,
        )
    return action_count, gross_premium
