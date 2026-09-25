"""Discover steps that finish a run: persist, report, email, dashboard refresh,
history upkeep."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from agno.workflow.types import StepInput, StepOutput

from ...data.chart_img import fetch_charts
from ...db.repository import (
    insert_run,
    insert_run_outputs,
)
from ...db.session import get_session
from ...discover.catalysts import catalysts_to_dicts
from ...discover.factor_tilt import average_factor_tilts, compute_factor_tilt
from ...discover.report import (
    build_sections,
    print_terminal_summary,
    render_html_email,
    render_pdf,
)
from ...discover.run_records import (
    insert_candidates,
    insert_picks,
    insert_scorecards,
    insert_snapshots,
)
from ...logging import current_log_file, get_logger
from ...reporting.smtp import SmtpServer
from ...usage import TRACKER
from ..pipeline_base import PipelineBase
from .helpers import (
    _log_discover_analysis,
    _save_local_pdf,
)

logger = get_logger("stock_analyzer.cli.discover")


class ReportSteps(PipelineBase):
    def step_refresh_dashboard(self, step_input: StepInput) -> StepOutput:
        """Rewrite the static dashboard so it reflects the run that just
        finished.

        The page is a generated file, not a service, so it only changes
        when something regenerates it. Cron does that after the close; a
        run started by hand would otherwise leave the page showing the
        previous day's plan while the email showed the new one — the two
        disagreeing is worse than the page being a few hours stale.

        Never fails the run. The plan is already emailed and persisted by
        the time this executes; a dashboard that did not refresh is a
        cosmetic problem, and the next scheduled build fixes it.
        """
        if not self.settings.dashboard_after_run:
            return StepOutput(content="dashboard: disabled via DASHBOARD_AFTER_RUN=0")
        from datetime import date as _date
        from pathlib import Path

        from ...cli.dashboard import collect
        from ...dashboard_page import render_page

        try:
            data = collect(self.settings, today=_date.today())
            out = Path(self.settings.dashboard_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render_page(data))
        except Exception as e:  # noqa: BLE001 — cosmetic, never fatal
            logger.warning("Dashboard refresh failed (%s) — the scheduled build will retry", e)
            return StepOutput(content=f"dashboard: failed ({type(e).__name__})")
        return StepOutput(
            content=(
                f"dashboard: {len(data['holdings'])} holding(s), "
                f"{len(data['suggestions'])} graded suggestion(s) -> {out}"
            )
        )

    def step_history_upkeep(self, step_input: StepInput) -> StepOutput:
        """Last step of every run: add new history, trim old (db/retention.py).
        Never fails the run — every sub-step only logs on error."""
        if not self.settings.history_upkeep:
            return StepOutput(content="history upkeep: disabled")
        from ...db.retention import RetentionPolicy, default_file_targets, run_history_upkeep

        report = run_history_upkeep(
            self.settings.discover_db_path,
            policy=RetentionPolicy(
                text_days=self.settings.history_text_retention_days,
                session_days=self.settings.history_session_retention_days,
                candidate_days=self.settings.history_candidate_retention_days,
                keep_models=self.settings.history_keep_model_versions,
                file_days=self.settings.history_file_retention_days,
                reference_days=self.settings.history_reference_retention_days,
                vacuum_min_free_pct=self.settings.history_vacuum_min_free_pct,
                warn_mb=self.settings.history_db_warn_mb,
            ),
            file_targets=default_file_targets(self.settings.model_cache_dir),
        )
        return StepOutput(content=report.summary())

    def _record_pick_suggestions(self, run_id: int) -> None:
        """Keep this run's picks in the suggestions ledger with the shares
        already held, so the quarterly review can tell a new buy from a
        pick you owned before it was made."""
        from ...db.repository import record_suggestions
        from ...reporting.health import aggregate_positions

        picks = self.state.get("picks") or []
        if not picks:
            return
        held = aggregate_positions(self.state.get("holdings_raw") or {})
        prices = {c["ticker"]: c.get("price") for c in self.state.get("candidates") or []}
        rows = [
            {
                "suggested_on": date.today().isoformat(),
                "source": "discover",
                "action": "BUY",
                "ticker": ticker,
                "detail": f"discover pick #{rank}",
                "price": prices.get(ticker),
                "units_held": (held.get(ticker) or {}).get("units", 0.0)
                if self.state.get("holdings_raw") is not None
                else None,
                "run_id": run_id,
            }
            for rank, ticker, _ in picks
        ]
        try:
            with get_session(self.settings.discover_db_path) as session:
                record_suggestions(session, rows)
        except Exception as e:
            logger.warning("Could not record the picks as suggestions (%s)", e)

    def step_persist_and_report(self, step_input: StepInput) -> StepOutput:
        run_id = self._persist_run()
        self._record_pick_suggestions(run_id)

        # Fetch a chart for each pick (existing chart-img.com client).
        pick_tickers = [t for _, t, _ in self.state["picks"]]
        charts: dict[str, bytes] = {}
        try:
            charts = fetch_charts(pick_tickers)
        except Exception as e:
            logger.warning("Chart fetch failed (%s) — report will omit charts", e)
        chart_cids = {t: f"chart-{t.replace('.', '-')}" for t in charts}

        # One section list, rendered to both HTML and PDF.
        sections = self._report_sections(pick_tickers)
        html_body = render_html_email(sections, chart_cids)
        pdf_bytes = render_pdf(sections, charts)

        delivered, delivery_error, local_pdf_path = self._deliver_report(
            run_id, pick_tickers, html_body, pdf_bytes, charts, chart_cids
        )

        # Always dump the full analysis to the log so the user can recover
        # every section — even when email is offline or wasn't configured.
        _log_discover_analysis(
            delivered=delivered,
            delivery_error=delivery_error,
            local_pdf_path=str(local_pdf_path),
            ranker_text=self.state["ranker_text"],
            redteam_text=self.state["redteam_text"],
            sizer_text=self.state["sizer_text"],
        )

        self.state["run_id"] = run_id
        self.state["pdf_bytes"] = pdf_bytes
        self.state["html_body"] = html_body
        self.state["local_pdf_path"] = str(local_pdf_path)
        print_terminal_summary(self.state["ranker_text"], self.state["sizer_text"])
        print(f"\nPDF saved: {local_pdf_path}")
        log_path = current_log_file()
        if log_path:
            print(f"Log file:  {log_path}")
        status = "emailed" if delivered else "persisted (no email)"
        return StepOutput(
            content=(
                f"Run #{run_id} {status}; PDF {len(pdf_bytes)} bytes (saved to {local_pdf_path})"
            )
        )

    def _persist_run(self) -> int:
        """The run, its candidates, snapshots, scorecards, picks and outputs."""
        with get_session(self.settings.discover_db_path) as session:
            run_id = insert_run(
                session,
                universe_size=len(self.state["candidates"]),
                survivors=len(self.state["survivors"]),
                picks=len(self.state["picks"]),
                opus_model=self.settings.discover_opus_model,
                sonnet_model=self.settings.discover_sonnet_model,
                cash_budget=self.settings.discover_cash_budget,
                kind="discover",
            )
            insert_candidates(session, run_id, self.state["candidates"])
            insert_snapshots(
                session,
                run_id,
                self.state["candidates"],
                self.state.get("fundamentals") or {},
                self.state.get("eps_revisions") or {},
            )
            insert_scorecards(session, run_id, self.state["analyses"])
            insert_picks(
                session,
                run_id,
                self.state["picks"],
                ranker_output=self.state.get("ranker_output"),
                candidates=self.state["candidates"],
                analyses=self.state["analyses"],
            )
            insert_run_outputs(
                session,
                run_id,
                ranker_full=self.state["ranker_text"],
                redteam_full=self.state["redteam_text"],
                sizer_full=self.state["sizer_text"],
                holdings_summary=self.state["holdings_summary"],
            )
        return run_id

    def _pick_factor_tilts(
        self, pick_tickers: list[str]
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """Style factor tilt — remap each pick's existing score_breakdown
        leaves into named growth/value/quality/momentum/low_vol buckets
        for reporting only (no rescoring)."""
        candidates_by_ticker = {c["ticker"]: c for c in self.state["candidates"]}
        hv_data = self.state.get("historical_volatility") or {}
        pick_tilts: dict[str, dict[str, float]] = {}
        for ticker in pick_tickers:
            cand = candidates_by_ticker.get(ticker)
            if cand is None:
                continue
            tilt = compute_factor_tilt(cand.get("score_breakdown"), hv_data.get(ticker))
            if tilt:
                pick_tilts[ticker] = tilt
        portfolio_tilt = average_factor_tilts(list(pick_tilts.values()))
        return pick_tilts, portfolio_tilt

    def _report_sections(self, pick_tickers: list[str]) -> list[Any]:
        pick_tilts, portfolio_tilt = self._pick_factor_tilts(pick_tickers)
        analyses = self.state.get("analyses") or {}
        pick_catalysts = {
            t: catalysts_to_dicts(analyses[t].upcoming_catalysts)
            for t in pick_tickers
            if t in analyses
        }

        return build_sections(
            ranker_text=self.state["ranker_text"],
            redteam_text=self.state["redteam_text"],
            sizer_text=self.state["sizer_text"],
            candidates=self.state["candidates"],
            universe_size=len(self.state["candidates"]),
            holdings_summary=self.state["holdings_summary"],
            holdings_rows=self.state.get("holdings_table_rows"),
            macro_summary=self.state.get("macro_summary", ""),
            sector_rotation=self.state.get("sector_rotation"),
            track_record_block=self.state.get("track_record_block", ""),
            track_record=self.state.get("track_record"),
            ranker_output=self.state.get("ranker_output"),
            redteam_output=self.state.get("redteam_output"),
            sizer_output=self.state.get("sizer_output"),
            market_themes=self.state.get("market_themes"),
            data_warnings=(
                (self.state.get("output_validation_warnings") or [])
                + (self.state.get("macro_veto_reasons") or [])
                + (self.state.get("catalyst_warnings") or [])
            ),
            pick_tilts=pick_tilts,
            portfolio_tilt=portfolio_tilt,
            pick_catalysts=pick_catalysts,
            usage=TRACKER.report_data(),
            paper_ledger=self.state.get("paper_ledger"),
            thesis_checks=self.state.get("thesis_checks"),
        )

    def _deliver_report(
        self,
        run_id: int,
        pick_tickers: list[str],
        html_body: str,
        pdf_bytes: bytes,
        charts: dict[str, bytes],
        chart_cids: dict[str, str],
    ) -> tuple[bool, str | None, Path]:
        """Save the PDF, then email it (or log that EMAIL_TO is unset).
        Returns (delivered, delivery_error, local_pdf_path)."""
        today = date.today()
        picks_summary = ", ".join(pick_tickers[:5])
        subject = (
            f"Stock Discovery — {today.strftime('%b-%d')}: {picks_summary}"
            if pick_tickers
            else f"Stock Discovery — {today.strftime('%b-%d')}"
        )
        pdf_filename = f"discover-{today.isoformat()}.pdf"

        # Save PDF locally BEFORE the email attempt so a delivery failure
        # (SMTP outage, wrong creds, etc.) never costs the user the report.
        local_pdf_path = _save_local_pdf(pdf_bytes, pdf_filename)
        logger.info("Saved discover PDF locally: %s", local_pdf_path)

        delivered = False
        delivery_error: str | None = None
        if self.settings.email_to:
            try:
                SmtpServer().send_email(
                    self.settings.email_to,
                    subject,
                    html_body,
                    content_type="html",
                    inline_images={chart_cids[t]: data for t, data in charts.items()} or None,
                    attachments=[(pdf_filename, pdf_bytes, "pdf")],
                )
                delivered = True
                logger.info("Sent discovery email to %s", self.settings.email_to)
            except Exception as e:
                delivery_error = str(e)
                logger.error("Email delivery failed: %s", e)
        else:
            logger.warning(
                "EMAIL_TO not set; skipping email delivery. "
                "Run %d's HTML/PDF available via state if you want to inspect them.",
                run_id,
            )
        return delivered, delivery_error, local_pdf_path
