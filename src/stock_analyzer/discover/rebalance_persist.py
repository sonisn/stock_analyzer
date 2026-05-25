"""Persistence, delivery, and terminal output for rebalance pipeline runs."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from sqlmodel import Session

from ..config import Settings
from ..data.chart_img import fetch_charts
from ..db.repository import (
    insert_candidate,
    insert_holdings_review,
    insert_pick,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from .report import (
    parse_confidence,
    parse_verdict,
    print_terminal_summary,
)
from ..logging import current_log_file, get_logger
from ..reporting.smtp import SmtpServer

logger = get_logger(__name__)


def persist_rebalance_run(
    session: Session,
    *,
    state: dict[str, Any],
    settings: Settings,
    candidates: list[dict[str, Any]],
    survivors: list[Any],
    picks: list[tuple[int, str, str]],
    analyses: dict[str, Any],
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
) -> int:
    run_id = insert_run(
        session,
        universe_size=len(candidates),
        survivors=len(survivors),
        picks=len(picks),
        opus_model=settings.discover_opus_model,
        sonnet_model=settings.discover_sonnet_model,
        cash_budget=state.get("cash_balance"),
        kind="rebalance",
    )
    for c in candidates:
        insert_candidate(
            session,
            run_id,
            c["ticker"],
            passed_filter=c["passed_filter"],
            fail_reasons=c["fail_reasons"],
            score=c["score"],
            score_components=c["score_components"],
            score_breakdown=c["score_breakdown"],
            sources=c["sources"],
            conviction=c["conviction"],
            sector=c["sector"],
            price=c["price"],
        )
    for ticker, report in analyses.items():
        analyst_text = getattr(report, "full_text", None) or (
            report if isinstance(report, str) else ""
        )
        insert_scorecard(session, run_id, ticker, analyst_text)
    for ticker, review in state.get("holdings_reviews", {}).items():
        if not review:
            continue
        review_text = getattr(review, "full_text", None) or (
            review if isinstance(review, str) else ""
        )
        insert_holdings_review(
            session,
            run_id,
            ticker,
            verdict=parse_verdict(review),
            confidence=parse_confidence(review),
            review_text=review_text,
        )
    for rank, ticker, _ in picks:
        insert_pick(
            session,
            run_id,
            rank=rank,
            ticker=ticker,
            ranker_text=ranker_text,
            bear_case_text=redteam_text,
            allocation_text=sizer_text,
        )
    plan = state.get("rebalance_plan")
    insert_run_outputs(
        session,
        run_id,
        ranker_full=ranker_text,
        redteam_full=redteam_text,
        sizer_full=sizer_text,
        holdings_summary=state.get("holdings_summary", "") or "",
        rebalance_text=state.get("rebalance_text", "") or "",
        dashboard_data=plan.model_dump(mode="json") if plan else None,
    )
    return run_id


def fetch_pick_charts(
    picks: list[tuple[int, str, str]],
) -> tuple[dict[str, bytes], dict[str, str]]:
    pick_tickers = [t for _, t, _ in picks]
    charts: dict[str, bytes] = {}
    try:
        charts = fetch_charts(pick_tickers) if pick_tickers else {}
    except Exception as e:
        logger.warning("Chart fetch failed (%s) — report will omit charts", e)
    chart_cids = {t: f"chart-{t.replace('.', '-')}" for t in charts}
    return charts, chart_cids


def save_local_pdf(pdf_bytes: bytes, filename: str) -> Path:
    reports_dir = Path(
        os.path.expanduser(os.getenv("REPORTS_DIR", "~/.stock_analyzer/reports"))
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / filename
    path.write_bytes(pdf_bytes)
    return path


def deliver_rebalance_email(
    settings: Settings,
    *,
    subject: str,
    html_body: str,
    charts: dict[str, bytes],
    chart_cids: dict[str, str],
    pdf_bytes: bytes,
    pdf_filename: str,
) -> tuple[bool, str | None, Path]:
    local_pdf_path = save_local_pdf(pdf_bytes, pdf_filename)
    logger.info("Saved rebalance PDF locally: %s", local_pdf_path)

    delivered = False
    delivery_error: str | None = None
    if settings.email_to:
        try:
            SmtpServer().send_email(
                settings.email_to,
                subject,
                html_body,
                content_type="html",
                inline_images={
                    chart_cids[t]: data for t, data in charts.items()
                } or None,
                attachments=[(pdf_filename, pdf_bytes, "pdf")],
            )
            delivered = True
            logger.info("Sent rebalance email to %s", settings.email_to)
        except Exception as e:
            delivery_error = str(e)
            logger.error("Email delivery failed: %s", e)
    else:
        logger.warning("EMAIL_TO not set; skipping email")
    return delivered, delivery_error, local_pdf_path


def log_full_analysis(
    *,
    delivered: bool,
    delivery_error: str | None,
    local_pdf_path: str,
    rebalance_text: str,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    holdings_reviews: dict[str, Any],
) -> None:
    from ..models.llm import HoldingReview

    bar = "=" * 70
    if not delivered:
        logger.error(
            "%s\nEMAIL NOT DELIVERED — full analysis follows in this log.\n"
            "Reason: %s\nPDF: %s\n%s",
            bar,
            delivery_error or "EMAIL_TO not configured",
            local_pdf_path,
            bar,
        )
    logger.info("%s\nREBALANCE PLAN\n%s\n%s", bar, bar, rebalance_text)
    logger.info("%s\nRANKER — discover picks\n%s\n%s", bar, bar, ranker_text)
    logger.info("%s\nRED TEAM — bear cases\n%s\n%s", bar, bar, redteam_text)
    logger.info("%s\nSIZER — allocation\n%s\n%s", bar, bar, sizer_text)
    for ticker in sorted(holdings_reviews):
        review = holdings_reviews.get(ticker)
        text = (
            review.full_text if isinstance(review, HoldingReview)
            else (review or "(review unavailable)")
        )
        logger.info("%s\nHOLDING REVIEW — %s\n%s\n%s", bar, ticker, bar, text)


def print_rebalance_terminal(
    *,
    plan: object | None,
    cc_block: str,
    ranker_text: str,
    sizer_text: str,
    rebalance_text: str,
    local_pdf_path: Path,
) -> None:
    if plan is not None:
        n_writes = sum(1 for a in plan.actions if a.action == "WRITE_CALL")
        print("\n" + "=" * 60)
        print("COVERED-CALL SUMMARY")
        print("=" * 60)
        if n_writes > 0:
            gross = sum(
                ow.contracts * ow.est_premium_per_share * 100.0
                for ow in plan.option_writes
            )
            print(f"  Recommendations: {n_writes} WRITE_CALL action(s)")
            print(f"  Gross premium:   ${gross:,.0f}")
            for ow in plan.option_writes:
                print(
                    f"    {ow.ticker}: {ow.contracts}x ${ow.strike:.2f}C "
                    f"expires {ow.expiry}, Δ={ow.delta:.2f}, "
                    f"premium ${ow.contracts * ow.est_premium_per_share * 100:,.0f}"
                )
        elif not cc_block:
            print("  No recommendations: CC context was empty.")
            print(
                "  (No eligible ≥100-share holdings, CC_ENABLED=0, "
                "or chain fetch failed.)"
            )
        else:
            print(
                "  No recommendations: rebalancer declined to write calls this run."
            )
            print(f"  CC context ({len(cc_block)} chars) WAS provided to Opus.")

    print_terminal_summary(ranker_text, sizer_text)
    print("\n" + "=" * 60)
    print("REBALANCE PLAN")
    print("=" * 60)
    print(rebalance_text or "(no plan produced)")
    print(f"\nPDF saved: {local_pdf_path}")
    log_path = current_log_file()
    if log_path:
        print(f"Log file:  {log_path}")


def gross_premium_from_plan(plan: object | None) -> tuple[int, float]:
    gross_premium = 0.0
    action_count = 0
    if plan is not None and getattr(plan, "option_writes", None):
        gross_premium = sum(
            ow.contracts * ow.est_premium_per_share * 100.0
            for ow in plan.option_writes
        )
    if plan is not None:
        action_count = len(plan.actions)
    return action_count, gross_premium
