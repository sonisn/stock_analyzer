"""Rebalance report section IR — HTML/PDF layout for portfolio rebalance runs."""
from __future__ import annotations

from datetime import date
from typing import Any

from ..models.reports import PreMortem, Section
from ..models.rebalance import RebalancePlan
from .report_sections import (
    build_sections,
    parse_confidence,
    parse_rebalance_status,
    parse_verdict,
)


def build_holdings_dashboard_rows(
    *,
    holdings_positions: dict[str, dict[str, Any]],
    holdings_technicals: dict[str, dict[str, Any]],
    holdings_fundamentals: dict[str, dict[str, Any]],
    holdings_reviews: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, float, dict[str, float], float | None]:
    """Return dashboard rows, total value, total cost, sector map, total P/L %."""
    dashboard_rows: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0
    sector_value: dict[str, float] = {}
    for ticker in sorted(holdings_positions.keys()):
        pos = holdings_positions[ticker]
        tech = holdings_technicals.get(ticker, {})
        fund = holdings_fundamentals.get(ticker, {})
        review = holdings_reviews.get(ticker) or ""
        current = tech.get("price")
        units = pos.get("units", 0)
        cost = pos.get("cost_basis", 0)
        value = (current or 0) * units
        total_value += value
        total_cost += cost
        pnl_pct = ((value - cost) / cost * 100) if cost else None
        sector = fund.get("sector") or "Unknown"
        if value > 0:
            sector_value[sector] = sector_value.get(sector, 0) + value
        dashboard_rows.append({
            "ticker": ticker,
            "verdict": parse_verdict(review),
            "confidence": parse_confidence(review),
            "pnl_pct": pnl_pct,
            "sector": sector,
            "note": "",
        })
    total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost else None
    return dashboard_rows, total_value, total_cost, sector_value, total_pnl_pct


def append_rebalance_overview(
    sections: list[Section],
    *,
    today: str,
    status: str,
    status_label: str,
    holdings_positions: dict[str, dict[str, Any]],
    cash_balance: float | None,
    dashboard_rows: list[dict[str, Any]],
    total_value: float,
    total_pnl_pct: float | None,
    holdings_news: dict[str, list[dict[str, Any]]] | None,
    sector_value: dict[str, float],
    track_record_block: str,
    market_themes: object,
    macro_summary: str,
) -> None:
    sections.extend([
        Section(kind="heading", text=f"Portfolio Rebalance — {today}", level=1),
        Section(kind="status_banner", text=status_label, status=status),
    ])
    metrics: list[tuple[str, str]] = [
        ("Holdings", f"{len(holdings_positions)}"),
        ("Portfolio value", f"${total_value:,.0f}" if total_value else "—"),
        ("Total P/L", f"{total_pnl_pct:+.1f}%" if total_pnl_pct is not None else "—"),
        ("Cash", f"${cash_balance:,.0f}" if cash_balance is not None else "—"),
    ]
    sections.append(Section(kind="metric_strip", metrics=metrics))

    if dashboard_rows:
        sections.append(Section(kind="heading", text="Holdings dashboard", level=2))
        sections.append(Section(kind="holdings_dashboard", holdings=dashboard_rows))

    catalyst_rows: list[list[str]] = []
    if holdings_news:
        for ticker in sorted(holdings_positions.keys()):
            items = holdings_news.get(ticker) or []
            for item in items[:2]:
                title = (item.get("title") or "").strip()
                if title:
                    catalyst_rows.append([ticker, title])
    if catalyst_rows:
        sections.append(Section(
            kind="heading", text="Recent catalysts (informational)", level=2,
        ))
        sections.append(Section(
            kind="para",
            text=(
                "Headlines worth scanning. Not used to compute verdicts or "
                "position sizing — your Reviewer/Rebalancer reads news as "
                "qualitative context only."
            ),
        ))
        sections.append(Section(
            kind="table",
            table_header=["Ticker", "Headline"],
            table_rows=catalyst_rows,
        ))

    if sector_value:
        pie_data = sorted(sector_value.items(), key=lambda x: x[1], reverse=True)
        sections.append(Section(kind="heading", text="Sector allocation", level=2))
        sections.append(Section(kind="sector_pie", pie_data=pie_data))

    if track_record_block:
        sections.append(Section(kind="heading", text="Track record", level=2))
        sections.append(Section(kind="preformatted", text=track_record_block))

    from ..models.llm import MarketThemes
    if isinstance(market_themes, MarketThemes) and market_themes.themes:
        sections.append(Section(kind="heading", text="Current market themes", level=2))
        sections.append(Section(
            kind="market_themes_panel",
            data={
                "themes": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "strength": t.strength,
                        "trending": t.trending,
                        "member_tickers": list(t.member_tickers),
                    }
                    for t in market_themes.themes
                ],
            },
        ))

    if macro_summary:
        sections.append(Section(kind="heading", text="Macro regime", level=2))
        sections.append(Section(kind="blockquote", text=macro_summary))


def append_rebalance_plan_body(
    sections: list[Section],
    *,
    rebalance_text: str,
    rebalance_plan: object,
    premortem: object,
    cash_balance: float | None,
    cc_round_lot_coverage: dict[str, Any] | None,
    cc_warnings: list[str] | None,
    cc_slippage_buffer: float,
) -> None:
    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Rebalance plan (action list)", level=1))

    plan = rebalance_plan if isinstance(rebalance_plan, RebalancePlan) else None
    if plan and plan.actions:
        sections.append(Section(
            kind="rebalance_action_table",
            data={
                "actions": [
                    {"action": a.action, "ticker": a.ticker, "sizing": a.sizing}
                    for a in plan.actions
                ],
                "summary": plan.summary,
            },
        ))

    if isinstance(premortem, PreMortem) and (premortem.failures or premortem.summary):
        sections.append(Section(
            kind="heading", text="Pre-mortem (adversarial hindsight)", level=2,
        ))
        sections.append(Section(
            kind="premortem_panel",
            data={
                "overall_verdict": premortem.overall_verdict,
                "summary": premortem.summary,
                "failures": [
                    {
                        "likelihood": f.likelihood,
                        "severity": f.severity,
                        "triggering_action": f.triggering_action,
                        "failure_narrative": f.failure_narrative,
                        "early_warning": f.early_warning,
                    }
                    for f in premortem.failures
                ],
            },
        ))

    from .cc_render import (
        compute_premium_deployment,
        compute_premium_income,
        compute_round_lot_summary,
    )
    if plan is not None and plan.option_writes:
        sections.append(Section(
            kind="premium_income",
            data=compute_premium_income(plan, slippage_buffer=cc_slippage_buffer),
        ))
    if cc_round_lot_coverage:
        rls = compute_round_lot_summary(cc_round_lot_coverage)
        if rls["rows"]:
            sections.append(Section(kind="round_lot_coverage", data=rls))
    if plan is not None and (
        plan.option_writes
        or any(
            a.action in ("ADD", "BUY")
            or (a.action == "TRIM" and "stub" in a.sizing.lower())
            for a in plan.actions
        )
    ):
        stub_usd = 0.0
        if cc_round_lot_coverage:
            for a in plan.actions:
                if a.action == "TRIM" and "stub" in a.sizing.lower():
                    rec = cc_round_lot_coverage.get(a.ticker)
                    if rec is not None:
                        stub_usd += getattr(rec, "stub_dollar_value", 0.0)
        deployment = compute_premium_deployment(
            plan, cash_balance=cash_balance, slippage_buffer=cc_slippage_buffer,
            stub_consolidation_usd=stub_usd,
        )
        if (
            deployment["gross_premium_usd"] > 0
            or deployment["deployments"]
            or stub_usd > 0
        ):
            sections.append(Section(kind="premium_deployment", data=deployment))

    if cc_warnings:
        sections.append(Section(
            kind="para",
            text="CC plan adjustments: " + "; ".join(cc_warnings),
        ))

    sections.append(Section(kind="preformatted", text=rebalance_text))


def append_holding_review_sections(
    sections: list[Section],
    holdings_reviews: dict[str, Any],
) -> None:
    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Per-holding reviews", level=1))
    from ..models.llm import HoldingReview
    for ticker in sorted(holdings_reviews.keys()):
        review = holdings_reviews[ticker]
        if isinstance(review, HoldingReview):
            sections.append(Section(
                kind="holding_review_card",
                data={
                    "ticker": ticker,
                    "verdict": review.verdict,
                    "confidence": review.confidence,
                    "trim_pct": review.trim_pct,
                    "position_context": review.position_context,
                    "forward_outlook": review.forward_outlook,
                    "reasoning": review.reasoning,
                    "tax_lot_plan": list(review.tax_lot_plan),
                    "what_would_change_mind": review.what_would_change_mind,
                    "wash_sale_notice": review.wash_sale_notice,
                },
            ))
        else:
            text = review or ""
            sections.append(Section(kind="heading", text=ticker, level=2))
            sections.append(Section(kind="preformatted", text=text))


def append_discover_appendix(
    sections: list[Section],
    *,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    candidates: list[dict[str, Any]],
    sector_rotation: dict[str, Any] | None,
) -> None:
    discover_sections = build_sections(
        ranker_text=ranker_text,
        redteam_text=redteam_text,
        sizer_text=sizer_text,
        candidates=candidates,
        universe_size=len(candidates),
        holdings_summary="",
        macro_summary="",
        sector_rotation=sector_rotation,
    )
    sections.append(Section(kind="page_break"))
    sections.append(
        Section(kind="heading", text="Discover picks (input to rebalancer)", level=1)
    )
    sections.extend(discover_sections[2:])


def build_rebalance_sections(
    *,
    rebalance_text: str,
    holdings_reviews: dict[str, Any],
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    candidates: list[dict[str, Any]],
    cash_balance: float | None,
    macro_summary: str,
    sector_rotation: dict[str, Any] | None,
    holdings_positions: dict[str, dict[str, Any]],
    holdings_technicals: dict[str, dict[str, Any]],
    holdings_fundamentals: dict[str, dict[str, Any]],
    track_record_block: str = "",
    rebalance_plan: object = None,
    market_themes: object = None,
    premortem: object = None,
    holdings_news: dict[str, list[dict[str, Any]]] | None = None,
    cc_eligibility: dict[str, Any] | None = None,
    cc_round_lot_coverage: dict[str, Any] | None = None,
    cc_stub_pool_total_usd: float = 0.0,
    cc_warnings: list[str] | None = None,
    cc_slippage_buffer: float = 0.10,
) -> list[Section]:
    """Rebalance-specific layout — status banner + metrics + dashboard +
    sector pie at the top, then the LLM's plan + per-holding reviews +
    discover-picks appendix."""
    del cc_eligibility, cc_stub_pool_total_usd  # reserved for future section use

    today = date.today().isoformat()
    status = parse_rebalance_status(rebalance_plan or rebalance_text)
    status_label = (
        "STATUS: NO ACTION RECOMMENDED" if status == "NO_ACTION"
        else "STATUS: ACTION RECOMMENDED" if status == "ACTION"
        else "STATUS: REVIEW REQUIRED"
    )

    dashboard_rows, total_value, _total_cost, sector_value, total_pnl_pct = (
        build_holdings_dashboard_rows(
            holdings_positions=holdings_positions,
            holdings_technicals=holdings_technicals,
            holdings_fundamentals=holdings_fundamentals,
            holdings_reviews=holdings_reviews,
        )
    )

    sections: list[Section] = []
    append_rebalance_overview(
        sections,
        today=today,
        status=status,
        status_label=status_label,
        holdings_positions=holdings_positions,
        cash_balance=cash_balance,
        dashboard_rows=dashboard_rows,
        total_value=total_value,
        total_pnl_pct=total_pnl_pct,
        holdings_news=holdings_news,
        sector_value=sector_value,
        track_record_block=track_record_block,
        market_themes=market_themes,
        macro_summary=macro_summary,
    )
    append_rebalance_plan_body(
        sections,
        rebalance_text=rebalance_text,
        rebalance_plan=rebalance_plan,
        premortem=premortem,
        cash_balance=cash_balance,
        cc_round_lot_coverage=cc_round_lot_coverage,
        cc_warnings=cc_warnings,
        cc_slippage_buffer=cc_slippage_buffer,
    )
    append_holding_review_sections(sections, holdings_reviews)
    append_discover_appendix(
        sections,
        ranker_text=ranker_text,
        redteam_text=redteam_text,
        sizer_text=sizer_text,
        candidates=candidates,
        sector_rotation=sector_rotation,
    )
    return sections
