"""Rebalance report section IR — HTML/PDF layout for portfolio rebalance runs."""

from __future__ import annotations

from datetime import date
from typing import Any

from ..models.rebalance import RebalancePlan
from ..models.reports import PreMortem, Section
from ..models.track_record import TrackRecord
from .report_sections import (
    append_thesis_check_section,
    append_track_record_section,
    append_usage_section,
    build_sections,
    market_themes_sections,
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
        dashboard_rows.append(
            {
                "ticker": ticker,
                "verdict": parse_verdict(review),
                "confidence": parse_confidence(review),
                "pnl_pct": pnl_pct,
                "sector": sector,
                "note": "",
            }
        )
    total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost else None
    return dashboard_rows, total_value, total_cost, sector_value, total_pnl_pct


def append_rebalance_glance(
    sections: list[Section],
    *,
    rebalance_plan: object,
    thesis_checks: list[dict[str, Any]] | None,
    harvest_candidates: list[dict[str, Any]] | None,
    stop_loss_warnings: list[str] | None,
    stale_accounts: list[str] | None = None,
) -> None:
    """'At a glance' under the status banner: the plan's actions in order,
    then what else needs a decision. The full plan, reviews and history
    follow further down."""
    plan = rebalance_plan if isinstance(rebalance_plan, RebalancePlan) else None
    rows = [[a.action, a.ticker, a.sizing] for a in (plan.actions if plan else [])]
    flags: list[str] = []
    # First, because a frozen connection means the shares, cash and
    # collateral the plan below was sized against are out of date.
    for note in stale_accounts or []:
        flags.append(f"Stale account data — {note}.")
    for c in thesis_checks or []:
        if c["status"] == "BROKEN":
            flags.append(
                f"{c['ticker']}: original thesis broken ({c['return_pct']:+.1f}% since pick)."
            )
        elif c["status"] == "TARGET HIT":
            flags.append(
                f"{c['ticker']}: past its bull target ({c['return_pct']:+.1f}% since pick)."
            )
    for w in stop_loss_warnings or []:
        flags.append(w.rstrip(".") + ".")
    if harvest_candidates:
        loss = sum(-c["loss_usd"] for c in harvest_candidates)
        saving = sum(c["est_tax_saving_usd"] for c in harvest_candidates)
        flags.append(
            f"Tax-loss harvesting: {len(harvest_candidates)} candidate(s), ${loss:,.0f} of losses "
            f"(~${saving:,.0f} tax) — see the harvesting table."
        )
    if not rows and not flags:
        return
    sections.append(Section(kind="heading", text="At a glance", level=2))
    if rows:
        sections.append(
            Section(kind="table", table_header=["Action", "Ticker", "Size"], table_rows=rows)
        )
    if not flags:
        sections.append(Section(kind="para", text="Nothing else needs a decision."))
        return
    sections.append(Section(kind="para", text="Also decide:"))
    sections.extend(Section(kind="para", text=f"• {f}") for f in flags)


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
    track_record: TrackRecord | None,
    thesis_checks: list[dict[str, Any]] | None,
    market_themes: object,
    rebalance_plan: object = None,
    harvest_candidates: list[dict[str, Any]] | None = None,
    stop_loss_warnings: list[str] | None = None,
    stale_accounts: list[str] | None = None,
    macro_summary: str,
) -> None:
    sections.extend(
        [
            Section(kind="heading", text=f"Portfolio Rebalance — {today}", level=1),
            Section(kind="status_banner", text=status_label, status=status),
        ]
    )
    metrics: list[tuple[str, str]] = [
        ("Holdings", f"{len(holdings_positions)}"),
        ("Portfolio value", f"${total_value:,.0f}" if total_value else "—"),
        ("Total P/L", f"{total_pnl_pct:+.1f}%" if total_pnl_pct is not None else "—"),
        ("Cash", f"${cash_balance:,.0f}" if cash_balance is not None else "—"),
    ]
    sections.append(Section(kind="metric_strip", metrics=metrics))
    append_rebalance_glance(
        sections,
        rebalance_plan=rebalance_plan,
        thesis_checks=thesis_checks,
        harvest_candidates=harvest_candidates,
        stop_loss_warnings=stop_loss_warnings,
        stale_accounts=stale_accounts,
    )

    if dashboard_rows:
        sections.append(Section(kind="heading", text="Holdings dashboard", level=2))
        sections.append(Section(kind="holdings_dashboard", holdings=dashboard_rows))

    _append_catalyst_headlines(sections, holdings_positions, holdings_news)

    if sector_value:
        pie_data = sorted(sector_value.items(), key=lambda x: x[1], reverse=True)
        sections.append(Section(kind="heading", text="Sector allocation", level=2))
        sections.append(Section(kind="sector_pie", pie_data=pie_data))

    append_track_record_section(sections, track_record, track_record_block)
    append_thesis_check_section(sections, thesis_checks)

    sections.extend(market_themes_sections(market_themes))

    if macro_summary:
        sections.append(Section(kind="heading", text="Macro regime", level=2))
        sections.append(Section(kind="blockquote", text=macro_summary))


def _append_catalyst_headlines(
    sections: list[Section],
    holdings_positions: dict[str, dict[str, Any]],
    holdings_news: dict[str, list[dict[str, Any]]] | None,
) -> None:
    """Up to two headlines per holding, for reading — not for the verdicts."""
    catalyst_rows: list[list[str]] = []
    if holdings_news:
        for ticker in sorted(holdings_positions.keys()):
            items = holdings_news.get(ticker) or []
            for item in items[:2]:
                title = (item.get("title") or "").strip()
                if title:
                    catalyst_rows.append([ticker, title])
    if catalyst_rows:
        sections.append(
            Section(
                kind="heading",
                text="Recent catalysts (informational)",
                level=2,
            )
        )
        sections.append(
            Section(
                kind="para",
                text=(
                    "Headlines worth scanning. Not used to compute verdicts or "
                    "position sizing — your Reviewer/Rebalancer reads news as "
                    "qualitative context only."
                ),
            )
        )
        sections.append(
            Section(
                kind="table",
                table_header=["Ticker", "Headline"],
                table_rows=catalyst_rows,
            )
        )


def append_harvest_section(
    sections: list[Section], candidates: list[dict[str, Any]] | None
) -> None:
    """'Tax-loss harvesting candidates' — deterministic, taxable accounts
    only. One row per account slice; the specific loss lots and any wash-
    sale or plan conflict go in the notes column."""
    if not candidates:
        return
    total_loss = sum(-c["loss_usd"] for c in candidates)
    total_saving = sum(c["est_tax_saving_usd"] for c in candidates)
    sections.append(Section(kind="heading", text="Tax-loss harvesting candidates", level=2))
    sections.append(
        Section(
            kind="para",
            text=(
                f"{len(candidates)} taxable position slice(s) could realize "
                f"${total_loss:,.0f} of losses, worth roughly ${total_saving:,.0f} in tax "
                f"at the report's assumed 32% short-term / 18% long-term rates. Selling "
                f"means not buying the same ticker back (in any account, IRAs included) "
                f"for 31 days; a listed peer keeps similar exposure meanwhile. Estimates "
                f"only: confirm lots and your own rates with your broker."
            ),
        )
    )
    rows = []
    for c in candidates:
        notes = []
        if c["lots"]:
            notes.append(
                "sell lots "
                + ", ".join(
                    f"{lot['date']} ({lot['units']:g} sh @ ${lot['cost_per_share']:,.2f}, "
                    f"{'LT' if lot['long_term'] else 'ST'})"
                    for lot in c["lots"][:3]
                )
                + (" …" if len(c["lots"]) > 3 else "")
            )
        if c.get("wash_sale_until"):
            notes.append(
                f"bought within 30 days: a loss sale before {c['wash_sale_until']} may be "
                f"a wash sale unless those shares are sold too"
            )
        if c.get("plan_conflict"):
            notes.append(c["plan_conflict"])
        notes.append(f"rebuy after {c['rebuy_ok_after']}")
        rows.append(
            [
                c["ticker"],
                c["account"],
                f"-${-c['loss_usd']:,.0f} ({c['loss_pct']:+.1f}%)",
                f"${-c['short_term_loss_usd']:,.0f} / ${-c['long_term_loss_usd']:,.0f}",
                f"~${c['est_tax_saving_usd']:,.0f}",
                ", ".join(c["swap_candidates"]) or "—",
                "; ".join(notes),
            ]
        )
    sections.append(
        Section(
            kind="table",
            table_header=[
                "Ticker",
                "Account",
                "Loss",
                "ST / LT loss",
                "Est. saving",
                "Swap into",
                "Notes",
            ],
            table_rows=rows,
        )
    )


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
    stop_loss_warnings: list[str] | None = None,
    csp_summary: dict[str, Any] | None = None,
    csp_warnings: list[str] | None = None,
    reinvest: dict[str, Any] | None = None,
    plan_failure: str | None = None,
) -> None:
    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Rebalance plan (action list)", level=1))

    if plan_failure:
        _append_plan_failure(sections, plan_failure, rebalance_text)
    plan = rebalance_plan if isinstance(rebalance_plan, RebalancePlan) else None
    if plan and plan.actions:
        _append_action_table(sections, plan)
    if isinstance(premortem, PreMortem) and (premortem.failures or premortem.summary):
        _append_premortem(sections, premortem)
    if plan is not None:
        _append_option_income(
            sections,
            plan,
            cash_balance=cash_balance,
            cc_round_lot_coverage=cc_round_lot_coverage,
            cc_slippage_buffer=cc_slippage_buffer,
        )
    elif cc_round_lot_coverage:
        _append_round_lot_coverage(sections, cc_round_lot_coverage)

    if cc_warnings:
        sections.append(
            Section(
                kind="para",
                text="CC plan adjustments: " + "; ".join(cc_warnings),
            )
        )

    append_csp_section(sections, csp_summary, csp_warnings)
    append_reinvest_section(sections, reinvest)

    if stop_loss_warnings:
        sections.append(
            Section(
                kind="para",
                text="Down 20%+ from cost (long-term thesis re-checked): "
                + "; ".join(stop_loss_warnings),
            )
        )

    sections.append(Section(kind="preformatted", text=rebalance_text))


def _append_plan_failure(sections: list[Section], plan_failure: str, rebalance_text: str) -> None:
    sections.append(
        Section(
            kind="para",
            text=(
                f"<b>{plan_failure}.</b> There is no action list below because the "
                "plan could not be read back, not because the rebalancer decided to "
                "hold. Whatever of it survived is printed underneath, unedited — read "
                "it as notes, not as instructions — and the run should be repeated "
                "before you act."
            ),
        )
    )
    if rebalance_text:
        sections.append(Section(kind="heading", text="Plan text as far as it got", level=2))
        sections.append(Section(kind="preformatted", text=rebalance_text))


def _append_action_table(sections: list[Section], plan: RebalancePlan) -> None:
    sections.append(
        Section(
            kind="rebalance_action_table",
            data={
                "actions": [
                    {"action": a.action, "ticker": a.ticker, "sizing": a.sizing}
                    for a in plan.actions
                ],
                "summary": plan.summary,
            },
        )
    )


def _append_premortem(sections: list[Section], premortem: PreMortem) -> None:
    sections.append(
        Section(
            kind="heading",
            text="Pre-mortem (adversarial hindsight)",
            level=2,
        )
    )
    sections.append(
        Section(
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
        )
    )


def _append_round_lot_coverage(
    sections: list[Section], cc_round_lot_coverage: dict[str, Any]
) -> None:
    from .cc_render import compute_round_lot_summary

    rls = compute_round_lot_summary(cc_round_lot_coverage)
    if rls["rows"]:
        sections.append(Section(kind="round_lot_coverage", data=rls))


def _append_option_income(
    sections: list[Section],
    plan: RebalancePlan,
    *,
    cash_balance: float | None,
    cc_round_lot_coverage: dict[str, Any] | None,
    cc_slippage_buffer: float,
) -> None:
    """Premium from the calls written, part-lot coverage, and where the
    premium (plus any stub sales) gets deployed."""
    from .cc_render import compute_premium_deployment, compute_premium_income

    if plan.option_writes:
        sections.append(
            Section(
                kind="premium_income",
                data=compute_premium_income(plan, slippage_buffer=cc_slippage_buffer),
            )
        )
    if cc_round_lot_coverage:
        _append_round_lot_coverage(sections, cc_round_lot_coverage)
    if not (
        plan.option_writes
        or any(
            a.action in ("ADD", "BUY") or (a.action == "TRIM" and "stub" in a.sizing.lower())
            for a in plan.actions
        )
    ):
        return
    stub_usd = 0.0
    if cc_round_lot_coverage:
        for a in plan.actions:
            if a.action == "TRIM" and "stub" in a.sizing.lower():
                rec = cc_round_lot_coverage.get(a.ticker)
                if rec is not None:
                    stub_usd += getattr(rec, "stub_dollar_value", 0.0)
    deployment = compute_premium_deployment(
        plan,
        cash_balance=cash_balance,
        slippage_buffer=cc_slippage_buffer,
        stub_consolidation_usd=stub_usd,
    )
    if deployment["gross_premium_usd"] > 0 or deployment["deployments"] or stub_usd > 0:
        sections.append(Section(kind="premium_deployment", data=deployment))


def append_reinvest_section(sections: list[Section], reinvest: dict[str, Any] | None) -> None:
    """When the plan sells but deploys nothing, name where the money could
    go: `reinvest` = {"sold": [tickers], "ideas": [discover/reinvest.py ideas]}."""
    if not reinvest or not reinvest.get("sold") or not reinvest.get("ideas"):
        return
    sections.append(Section(kind="heading", text="Where the sale proceeds could go", level=2))
    sections.append(
        Section(
            kind="para",
            text=(
                f"The plan sells {', '.join(reinvest['sold'])} but names no BUY or ADD. "
                "These are the most recent discover picks you don't hold, outside any "
                "over-cap sector — long-term candidates for the proceeds."
            ),
        )
    )
    sections.append(
        Section(
            kind="table",
            table_header=["Ticker", "Pick", "Picked on", "Sector", "Conviction"],
            table_rows=[
                [
                    i["ticker"],
                    f"#{i['rank']}",
                    i["pick_date"],
                    i.get("sector") or "—",
                    str(i["conviction"]) if i.get("conviction") is not None else "—",
                ]
                for i in reinvest["ideas"]
            ],
        )
    )


def append_csp_section(
    sections: list[Section],
    summary: dict[str, Any] | None,
    warnings: list[str] | None,
) -> None:
    """Cash-secured puts the plan sells: what each pays, the cash it ties
    up, and what the shares would cost if assigned."""
    if summary and summary.get("rows"):
        sections.append(Section(kind="heading", text="Cash-secured puts", level=2))
        rows = [
            [
                r["ticker"],
                r.get("account") or "—",
                f"{r['contracts']} × ${r['strike']:,.2f}P",
                r["expiry"],
                f"{r['delta']:.2f}",
                f"${r['premium_usd']:,.0f}",
                "—" if r["annualized_yield_pct"] is None else f"{r['annualized_yield_pct']:.1f}%",
                f"${r['cash_reserved']:,.0f}",
                f"${r['net_cost_if_assigned']:,.2f}",
            ]
            for r in summary["rows"]
        ]
        sections.append(
            Section(
                kind="table",
                table_header=[
                    "Ticker",
                    "Account",
                    "Put",
                    "Expiry",
                    "Delta",
                    "Premium",
                    "Yield (ann.)",
                    "Cash reserved",
                    "Cost if assigned",
                ],
                table_rows=rows,
            )
        )
        n = len(summary["rows"])
        pct = summary.get("pct_of_budget")
        share = f" ({pct:.1f}% of the ${summary['cash_budget']:,.0f} put budget)" if pct else ""
        sections.append(
            Section(
                kind="para",
                text=(
                    f"Puts reserve ${summary['total_cash_reserved']:,.0f} of cash across "
                    f"{n} ticker{'s' if n != 1 else ''}{share} for "
                    f"${summary['total_premium_usd']:,.0f} of premium. That cash stays "
                    "in the account until expiry; if a put is assigned you buy 100 "
                    "shares per contract at the strike."
                ),
            )
        )
    if warnings:
        sections.append(Section(kind="para", text="Put plan adjustments: " + "; ".join(warnings)))


def append_holding_review_sections(
    sections: list[Section],
    holdings_reviews: dict[str, Any],
) -> None:
    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Per-holding reviews", level=1))
    from ..models.llm import HoldingReview
    from .catalysts import catalysts_to_dicts

    for ticker in sorted(holdings_reviews.keys()):
        review = holdings_reviews[ticker]
        if isinstance(review, HoldingReview):
            sections.append(
                Section(
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
                        "catalysts": catalysts_to_dicts(review.upcoming_catalysts),
                    },
                )
            )
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
    ranker_output: object = None,
    redteam_output: object = None,
    sizer_output: object = None,
) -> None:
    # The structured objects have to travel with the text. Without them
    # `build_sections` falls back to re-parsing prose: "At a glance" reads
    # the objects directly and renders an em dash in every column, and the
    # per-pick bear case comes back "(missing)" even though the critique
    # ran and is sitting in the database.
    discover_sections = build_sections(
        ranker_text=ranker_text,
        redteam_text=redteam_text,
        sizer_text=sizer_text,
        candidates=candidates,
        universe_size=len(candidates),
        holdings_summary="",
        macro_summary="",
        sector_rotation=sector_rotation,
        ranker_output=ranker_output,
        redteam_output=redteam_output,
        sizer_output=sizer_output,
    )
    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Discover picks (input to rebalancer)", level=1))
    sections.extend(discover_sections[2:])


def _plan_status(
    plan_failure: str | None, rebalance_plan: object, rebalance_text: str
) -> tuple[str, str]:
    """(status, banner text) for the top of the report."""
    if plan_failure:
        # A plan that never arrived is not a plan that said "do nothing".
        # The banner has to carry that, because every other part of this
        # report looks identical in both cases.
        return "FAILED", "STATUS: PLAN INCOMPLETE — DO NOT READ AS 'NO TRADES'"
    status = parse_rebalance_status(rebalance_plan or rebalance_text)
    status_label = (
        "STATUS: NO ACTION RECOMMENDED"
        if status == "NO_ACTION"
        else "STATUS: ACTION RECOMMENDED"
        if status == "ACTION"
        else "STATUS: REVIEW REQUIRED"
    )
    return status, status_label


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
    track_record: TrackRecord | None = None,
    thesis_checks: list[dict[str, Any]] | None = None,
    harvest_candidates: list[dict[str, Any]] | None = None,
    rebalance_plan: object = None,
    market_themes: object = None,
    premortem: object = None,
    holdings_news: dict[str, list[dict[str, Any]]] | None = None,
    cc_eligibility: dict[str, Any] | None = None,
    cc_round_lot_coverage: dict[str, Any] | None = None,
    cc_stub_pool_total_usd: float = 0.0,
    cc_warnings: list[str] | None = None,
    cc_slippage_buffer: float = 0.10,
    stop_loss_warnings: list[str] | None = None,
    stale_accounts: list[str] | None = None,
    csp_summary: dict[str, Any] | None = None,
    csp_warnings: list[str] | None = None,
    reinvest: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    plan_failure: str | None = None,
    ranker_output: object = None,
    redteam_output: object = None,
    sizer_output: object = None,
) -> list[Section]:
    """Rebalance-specific layout — status banner + metrics + dashboard +
    sector pie at the top, then the LLM's plan + per-holding reviews +
    discover-picks appendix."""
    del cc_eligibility, cc_stub_pool_total_usd  # reserved for future section use

    today = date.today().isoformat()
    status, status_label = _plan_status(plan_failure, rebalance_plan, rebalance_text)

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
        track_record=track_record,
        thesis_checks=thesis_checks,
        market_themes=market_themes,
        rebalance_plan=rebalance_plan,
        harvest_candidates=harvest_candidates,
        stop_loss_warnings=stop_loss_warnings,
        stale_accounts=stale_accounts,
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
        stop_loss_warnings=stop_loss_warnings,
        csp_summary=csp_summary,
        csp_warnings=csp_warnings,
        reinvest=reinvest,
        plan_failure=plan_failure,
    )
    append_harvest_section(sections, harvest_candidates)
    append_holding_review_sections(sections, holdings_reviews)
    append_discover_appendix(
        sections,
        ranker_text=ranker_text,
        redteam_text=redteam_text,
        sizer_text=sizer_text,
        candidates=candidates,
        sector_rotation=sector_rotation,
        ranker_output=ranker_output,
        redteam_output=redteam_output,
        sizer_output=sizer_output,
    )
    append_usage_section(sections, usage)
    return sections
