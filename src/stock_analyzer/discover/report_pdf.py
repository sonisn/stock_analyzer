"""PDF renderer for the report Section IR.

Generates the email attachment via ReportLab. Reads Section objects
produced by `report_sections.build_sections` and emits a paginated
PDF that mirrors the HTML email layout.

Stays in sync with `report_html.py` because both pull palettes from
`report_sections` — same colors, same fragility/verdict tiers, same
pre-mortem verdict banner. If you want the two renderers to render
"the same thing", they read from the same shared constants.
"""
from __future__ import annotations

import html
from io import BytesIO
from typing import Any

from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .report_sections import (
    _FRAGILITY_COLORS,
    _LIKELIHOOD_COLOR,
    _PIE_PALETTE,
    _SEVERITY_COLOR,
    _STATUS_COLORS,
    _TREND_GLYPHS,
    _VERDICT_COLORS,
    _VERDICT_PALETTE_PREMORTEM,
    Section,
    _conviction_swatch,
    _theme_strength_color,
)


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles["Heading1"].fontSize = 20
    styles["Heading1"].textColor = colors.HexColor("#111827")
    styles["Heading2"].fontSize = 14
    styles["Heading2"].textColor = colors.HexColor("#111827")
    styles["Heading3"].fontSize = 12
    styles["Heading3"].textColor = colors.HexColor("#374151")
    styles["BodyText"].textColor = colors.HexColor("#1f2937")
    styles["Code"].fontSize = 9
    styles["Code"].backColor = colors.HexColor("#f3f4f6")
    styles["Code"].borderColor = colors.HexColor("#e5e7eb")
    styles["Code"].borderWidth = 0.5
    styles["Code"].borderPadding = 6
    styles.add(ParagraphStyle(
        name="Quote",
        parent=styles["BodyText"],
        backColor=colors.HexColor("#eff6ff"),
        borderColor=colors.HexColor("#3b82f6"),
        borderWidth=1,
        borderPadding=8,
        leftIndent=10,
    ))
    styles.add(ParagraphStyle(
        name="Banner",
        parent=styles["BodyText"],
        fontSize=14,
        fontName="Helvetica-Bold",
        borderPadding=12,
        borderWidth=1.5,
        leftIndent=8,
    ))
    return styles


def _pdf_status_banner(status: str, text: str, styles):
    """A status banner rendered as a single-row colored Table for PDF."""
    cs = _STATUS_COLORS.get(status, _STATUS_COLORS["UNKNOWN"])
    para = Paragraph(html.escape(text), styles["Banner"])
    t = Table([[para]], colWidths=[6.7 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(cs["bg"])),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(cs["fg"])),
        ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor(cs["border"])),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    return t


def _pdf_metric_strip(metrics: list[tuple[str, str]], styles):
    """Horizontal strip of metric cards rendered as a 1-row Table."""
    if not metrics:
        return None
    cell_paras = []
    for label, value in metrics:
        para = Paragraph(
            f"<font color='#6b7280' size='8'>{html.escape(label.upper())}</font><br/>"
            f"<font color='#111827' size='14'><b>{html.escape(value)}</b></font>",
            styles["BodyText"],
        )
        cell_paras.append(para)
    t = Table([cell_paras], colWidths=[6.7 / len(metrics) * inch] * len(metrics))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def _pdf_holdings_dashboard(holdings: list[dict[str, Any]]):
    """Dashboard table with colored verdict badges + P/L styling."""
    header = ["Ticker", "Verdict", "Conf", "P/L", "Sector"]
    rows: list[list[Any]] = [header]
    for h in holdings:
        verdict = (h.get("verdict") or "HOLD").upper()
        conf = h.get("confidence")
        pnl = h.get("pnl_pct")
        pnl_str = f"{pnl:+.1f}%" if isinstance(pnl, (int, float)) else "—"
        conf_str = f"{conf}/10" if conf is not None else "—"
        rows.append([
            h.get("ticker", ""),
            verdict,
            conf_str,
            pnl_str,
            h.get("sector") or "—",
        ])
    table = Table(rows, repeatRows=1, hAlign="LEFT",
                  colWidths=[1.0 * inch, 0.9 * inch, 0.7 * inch, 0.9 * inch, 1.8 * inch])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    # Per-row coloring on verdict + P/L cells.
    for i, h in enumerate(holdings, start=1):
        verdict = (h.get("verdict") or "HOLD").upper()
        vc = _VERDICT_COLORS.get(verdict, _VERDICT_COLORS["HOLD"])
        style_cmds.append(("BACKGROUND", (1, i), (1, i), colors.HexColor(vc["bg"])))
        style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor(vc["fg"])))
        style_cmds.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
        pnl = h.get("pnl_pct")
        if isinstance(pnl, (int, float)):
            pl_color = colors.HexColor("#16a34a" if pnl >= 0 else "#dc2626")
            style_cmds.append(("TEXTCOLOR", (3, i), (3, i), pl_color))
            style_cmds.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
    table.setStyle(TableStyle(style_cmds))
    return table


def _pdf_sector_pie(pie_data: list[tuple[str, float]]):
    """ReportLab donut pie + side legend, rendered as a Drawing flowable."""
    if not pie_data:
        return None
    drawing = Drawing(440, 200)
    pie = Pie()
    pie.x = 30
    pie.y = 25
    pie.width = 150
    pie.height = 150
    pie.data = [v for _, v in pie_data]
    pie.labels = None
    pie.slices.strokeWidth = 1
    pie.slices.strokeColor = colors.white
    pie.innerRadiusFraction = 0.55
    for i, _ in enumerate(pie_data):
        pie.slices[i].fillColor = colors.HexColor(_PIE_PALETTE[i % len(_PIE_PALETTE)])
    drawing.add(pie)

    legend = Legend()
    legend.x = 210
    legend.y = 160
    legend.alignment = "right"
    legend.fontSize = 9
    legend.fontName = "Helvetica"
    total = sum(v for _, v in pie_data if v > 0) or 1
    legend.colorNamePairs = [
        (
            colors.HexColor(_PIE_PALETTE[i % len(_PIE_PALETTE)]),
            f"{label}  {(value / total) * 100:.1f}%",
        )
        for i, (label, value) in enumerate(pie_data)
        if value > 0
    ]
    legend.columnMaximum = 12
    legend.dy = 12
    legend.deltay = 4
    drawing.add(legend)
    return drawing


def _pdf_pill(text: str, fg: str, bg: str, styles) -> Paragraph:
    """Single colored pill, rendered as a Paragraph for compositing in a Table cell."""
    return Paragraph(
        f"<font color='{fg}' size='9'><b>{html.escape(text)}</b></font>",
        ParagraphStyle(
            name="pill",
            parent=styles["BodyText"],
            backColor=colors.HexColor(bg),
            borderPadding=(2, 6, 2, 6),
            alignment=1,  # center
            spaceBefore=0, spaceAfter=0,
        ),
    )


def _pdf_pick_card(d: dict[str, Any], styles) -> list[Any]:
    """Render a structured pick as a header row of colored pill badges
    + per-section paragraphs. Returns a list of flowables (no Spacer
    around the page-break boundary)."""
    ticker = str(d.get("ticker", ""))
    rank = d.get("rank")
    conviction = d.get("conviction") if isinstance(d.get("conviction"), int) else None
    fragility = d.get("fragility_rank") if isinstance(d.get("fragility_rank"), int) else None
    alloc_pct = d.get("allocation_pct")
    alloc_usd = d.get("allocation_usd")

    flow: list[Any] = []

    # Header row: ticker + pill badges as a single 5-column Table.
    pill_cells: list[Paragraph] = []
    pill_cells.append(Paragraph(
        f"<font size='14'><b>{html.escape(ticker)}</b></font>",
        styles["BodyText"],
    ))
    if rank is not None:
        pill_cells.append(_pdf_pill(f"#{rank}", "#fff", "#1f2937", styles))
    if conviction is not None:
        pill_cells.append(_pdf_pill(
            f"Conviction {conviction}/10", "#fff",
            _conviction_swatch(conviction), styles,
        ))
    if fragility in _FRAGILITY_COLORS:
        c = _FRAGILITY_COLORS[fragility]
        pill_cells.append(_pdf_pill(
            f"Fragility {fragility}/5", c["fg"], c["bg"], styles,
        ))
    if alloc_pct is not None:
        pill_cells.append(_pdf_pill(
            f"Allocation {alloc_pct:.1f}%", "#4c1d95", "#ece8fb", styles,
        ))
    elif alloc_usd is not None:
        pill_cells.append(_pdf_pill(
            f"Allocation ${alloc_usd:,.0f}", "#4c1d95", "#ece8fb", styles,
        ))

    # Pad to 5 cells so all rows column-align.
    while len(pill_cells) < 5:
        pill_cells.append(Paragraph("", styles["BodyText"]))
    header = Table(
        [pill_cells],
        colWidths=[2.1 * inch] + [1.1 * inch] * 4,
        hAlign="LEFT",
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(header)

    one_liner = str(d.get("one_liner") or "").strip()
    if one_liner:
        flow.append(Paragraph(html.escape(one_liner), styles["BodyText"]))
        flow.append(Spacer(1, 6))

    def _section(label: str, body: str | None, color: str = "#374151") -> None:
        if not body:
            return
        flow.append(Paragraph(
            f"<font color='{color}' size='8'><b>{label.upper()}</b></font>",
            styles["BodyText"],
        ))
        flow.append(Paragraph(html.escape(str(body)), styles["BodyText"]))
        flow.append(Spacer(1, 4))

    _section("Bull thesis", d.get("bull_thesis"))
    if d.get("what_youre_betting_on"):
        flow.append(Paragraph(
            f"<i>You're betting on: {html.escape(str(d.get('what_youre_betting_on')))}</i>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 4))
    _section("Bear case", d.get("bear_case"), color="#9c1010")
    if d.get("most_fragile_assumption"):
        flow.append(Paragraph(
            f"<b>Most fragile assumption:</b> "
            f"{html.escape(str(d.get('most_fragile_assumption')))}",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 2))
    if d.get("watch_metric"):
        flow.append(Paragraph(
            f"<b>Watch:</b> {html.escape(str(d.get('watch_metric')))}",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 4))
    _section("Why this over alternatives", d.get("why_over_alternatives"))
    if d.get("sector_concentration_check"):
        flow.append(Paragraph(
            f"<font size='9' color='#6b7280'>Sector concentration: "
            f"{html.escape(str(d.get('sector_concentration_check')))}</font>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 4))
    if d.get("allocation_rationale"):
        rationale_para = Paragraph(
            f"<b>Sizing rationale:</b> "
            f"{html.escape(str(d.get('allocation_rationale')))}",
            styles["BodyText"],
        )
        wrap = Table([[rationale_para]], colWidths=[6.7 * inch])
        wrap.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#7c3aed")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        flow.append(wrap)
        flow.append(Spacer(1, 8))
    return flow


def _pdf_allocation_table(d: dict[str, Any], styles) -> list[Any]:
    """Structured sizing table rendered as a proper ReportLab Table."""
    allocations = d.get("allocations") or []
    warnings = d.get("warnings") or []
    if not allocations:
        return [Paragraph("(no allocations)", styles["BodyText"]), Spacer(1, 4)]
    rows: list[list[Any]] = [
        [
            Paragraph("<b>Ticker</b>", styles["BodyText"]),
            Paragraph("<b>Size</b>", styles["BodyText"]),
            Paragraph("<b>Rationale</b>", styles["BodyText"]),
        ]
    ]
    for a in allocations:
        pct = a.get("pct")
        usd = a.get("usd")
        size_str = (
            f"{pct:.1f}%" if pct is not None
            else (f"${usd:,.0f}" if usd is not None else "—")
        )
        rows.append([
            Paragraph(f"<b>{html.escape(str(a.get('ticker', '')))}</b>", styles["BodyText"]),
            Paragraph(
                f"<font color='#4c1d95'><b>{html.escape(size_str)}</b></font>",
                styles["BodyText"],
            ),
            Paragraph(html.escape(str(a.get("rationale") or "")), styles["BodyText"]),
        ])
    t = Table(
        rows,
        repeatRows=1,
        hAlign="LEFT",
        colWidths=[0.9 * inch, 1.0 * inch, 4.8 * inch],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow: list[Any] = [t, Spacer(1, 6)]
    if warnings:
        body = "<b>Concentration warnings:</b> " + " · ".join(
            html.escape(str(w)) for w in warnings
        )
        wpara = Paragraph(body, styles["BodyText"])
        wrap = Table([[wpara]], colWidths=[6.7 * inch])
        wrap.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4e0")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#e89c00")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#8a4a00")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        flow.append(wrap)
        flow.append(Spacer(1, 6))
    return flow


def _pdf_market_themes_panel(d: dict[str, Any], styles) -> list[Any]:
    """PDF: per-theme bordered block with strength pill + trend arrow +
    description + member tickers."""
    themes = d.get("themes") or []
    if not themes:
        return []
    flow: list[Any] = []
    for theme in themes:
        name = str(theme.get("name") or "")
        description = str(theme.get("description") or "")
        strength = theme.get("strength")
        trending = str(theme.get("trending") or "flat")
        members = theme.get("member_tickers") or []
        glyph, glyph_color = _TREND_GLYPHS.get(trending, _TREND_GLYPHS["flat"])
        strength_color = _theme_strength_color(
            strength if isinstance(strength, int) else None
        )
        strength_text = f"{strength}/10" if strength is not None else ""

        header_cells: list[Paragraph] = [
            Paragraph(
                f"<b>{html.escape(name)}</b>",
                styles["BodyText"],
            ),
            _pdf_pill(strength_text, "#fff", strength_color, styles),
            Paragraph(
                f"<font color='{glyph_color}' size='12'><b>{glyph}</b></font>",
                styles["BodyText"],
            ),
        ]
        while len(header_cells) < 3:
            header_cells.append(Paragraph("", styles["BodyText"]))
        header = Table(
            [header_cells],
            colWidths=[3.5 * inch, 1.0 * inch, 0.5 * inch],
            hAlign="LEFT",
        )
        header.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        flow.append(header)
        flow.append(Paragraph(
            f"<font size='9' color='#374151'>{html.escape(description)}</font>",
            styles["BodyText"],
        ))
        member_strip = ", ".join(str(t) for t in members[:18])
        suffix = f" +{len(members)-18} more" if len(members) > 18 else ""
        flow.append(Paragraph(
            f"<font size='8' color='#6b7280'><b>Members:</b> "
            f"{html.escape(member_strip + suffix)}</font>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 8))
    return flow


def _pdf_premortem_panel(d: dict[str, Any], styles) -> list[Any]:
    """PDF counterpart of `_premortem_panel_html` — verdict banner, summary,
    then per-failure cards with likelihood/severity pills and warning callout."""
    verdict = str(d.get("overall_verdict") or "proceed_with_caveat")
    summary = str(d.get("summary") or "")
    failures = d.get("failures") or []
    if not failures and not summary:
        return []
    flow: list[Any] = []
    vp = _VERDICT_PALETTE_PREMORTEM.get(verdict, _VERDICT_PALETTE_PREMORTEM["proceed_with_caveat"])
    verdict_label = {
        "proceed_as_planned":  "PROCEED AS PLANNED",
        "proceed_with_caveat": "PROCEED WITH CAVEAT",
        "reconsider":          "RECONSIDER",
    }.get(verdict, verdict.upper())

    banner = Table(
        [[Paragraph(
            f"<font color='{vp['fg']}' size='11'><b>Verdict: "
            f"{html.escape(verdict_label)}</b></font>",
            styles["BodyText"],
        )]],
        colWidths=[6.5 * inch],
        hAlign="LEFT",
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(vp["bg"])),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(vp["border"])),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    flow.append(banner)
    flow.append(Spacer(1, 4))
    if summary:
        flow.append(Paragraph(
            f"<font size='9' color='#374151'>{html.escape(summary)}</font>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 6))

    for f in failures:
        likelihood = str(f.get("likelihood") or "medium").lower()
        severity = str(f.get("severity") or "moderate").lower()
        trig = str(f.get("triggering_action") or "")
        narrative = str(f.get("failure_narrative") or "")
        warning = str(f.get("early_warning") or "")
        like_color = _LIKELIHOOD_COLOR.get(likelihood, "#6b7280")
        sev_color = _SEVERITY_COLOR.get(severity, "#6b7280")

        pill_row = Table(
            [[
                _pdf_pill(f"Likelihood: {likelihood}", "#fff", like_color, styles),
                _pdf_pill(f"Severity: {severity}", "#fff", sev_color, styles),
                Paragraph("", styles["BodyText"]),
            ]],
            colWidths=[1.5 * inch, 1.5 * inch, 3.5 * inch],
            hAlign="LEFT",
        )
        pill_row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        flow.append(pill_row)
        flow.append(Paragraph(
            "<font size='8' color='#6b7280'><b>TRIGGERING ACTION</b></font>",
            styles["BodyText"],
        ))
        flow.append(Paragraph(
            f"<font size='9' color='#1f2937'><i>{html.escape(trig)}</i></font>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 3))
        flow.append(Paragraph(
            f"<font size='9' color='#1f2937'>{html.escape(narrative)}</font>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 3))
        warn_table = Table(
            [[Paragraph(
                f"<font size='8' color='#8a4a00'><b>Early warning:</b> "
                f"{html.escape(warning)}</font>",
                styles["BodyText"],
            )]],
            colWidths=[6.5 * inch],
            hAlign="LEFT",
        )
        warn_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4e0")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#e89c00")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.append(warn_table)
        flow.append(Spacer(1, 8))
    return flow


def _pdf_holding_review_card(d: dict[str, Any], styles) -> list[Any]:
    """PDF counterpart of `_holding_review_card_html` — ticker header
    with verdict + confidence pills, then labeled sections."""
    ticker = str(d.get("ticker", ""))
    verdict = str(d.get("verdict") or "HOLD").upper()
    confidence = d.get("confidence") if isinstance(d.get("confidence"), int) else None
    trim_pct = d.get("trim_pct")
    position_context = str(d.get("position_context") or "")
    forward_outlook = str(d.get("forward_outlook") or "")
    reasoning = str(d.get("reasoning") or "")
    tax_lot_plan = d.get("tax_lot_plan") or []
    what_change = str(d.get("what_would_change_mind") or "")
    wash_sale_notice = d.get("wash_sale_notice")

    flow: list[Any] = []

    # Header row: ticker + pills as a 4-column Table for alignment.
    vc = _VERDICT_COLORS.get(verdict) or _VERDICT_COLORS["HOLD"]
    cells: list[Paragraph] = [
        Paragraph(
            f"<font size='14'><b>{html.escape(ticker)}</b></font>",
            styles["BodyText"],
        ),
        _pdf_pill(verdict, vc["fg"], vc["bg"], styles),
    ]
    if confidence is not None:
        cs = _conviction_swatch(confidence)
        cells.append(_pdf_pill(f"Conviction {confidence}/10", "#fff", cs, styles))
    if verdict == "TRIM" and isinstance(trim_pct, (int, float)) and trim_pct > 0:
        cells.append(_pdf_pill(f"Trim {trim_pct:.0f}%", "#a36500", "#fff4e0", styles))
    while len(cells) < 4:
        cells.append(Paragraph("", styles["BodyText"]))
    header = Table(
        [cells],
        colWidths=[1.8 * inch, 1.0 * inch, 1.5 * inch, 1.5 * inch],
        hAlign="LEFT",
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(header)

    if position_context:
        flow.append(Paragraph(
            f"<font color='#6b7280' size='9'>"
            f"{html.escape(position_context)}</font>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 6))

    def _section(label: str, body: str, *, color: str = "#374151") -> None:
        if not body:
            return
        flow.append(Paragraph(
            f"<font color='{color}' size='8'><b>{html.escape(label.upper())}</b></font>",
            styles["BodyText"],
        ))
        flow.append(Paragraph(html.escape(body), styles["BodyText"]))
        flow.append(Spacer(1, 6))

    _section("Forward outlook", forward_outlook)
    _section("Reasoning", reasoning)

    if tax_lot_plan:
        flow.append(Paragraph(
            "<font color='#4c1d95' size='8'><b>TAX LOT PLAN</b></font>",
            styles["BodyText"],
        ))
        for line in tax_lot_plan:
            flow.append(Paragraph(
                f"• {html.escape(str(line))}", styles["BodyText"],
            ))
        flow.append(Spacer(1, 6))

    if wash_sale_notice:
        notice_para = Paragraph(
            f"<b>Wash-sale notice:</b> {html.escape(str(wash_sale_notice))}",
            styles["BodyText"],
        )
        wrap = Table([[notice_para]], colWidths=[6.7 * inch])
        wrap.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fde4e4")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#d73030")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#9c1010")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        flow.append(wrap)
        flow.append(Spacer(1, 6))

    if what_change:
        flow.append(Paragraph(
            f"<i><font color='#6b7280' size='9'>"
            f"<b>What would change my mind:</b> "
            f"{html.escape(what_change)}</font></i>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 10))

    return flow


_CC_HEADER_BG = colors.HexColor("#0d9488")  # teal for Premium Income
_CC_RLC_BG = colors.HexColor("#a16207")     # amber for Round-Lot Coverage
_CC_GRID = colors.HexColor("#d1d5db")


def _pdf_premium_income(data: dict, styles) -> list:
    """Render the Premium Income section as a ReportLab table + caption."""
    flow: list = []
    flow.append(Paragraph("<b>Premium Income</b>", styles["Heading3"]))
    header = ["Ticker", "Strike", "Expiry", "Qty", "Premium", "Δ", "Assign %"]
    rows = [header]
    for r in data.get("rows") or []:
        rows.append([
            r["ticker"], f"${r['strike']:,.2f}", r["expiry"],
            str(r["contracts"]), f"${r['premium_usd']:,.0f}",
            f"{r['delta']:.2f}", f"{r['assignment_pct']}%",
        ])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _CC_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, _CC_GRID),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    flow.append(t)
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        f"Gross premium: <b>${data.get('gross_premium_usd', 0):,.0f}</b> &nbsp;"
        f"Buffer (10%): -${data.get('slippage_buffer_usd', 0):,.0f} &nbsp;"
        f"Deployable: <b>${data.get('deployable_premium_usd', 0):,.0f}</b>",
        styles["BodyText"],
    ))
    flow.append(Spacer(1, 12))
    return flow


def _pdf_round_lot_coverage(data: dict, styles) -> list:
    """Render the Round-Lot Coverage section as a table."""
    rows = data.get("rows") or []
    if not rows:
        return []
    flow: list = [Paragraph("<b>Round-Lot Coverage</b>", styles["Heading3"])]
    table_rows = [["Position", "Shares", "Round Lots", "Stub", "Stub $", "To-next-lot"]]
    for r in rows:
        table_rows.append([
            r["ticker"], str(r["shares"]),
            f"{r['round_lots']} ({r['round_lot_shares']})",
            str(r["stub_shares"]),
            f"${r['stub_dollar_value']:,.0f}",
            f"${r['to_next_lot_cost']:,.0f}",
        ])
    t = Table(table_rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _CC_RLC_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, _CC_GRID),
    ]))
    flow.append(t)
    flow.append(Paragraph(
        f"Stub pool total: <b>${data.get('stub_pool_total_usd', 0):,.0f}</b>",
        styles["BodyText"],
    ))
    flow.append(Spacer(1, 12))
    return flow


def _pdf_premium_deployment(data: dict, styles) -> list:
    """Render the Premium → Deployment dry-powder box."""
    flow: list = [Paragraph("<b>Premium &rarr; Deployment</b>", styles["Heading3"])]
    lines = [
        f"Deployable premium: ${data.get('deployable_premium_usd', 0):,.0f}",
        f"Existing cash: ${data.get('existing_cash_usd', 0):,.0f}",
    ]
    if data.get("stub_consolidation_usd"):
        lines.append(
            f"Stub consolidation: ${data['stub_consolidation_usd']:,.0f}"
        )
    lines.append(
        f"<b>Total dry powder: ${data.get('total_dry_powder_usd', 0):,.0f}</b>"
    )
    flow.append(Paragraph("<br/>".join(lines), styles["BodyText"]))
    deps = data.get("deployments") or []
    if deps:
        body = "<br/>".join(
            f"&rarr; {d['action']} <b>{d['ticker']}</b> {d['sizing']}"
            for d in deps
        )
        flow.append(Paragraph(body, styles["BodyText"]))
    flow.append(Spacer(1, 12))
    return flow


def _pdf_rebalance_action_table(d: dict[str, Any], styles) -> list[Any]:
    """Per-action table for the rebalance plan section. Each row gets a
    pale-tinted action cell using the SELL/TRIM/ADD/BUY palette."""
    actions = d.get("actions") or []
    summary = d.get("summary") or ""
    if not actions:
        return []
    flow: list[Any] = []
    if summary:
        flow.append(Paragraph(
            f"<i><font color='#6b7280'>{html.escape(str(summary))}</font></i>",
            styles["BodyText"],
        ))
        flow.append(Spacer(1, 4))
    rows: list[list[Any]] = [
        [
            Paragraph("<b>Action</b>", styles["BodyText"]),
            Paragraph("<b>Ticker</b>", styles["BodyText"]),
            Paragraph("<b>Sizing</b>", styles["BodyText"]),
        ]
    ]
    style_cmds: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i, a in enumerate(actions, start=1):
        action_type = str(a.get("action") or "")
        ticker = str(a.get("ticker") or "")
        sizing = str(a.get("sizing") or "")
        c = _VERDICT_COLORS.get(action_type) or _VERDICT_COLORS["HOLD"]
        rows.append([
            Paragraph(
                f"<font color='{c['fg']}'><b>{html.escape(action_type)}</b></font>",
                styles["BodyText"],
            ),
            Paragraph(f"<b>{html.escape(ticker)}</b>", styles["BodyText"]),
            Paragraph(html.escape(sizing), styles["BodyText"]),
        ])
        # Tint the Action column with the badge background so the row reads
        # at a glance the same way the HTML pill does.
        style_cmds.append(
            ("BACKGROUND", (0, i), (0, i), colors.HexColor(c["bg"]))
        )
    t = Table(
        rows,
        repeatRows=1,
        hAlign="LEFT",
        colWidths=[0.9 * inch, 0.9 * inch, 4.9 * inch],
    )
    t.setStyle(TableStyle(style_cmds))
    flow.append(t)
    flow.append(Spacer(1, 8))
    return flow


def render_pdf(sections: list[Section], chart_bytes: dict[str, bytes]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Stock Discovery",
        author="stock-analyzer",
    )
    styles = _pdf_styles()
    flow: list[Any] = []

    for s in sections:
        if s.kind == "heading":
            flow.append(Paragraph(html.escape(s.text), styles[f"Heading{min(s.level, 3)}"]))
            flow.append(Spacer(1, 4))
        elif s.kind == "para":
            flow.append(Paragraph(html.escape(s.text), styles["BodyText"]))
            flow.append(Spacer(1, 4))
        elif s.kind == "preformatted":
            flow.append(Preformatted(s.text, styles["Code"]))
            flow.append(Spacer(1, 4))
        elif s.kind == "blockquote":
            flow.append(Paragraph(html.escape(s.text), styles["Quote"]))
            flow.append(Spacer(1, 6))
        elif s.kind == "image" and s.image_ticker:
            data = chart_bytes.get(s.image_ticker)
            if data:
                try:
                    img = Image(BytesIO(data), width=6.5 * inch, height=3.5 * inch)
                    flow.append(img)
                    flow.append(Spacer(1, 4))
                except Exception:
                    pass
        elif s.kind == "table" and s.table_header and s.table_rows:
            tdata = [s.table_header] + s.table_rows
            t = Table(tdata, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 4))
        elif s.kind == "status_banner":
            flow.append(_pdf_status_banner(s.status, s.text, styles))
            flow.append(Spacer(1, 6))

        elif s.kind == "metric_strip":
            strip = _pdf_metric_strip(s.metrics or [], styles)
            if strip is not None:
                flow.append(strip)
                flow.append(Spacer(1, 6))

        elif s.kind == "holdings_dashboard" and s.holdings:
            flow.append(_pdf_holdings_dashboard(s.holdings))
            flow.append(Spacer(1, 6))

        elif s.kind == "sector_pie" and s.pie_data:
            pie = _pdf_sector_pie(s.pie_data)
            if pie is not None:
                flow.append(pie)
                flow.append(Spacer(1, 6))

        elif s.kind == "pick_card" and s.data:
            for el in _pdf_pick_card(s.data, styles):
                flow.append(el)

        elif s.kind == "allocation_table" and s.data:
            for el in _pdf_allocation_table(s.data, styles):
                flow.append(el)

        elif s.kind == "rebalance_action_table" and s.data:
            for el in _pdf_rebalance_action_table(s.data, styles):
                flow.append(el)

        elif s.kind == "holding_review_card" and s.data:
            for el in _pdf_holding_review_card(s.data, styles):
                flow.append(el)

        elif s.kind == "market_themes_panel" and s.data:
            for el in _pdf_market_themes_panel(s.data, styles):
                flow.append(el)

        elif s.kind == "premortem_panel" and s.data:
            for el in _pdf_premortem_panel(s.data, styles):
                flow.append(el)

        elif s.kind == "premium_income" and s.data:
            for el in _pdf_premium_income(s.data, styles):
                flow.append(el)

        elif s.kind == "round_lot_coverage" and s.data:
            for el in _pdf_round_lot_coverage(s.data, styles):
                flow.append(el)

        elif s.kind == "premium_deployment" and s.data:
            for el in _pdf_premium_deployment(s.data, styles):
                flow.append(el)

        elif s.kind == "page_break":
            flow.append(PageBreak())

    doc.build(flow)
    return buf.getvalue()
