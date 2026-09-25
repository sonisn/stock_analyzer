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
from collections.abc import Callable
from io import BytesIO
from typing import Any

from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
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

from ..logging import get_logger
from ..models.reports import Section
from .report_html import (
    _GRID,
    _INK,
    _INK_MUTED,
    _REFERENCE_LINE,
    _SERIES_1,
    _nudge_apart,
    _short_date,
    ledger_chart_model,
    pct_tick,
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
    _conviction_swatch,
    _theme_strength_color,
)

logger = get_logger(__name__)

_PDF_FONT = "Helvetica"


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
    styles.add(
        ParagraphStyle(
            name="Quote",
            parent=styles["BodyText"],
            backColor=colors.HexColor("#eff6ff"),
            borderColor=colors.HexColor("#3b82f6"),
            borderWidth=1,
            borderPadding=8,
            leftIndent=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Banner",
            parent=styles["BodyText"],
            fontSize=14,
            fontName="Helvetica-Bold",
            borderPadding=12,
            borderWidth=1.5,
            leftIndent=8,
        )
    )
    return styles


_TABLE_WIDTH = 7.3 * inch  # letter width minus the 0.6in side margins
_CELL_PAD = 12  # Table's default 6pt left + right padding


def _fit_table(
    header: list[str], rows: list[list[str]], styles
) -> tuple[list[list[Any]], list[float] | None]:
    """Plain table cells never wrap in ReportLab, so a long cell (a headline,
    a list of signals) ran off the page. When the natural widths overflow,
    narrow columns keep their width, the rest share what is left, and cells
    in the shrunk columns become wrapping Paragraphs."""
    n = len(header)
    natural = [
        max(
            stringWidth(str(header[i]), "Helvetica-Bold", 9),
            *(stringWidth(str(r[i]), _PDF_FONT, 9) for r in rows if i < len(r)),
        )
        + _CELL_PAD
        for i in range(n)
    ]
    if sum(natural) <= _TABLE_WIDTH:
        return [header, *rows], None
    widths: list[float] = [0.0] * n  # every column is set below
    remaining, open_cols = _TABLE_WIDTH, list(range(n))
    while open_cols:
        share = remaining / len(open_cols)
        fits = [i for i in open_cols if natural[i] <= share]
        if not fits:
            for i in open_cols:
                widths[i] = share
            break
        for i in fits:
            widths[i] = natural[i]
            remaining -= natural[i]
            open_cols.remove(i)
    wrapped = {i for i in range(n) if widths[i] < natural[i]}
    cell_style = ParagraphStyle("TableCell", parent=styles["BodyText"], fontSize=9, leading=11)

    def cell(i: int, v: Any) -> Any:
        return Paragraph(html.escape(str(v)), cell_style) if i in wrapped else v

    body = [[cell(i, v) for i, v in enumerate(r)] for r in rows]
    return [header, *body], [float(w) for w in widths]


def _pdf_status_banner(status: str, text: str, styles):
    """A status banner rendered as a single-row colored Table for PDF."""
    cs = _STATUS_COLORS.get(status, _STATUS_COLORS["UNKNOWN"])
    para = Paragraph(html.escape(text), styles["Banner"])
    t = Table([[para]], colWidths=[6.7 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(cs["bg"])),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(cs["fg"])),
                ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor(cs["border"])),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
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
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
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
        rows.append(
            [
                h.get("ticker", ""),
                verdict,
                conf_str,
                pnl_str,
                h.get("sector") or "—",
            ]
        )
    table = Table(
        rows,
        repeatRows=1,
        hAlign="LEFT",
        colWidths=[1.0 * inch, 0.9 * inch, 0.7 * inch, 0.9 * inch, 1.8 * inch],
    )
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
    # reportlab builds `slices` at runtime, so a type checker can't see it.
    pie.slices.strokeWidth = 1  # ty: ignore[unresolved-attribute]
    pie.slices.strokeColor = colors.white  # ty: ignore[unresolved-attribute]
    pie.innerRadiusFraction = 0.55
    for i, _ in enumerate(pie_data):
        pie.slices[i].fillColor = colors.HexColor(_PIE_PALETTE[i % len(_PIE_PALETTE)])  # ty: ignore[unresolved-attribute]
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
            spaceBefore=0,
            spaceAfter=0,
        ),
    )


def _pdf_equity_curve(d: dict[str, Any]) -> Drawing | None:
    """PDF counterpart of report_html._equity_curve_svg (same chart model)."""
    model = ledger_chart_model(d)
    if model is None:
        return None
    dates, series, lo, hi = model["dates"], model["series"], model["lo"], model["hi"]
    n = len(dates)
    w, h = 482, 200
    left, right, top, bottom = 38, 78, 22, 20
    pw, ph = w - left - right, h - top - bottom
    ink, muted = colors.HexColor(_INK), colors.HexColor(_INK_MUTED)

    def x(i: int) -> float:
        return left + pw * i / (n - 1)

    def y(v: float) -> float:
        return bottom + ph * (v - lo) / (hi - lo)

    def text(px: float, py: float, s: str, size: float, color, anchor: str = "start") -> String:
        return String(
            px, py, s, fontName=_PDF_FONT, fontSize=size, fillColor=color, textAnchor=anchor
        )

    drawing = Drawing(w, h)
    for t in model["ticks"]:
        drawing.add(
            Line(left, y(t), w - right, y(t), strokeColor=colors.HexColor(_GRID), strokeWidth=0.5)
        )
        drawing.add(text(left - 4, y(t) - 2.5, pct_tick(t), 7, muted, "end"))
    drawing.add(
        Line(
            left,
            y(0),
            w - right,
            y(0),
            strokeColor=colors.HexColor(_REFERENCE_LINE),
            strokeWidth=1,
            strokeDashArray=[3, 2],
        )
    )
    for i in sorted({0, (n - 1) // 2, n - 1}):
        anchor = "start" if i == 0 else "end" if i == n - 1 else "middle"
        drawing.add(text(x(i), 6, _short_date(dates[i]), 7, muted, anchor))
    for _, vals, color in series:
        points: list[float] = []
        for i, v in enumerate(vals):
            points.extend([x(i), y(v)])
        drawing.add(
            PolyLine(points, strokeColor=colors.HexColor(color), strokeWidth=1.6, strokeLineJoin=1)
        )
    # Label positions spread apart; PDF y grows upward, so nudge on -y.
    label_ys = [-v for v in _nudge_apart([-y(vals[-1]) for _, vals, _ in series], 10)]
    for (name, vals, color), ly in zip(series, label_ys, strict=True):
        drawing.add(
            Circle(
                x(n - 1),
                y(vals[-1]),
                2.5,
                fillColor=colors.HexColor(color),
                strokeColor=colors.white,
                strokeWidth=1,
            )
        )
        drawing.add(text(w - right + 8, ly - 2.5, f"{name} {vals[-1]:+.1f}%", 7.5, ink))
    legend_x = left
    drawing.add(text(legend_x, h - 11, "Return on invested capital —", 8, ink))
    legend_x += 118
    for name, color, dash in [
        *((name, color, None) for name, _, color in series),
        ("Break-even", _REFERENCE_LINE, [3, 2]),
    ]:
        drawing.add(
            Line(
                legend_x,
                h - 8,
                legend_x + 14,
                h - 8,
                strokeColor=colors.HexColor(color),
                strokeWidth=1.6,
                strokeDashArray=dash,
            )
        )
        drawing.add(text(legend_x + 18, h - 11, name, 8, ink))
        legend_x += 30 + len(name) * 5
    return drawing


def _pdf_bar_chart(d: dict[str, Any]) -> Drawing | None:
    """PDF counterpart of report_html._bar_chart_svg."""
    bars = d.get("bars") or []
    if not bars:
        return None
    unit = str(d.get("unit") or "")
    w, label_w, row_h = 482, 130, 20
    values = [float(b.get("value") or 0.0) for b in bars]
    value_texts = [
        f"{v:+.1f}{unit}" + (f"  {b.get('note')}" if b.get("note") else "")
        for b, v in zip(bars, values, strict=True)
    ]
    # Reserve room for the value text on whichever side of the bar it sits,
    # so a negative value never lands on top of its category label.
    neg_w = max(
        (stringWidth(t, _PDF_FONT, 7) for t, v in zip(value_texts, values, strict=True) if v < 0),
        default=0.0,
    )
    pos_w = max(
        (stringWidth(t, _PDF_FONT, 7) for t, v in zip(value_texts, values, strict=True) if v >= 0),
        default=0.0,
    )
    plot_l = label_w + 8 + (neg_w + 6 if neg_w else 0)
    plot_r = w - (pos_w + 6 if pos_w else 4)
    lo, hi = min(0.0, *values), max(0.0, *values)
    span = (hi - lo) or 1.0

    def x(v: float) -> float:
        return plot_l + (plot_r - plot_l) * (v - lo) / span

    h = row_h * len(bars) + 8
    drawing = Drawing(w, h)
    zero = x(0.0)
    drawing.add(
        Line(zero, 2, zero, h - 2, strokeColor=colors.HexColor(_INK_MUTED), strokeWidth=0.6)
    )
    for i, (b, v, value_text) in enumerate(zip(bars, values, value_texts, strict=True)):
        cy = h - 4 - row_h * i - row_h / 2
        x0, x1 = sorted((zero, x(v)))
        if x1 > x0:
            drawing.add(
                Rect(
                    x0,
                    cy - 5,
                    x1 - x0,
                    10,
                    fillColor=colors.HexColor(_SERIES_1),
                    strokeColor=None,
                )
            )
        drawing.add(
            String(
                label_w,
                cy - 3,
                str(b.get("label") or ""),
                fontName=_PDF_FONT,
                fontSize=8,
                fillColor=colors.HexColor(_INK),
                textAnchor="end",
            )
        )
        drawing.add(
            String(
                x1 + 4 if v >= 0 else x0 - 4,
                cy - 3,
                value_text,
                fontName=_PDF_FONT,
                fontSize=7,
                fillColor=colors.HexColor(_INK),
                textAnchor="start" if v >= 0 else "end",
            )
        )
    return drawing


_PDF_CATALYST_DIRECTION_COLORS: dict[str, str] = {
    "positive": "#166534",
    "negative": "#9c1010",
    "uncertain": "#9a5b00",
}


def _pdf_catalysts(catalysts: list[dict[str, Any]], styles) -> list[Any]:
    """PDF counterpart of report_html._catalysts_html."""
    if not catalysts:
        return []
    rows: list[list[Any]] = []
    for c in catalysts:
        direction = str(c.get("direction") or "uncertain")
        color = _PDF_CATALYST_DIRECTION_COLORS.get(direction, "#9a5b00")
        rows.append(
            [
                Paragraph(
                    html.escape(str(c.get("expected_date") or "Date TBD")), styles["BodyText"]
                ),
                Paragraph(
                    f"<font color='{color}'><b>{html.escape(direction)}</b></font>",
                    styles["BodyText"],
                ),
                Paragraph(
                    f"<font color='#6b7280'>{html.escape(str(c.get('impact') or ''))}</font>",
                    styles["BodyText"],
                ),
                Paragraph(html.escape(str(c.get("event") or "")), styles["BodyText"]),
            ]
        )
    t = Table(rows, hAlign="LEFT", colWidths=[0.95 * inch, 0.85 * inch, 0.7 * inch, 4.2 * inch])
    t.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return [
        Paragraph(
            "<font color='#0e7490' size='8'><b>UPCOMING CATALYSTS</b></font>", styles["BodyText"]
        ),
        t,
        Spacer(1, 6),
    ]


def _pdf_pick_card(d: dict[str, Any], styles, chart_data: bytes | None = None) -> list[Any]:
    """Render a structured pick as a header row of colored pill badges
    + per-section paragraphs. Returns a list of flowables (no Spacer
    around the page-break boundary).

    `chart_data`, when given, is drawn immediately under the header pills
    at a modest size — this keeps the chart on the SAME page as the pick's
    header/one-liner instead of spilling onto its own mostly-blank page,
    which is what happened when the chart was a full-width 3.5in image
    rendered as a separate flowable after a card that already filled most
    of a page."""
    flow: list[Any] = [_pdf_pick_header(d, styles)]
    if chart_data:
        try:
            flow.append(Image(BytesIO(chart_data), width=4.0 * inch, height=2.15 * inch))
            flow.append(Spacer(1, 6))
        except Exception as e:  # noqa: BLE001 — the page is worth more than its chart
            logger.warning("Dropped a chart from the PDF (%s)", e)
    flow.extend(_pdf_pick_body(d, styles))
    return flow


def _pdf_pick_header(d: dict[str, Any], styles) -> Table:
    """Ticker plus pill badges (rank, conviction, fragility, allocation,
    consensus) as one 6-column row."""
    ticker = str(d.get("ticker", ""))
    rank = d.get("rank")
    conviction = d.get("conviction") if isinstance(d.get("conviction"), int) else None
    fragility = d.get("fragility_rank") if isinstance(d.get("fragility_rank"), int) else None
    alloc_pct = d.get("allocation_pct")
    alloc_usd = d.get("allocation_usd")
    agreement_ratio = d.get("agreement_ratio")
    voting_providers = d.get("voting_providers")

    # Header row: ticker + pill badges as a single 6-column Table.
    pill_cells: list[Paragraph] = []
    pill_cells.append(
        Paragraph(
            f"<font size='14'><b>{html.escape(ticker)}</b></font>",
            styles["BodyText"],
        )
    )
    if rank is not None:
        pill_cells.append(_pdf_pill(f"#{rank}", "#fff", "#1f2937", styles))
    if conviction is not None:
        pill_cells.append(
            _pdf_pill(
                f"Conviction {conviction}/10",
                "#fff",
                _conviction_swatch(conviction),
                styles,
            )
        )
    if fragility in _FRAGILITY_COLORS:
        c = _FRAGILITY_COLORS[fragility]
        pill_cells.append(
            _pdf_pill(
                f"Fragility {fragility}/5",
                c["fg"],
                c["bg"],
                styles,
            )
        )
    if alloc_pct is not None:
        pill_cells.append(
            _pdf_pill(
                f"Allocation {alloc_pct:.1f}%",
                "#4c1d95",
                "#ece8fb",
                styles,
            )
        )
    elif alloc_usd is not None:
        pill_cells.append(
            _pdf_pill(
                f"Allocation ${alloc_usd:,.0f}",
                "#4c1d95",
                "#ece8fb",
                styles,
            )
        )
    if agreement_ratio is not None:
        n = len(voting_providers) if voting_providers else 0
        total = round(n / agreement_ratio) if agreement_ratio else n
        is_unanimous = agreement_ratio >= 0.999
        fg, bg = ("#1a7f37", "#e6f4ea") if is_unanimous else ("#9a5b00", "#fff4e5")
        pill_cells.append(_pdf_pill(f"Consensus {n}/{total}", fg, bg, styles))

    # Pad to 6 cells so all rows column-align.
    while len(pill_cells) < 6:
        pill_cells.append(Paragraph("", styles["BodyText"]))
    header = Table(
        [pill_cells],
        colWidths=[1.9 * inch] + [0.94 * inch] * 5,
        hAlign="LEFT",
    )
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return header


def _pdf_pick_body(d: dict[str, Any], styles) -> list[Any]:
    """One-liner, bull and bear cases, catalysts, alternatives, sizing."""
    flow: list[Any] = []
    one_liner = str(d.get("one_liner") or "").strip()
    if one_liner:
        flow.append(Paragraph(html.escape(one_liner), styles["BodyText"]))
        flow.append(Spacer(1, 6))

    def _section(label: str, body: str | None, color: str = "#374151") -> None:
        if not body:
            return
        flow.append(
            Paragraph(
                f"<font color='{color}' size='8'><b>{label.upper()}</b></font>",
                styles["BodyText"],
            )
        )
        flow.append(Paragraph(html.escape(str(body)), styles["BodyText"]))
        flow.append(Spacer(1, 4))

    _section("Bull thesis", d.get("bull_thesis"))
    if d.get("what_youre_betting_on"):
        flow.append(
            Paragraph(
                f"<i>You're betting on: {html.escape(str(d.get('what_youre_betting_on')))}</i>",
                styles["BodyText"],
            )
        )
        flow.append(Spacer(1, 4))
    _section("Bear case", d.get("bear_case"), color="#9c1010")
    if d.get("most_fragile_assumption"):
        flow.append(
            Paragraph(
                f"<b>Most fragile assumption:</b> "
                f"{html.escape(str(d.get('most_fragile_assumption')))}",
                styles["BodyText"],
            )
        )
        flow.append(Spacer(1, 2))
    if d.get("watch_metric"):
        flow.append(
            Paragraph(
                f"<b>Watch:</b> {html.escape(str(d.get('watch_metric')))}",
                styles["BodyText"],
            )
        )
        flow.append(Spacer(1, 4))
    flow.extend(_pdf_catalysts(d.get("catalysts") or [], styles))
    _section("Why this over alternatives", d.get("why_over_alternatives"))
    if d.get("sector_concentration_check"):
        flow.append(
            Paragraph(
                f"<font size='9' color='#6b7280'>Sector concentration: "
                f"{html.escape(str(d.get('sector_concentration_check')))}</font>",
                styles["BodyText"],
            )
        )
        flow.append(Spacer(1, 4))
    if d.get("allocation_rationale"):
        rationale_para = Paragraph(
            f"<b>Sizing rationale:</b> {html.escape(str(d.get('allocation_rationale')))}",
            styles["BodyText"],
        )
        wrap = Table([[rationale_para]], colWidths=[6.7 * inch])
        wrap.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#7c3aed")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
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
            f"{pct:.1f}%" if pct is not None else (f"${usd:,.0f}" if usd is not None else "—")
        )
        rows.append(
            [
                Paragraph(f"<b>{html.escape(str(a.get('ticker', '')))}</b>", styles["BodyText"]),
                Paragraph(
                    f"<font color='#4c1d95'><b>{html.escape(size_str)}</b></font>",
                    styles["BodyText"],
                ),
                Paragraph(html.escape(str(a.get("rationale") or "")), styles["BodyText"]),
            ]
        )
    t = Table(
        rows,
        repeatRows=1,
        hAlign="LEFT",
        colWidths=[0.9 * inch, 1.0 * inch, 4.8 * inch],
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    flow: list[Any] = [t, Spacer(1, 6)]
    if warnings:
        body = "<b>Concentration warnings:</b> " + " · ".join(html.escape(str(w)) for w in warnings)
        wpara = Paragraph(body, styles["BodyText"])
        wrap = Table([[wpara]], colWidths=[6.7 * inch])
        wrap.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4e0")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#e89c00")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#8a4a00")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        flow.append(wrap)
        flow.append(Spacer(1, 6))
    return flow


_PDF_FACTOR_TILT_COLORS: dict[str, str] = {
    "growth": "#0e7490",
    "value": "#166534",
    "quality": "#4c1d95",
    "momentum": "#b45309",
    "low_vol": "#374151",
}
_PDF_FACTOR_TILT_LABELS: dict[str, str] = {
    "growth": "Growth",
    "value": "Value",
    "quality": "Quality",
    "momentum": "Momentum",
    "low_vol": "Low-vol",
}
_PDF_FACTOR_TILT_ORDER = ["growth", "value", "quality", "momentum", "low_vol"]


def _pdf_factor_tilt_panel(d: dict[str, Any], styles) -> list[Any]:
    """PDF counterpart of `_factor_tilt_panel_html` — a compact table,
    one row per scope (portfolio + each pick), one column per named
    style-factor bucket (0-100, '—' when unavailable for that pick)."""
    portfolio = d.get("portfolio") or {}
    picks = d.get("picks") or []
    if not portfolio and not picks:
        return []
    header = [Paragraph("<b>Scope</b>", styles["BodyText"])] + [
        Paragraph(
            f"<font color='{_PDF_FACTOR_TILT_COLORS[b]}'><b>{_PDF_FACTOR_TILT_LABELS[b]}</b></font>",
            styles["BodyText"],
        )
        for b in _PDF_FACTOR_TILT_ORDER
    ]
    rows: list[list[Any]] = [header]

    def _row(label: str, tilt: dict[str, Any]) -> list[Any]:
        cells: list[Any] = [Paragraph(f"<b>{html.escape(label)}</b>", styles["BodyText"])]
        for b in _PDF_FACTOR_TILT_ORDER:
            v = tilt.get(b)
            cells.append(Paragraph(f"{v:.0f}" if v is not None else "—", styles["BodyText"]))
        return cells

    if portfolio:
        rows.append(_row("Portfolio", portfolio))
    for pick in picks:
        tilt = pick.get("tilt") or {}
        if not tilt:
            continue
        rows.append(_row(str(pick.get("ticker") or ""), tilt))

    t = Table(rows, repeatRows=1, hAlign="LEFT", colWidths=[1.3 * inch] + [1.08 * inch] * 5)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return [t, Spacer(1, 6)]


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
        strength_color = _theme_strength_color(strength if isinstance(strength, int) else None)
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
        header.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        flow.append(header)
        flow.append(
            Paragraph(
                f"<font size='9' color='#374151'>{html.escape(description)}</font>",
                styles["BodyText"],
            )
        )
        member_strip = ", ".join(str(t) for t in members[:18])
        suffix = f" +{len(members) - 18} more" if len(members) > 18 else ""
        flow.append(
            Paragraph(
                f"<font size='8' color='#6b7280'><b>Members:</b> "
                f"{html.escape(member_strip + suffix)}</font>",
                styles["BodyText"],
            )
        )
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
    flow: list[Any] = [_pdf_premortem_banner(verdict, styles), Spacer(1, 4)]
    if summary:
        flow.append(
            Paragraph(
                f"<font size='9' color='#374151'>{html.escape(summary)}</font>",
                styles["BodyText"],
            )
        )
        flow.append(Spacer(1, 6))

    for f in failures:
        flow.extend(_pdf_premortem_failure(f, styles))
    return flow


def _pdf_premortem_banner(verdict: str, styles) -> Table:
    vp = _VERDICT_PALETTE_PREMORTEM.get(verdict, _VERDICT_PALETTE_PREMORTEM["proceed_with_caveat"])
    verdict_label = {
        "proceed_as_planned": "PROCEED AS PLANNED",
        "proceed_with_caveat": "PROCEED WITH CAVEAT",
        "reconsider": "RECONSIDER",
    }.get(verdict, verdict.upper())

    banner = Table(
        [
            [
                Paragraph(
                    f"<font color='{vp['fg']}' size='11'><b>Verdict: "
                    f"{html.escape(verdict_label)}</b></font>",
                    styles["BodyText"],
                )
            ]
        ],
        colWidths=[6.5 * inch],
        hAlign="LEFT",
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(vp["bg"])),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(vp["border"])),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return banner


def _pdf_premortem_failure(f: dict[str, Any], styles) -> list[Any]:
    """Likelihood/severity pills, the triggering action, the story, and the
    early-warning callout for one imagined failure."""
    flow: list[Any] = []
    likelihood = str(f.get("likelihood") or "medium").lower()
    severity = str(f.get("severity") or "moderate").lower()
    trig = str(f.get("triggering_action") or "")
    narrative = str(f.get("failure_narrative") or "")
    warning = str(f.get("early_warning") or "")
    like_color = _LIKELIHOOD_COLOR.get(likelihood, "#6b7280")
    sev_color = _SEVERITY_COLOR.get(severity, "#6b7280")

    pill_row = Table(
        [
            [
                _pdf_pill(f"Likelihood: {likelihood}", "#fff", like_color, styles),
                _pdf_pill(f"Severity: {severity}", "#fff", sev_color, styles),
                Paragraph("", styles["BodyText"]),
            ]
        ],
        colWidths=[1.5 * inch, 1.5 * inch, 3.5 * inch],
        hAlign="LEFT",
    )
    pill_row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    flow.append(pill_row)
    flow.append(
        Paragraph(
            "<font size='8' color='#6b7280'><b>TRIGGERING ACTION</b></font>",
            styles["BodyText"],
        )
    )
    flow.append(
        Paragraph(
            f"<font size='9' color='#1f2937'><i>{html.escape(trig)}</i></font>",
            styles["BodyText"],
        )
    )
    flow.append(Spacer(1, 3))
    flow.append(
        Paragraph(
            f"<font size='9' color='#1f2937'>{html.escape(narrative)}</font>",
            styles["BodyText"],
        )
    )
    flow.append(Spacer(1, 3))
    warn_table = Table(
        [
            [
                Paragraph(
                    f"<font size='8' color='#8a4a00'><b>Early warning:</b> "
                    f"{html.escape(warning)}</font>",
                    styles["BodyText"],
                )
            ]
        ],
        colWidths=[6.5 * inch],
        hAlign="LEFT",
    )
    warn_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4e0")),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#e89c00")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    flow.append(warn_table)
    flow.append(Spacer(1, 8))
    return flow


def _pdf_holding_review_card(d: dict[str, Any], styles) -> list[Any]:
    """PDF counterpart of `_holding_review_card_html` — ticker header
    with verdict + confidence pills, then labeled sections."""
    position_context = str(d.get("position_context") or "")
    flow: list[Any] = [_pdf_review_header(d, styles)]
    if position_context:
        flow.append(
            Paragraph(
                f"<font color='#6b7280' size='9'>{html.escape(position_context)}</font>",
                styles["BodyText"],
            )
        )
        flow.append(Spacer(1, 6))

    flow.extend(_pdf_review_body(d, styles))
    return flow


def _pdf_review_header(d: dict[str, Any], styles) -> Table:
    """Ticker plus verdict, conviction and trim pills as one 4-column row."""
    ticker = str(d.get("ticker", ""))
    verdict = str(d.get("verdict") or "HOLD").upper()
    confidence = d.get("confidence") if isinstance(d.get("confidence"), int) else None
    trim_pct = d.get("trim_pct")

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
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return header


def _pdf_review_body(d: dict[str, Any], styles) -> list[Any]:
    """Outlook, reasoning, tax-lot plan, catalysts, wash-sale notice and
    what would change the reviewer's mind."""
    forward_outlook = str(d.get("forward_outlook") or "")
    reasoning = str(d.get("reasoning") or "")
    tax_lot_plan = d.get("tax_lot_plan") or []
    what_change = str(d.get("what_would_change_mind") or "")
    wash_sale_notice = d.get("wash_sale_notice")
    flow: list[Any] = []

    def _section(label: str, body: str, *, color: str = "#374151") -> None:
        if not body:
            return
        flow.append(
            Paragraph(
                f"<font color='{color}' size='8'><b>{html.escape(label.upper())}</b></font>",
                styles["BodyText"],
            )
        )
        flow.append(Paragraph(html.escape(body), styles["BodyText"]))
        flow.append(Spacer(1, 6))

    _section("Forward outlook", forward_outlook)
    _section("Reasoning", reasoning)

    if tax_lot_plan:
        flow.append(
            Paragraph(
                "<font color='#4c1d95' size='8'><b>TAX LOT PLAN</b></font>",
                styles["BodyText"],
            )
        )
        for line in tax_lot_plan:
            flow.append(
                Paragraph(
                    f"• {html.escape(str(line))}",
                    styles["BodyText"],
                )
            )
        flow.append(Spacer(1, 6))

    flow.extend(_pdf_catalysts(d.get("catalysts") or [], styles))

    if wash_sale_notice:
        notice_para = Paragraph(
            f"<b>Wash-sale notice:</b> {html.escape(str(wash_sale_notice))}",
            styles["BodyText"],
        )
        wrap = Table([[notice_para]], colWidths=[6.7 * inch])
        wrap.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fde4e4")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#d73030")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#9c1010")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        flow.append(wrap)
        flow.append(Spacer(1, 6))

    if what_change:
        flow.append(
            Paragraph(
                f"<i><font color='#6b7280' size='9'>"
                f"<b>What would change my mind:</b> "
                f"{html.escape(what_change)}</font></i>",
                styles["BodyText"],
            )
        )
        flow.append(Spacer(1, 10))

    return flow


_CC_HEADER_BG = colors.HexColor("#0d9488")  # teal for Premium Income
_CC_RLC_BG = colors.HexColor("#a16207")  # amber for Round-Lot Coverage
_CC_GRID = colors.HexColor("#d1d5db")


def _pdf_premium_income(data: dict, styles) -> list:
    """Render the Premium Income section as a ReportLab table + caption."""
    flow: list = []
    flow.append(Paragraph("<b>Premium Income</b>", styles["Heading3"]))
    header = ["Ticker", "Account", "Strike", "Expiry", "Qty", "Premium", "Δ", "Assign %"]
    rows = [header]
    for r in data.get("rows") or []:
        rows.append(
            [
                r["ticker"],
                r.get("account", "—"),
                f"${r['strike']:,.2f}",
                r["expiry"],
                str(r["contracts"]),
                f"${r['premium_usd']:,.0f}",
                f"{r['delta']:.2f}",
                f"{r['assignment_pct']}%",
            ]
        )
    t = Table(rows, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _CC_HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, _CC_GRID),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    flow.append(t)
    flow.append(Spacer(1, 6))
    flow.append(
        Paragraph(
            f"Gross premium: <b>${data.get('gross_premium_usd', 0):,.0f}</b> &nbsp;"
            f"Buffer (10%): -${data.get('slippage_buffer_usd', 0):,.0f} &nbsp;"
            f"Deployable: <b>${data.get('deployable_premium_usd', 0):,.0f}</b>",
            styles["BodyText"],
        )
    )
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
        table_rows.append(
            [
                r["ticker"],
                str(r["shares"]),
                f"{r['round_lots']} ({r['round_lot_shares']})",
                str(r["stub_shares"]),
                f"${r['stub_dollar_value']:,.0f}",
                f"${r['to_next_lot_cost']:,.0f}",
            ]
        )
    t = Table(table_rows, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _CC_RLC_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, _CC_GRID),
            ]
        )
    )
    flow.append(t)
    flow.append(
        Paragraph(
            f"Stub pool total: <b>${data.get('stub_pool_total_usd', 0):,.0f}</b>",
            styles["BodyText"],
        )
    )
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
        lines.append(f"Stub consolidation: ${data['stub_consolidation_usd']:,.0f}")
    lines.append(f"<b>Total dry powder: ${data.get('total_dry_powder_usd', 0):,.0f}</b>")
    flow.append(Paragraph("<br/>".join(lines), styles["BodyText"]))
    deps = data.get("deployments") or []
    if deps:
        body = "<br/>".join(
            f"&rarr; {d['action']} <b>{d['ticker']}</b> {d['sizing']}" for d in deps
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
        flow.append(
            Paragraph(
                f"<i><font color='#6b7280'>{html.escape(str(summary))}</font></i>",
                styles["BodyText"],
            )
        )
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
        rows.append(
            [
                Paragraph(
                    f"<font color='{c['fg']}'><b>{html.escape(action_type)}</b></font>",
                    styles["BodyText"],
                ),
                Paragraph(f"<b>{html.escape(ticker)}</b>", styles["BodyText"]),
                Paragraph(html.escape(sizing), styles["BodyText"]),
            ]
        )
        # Tint the Action column with the badge background so the row reads
        # at a glance the same way the HTML pill does.
        style_cmds.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(c["bg"])))
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
    doc.build(_pdf_flowables(sections, chart_bytes, _pdf_styles()))
    return buf.getvalue()


def _pdf_flowables(sections: list[Section], chart_bytes: dict[str, bytes], styles) -> list[Any]:
    flow: list[Any] = []
    # A pick_card is immediately followed by its own "image" section
    # (report_sections.py). Consumed inline by _pdf_pick_card instead of
    # dispatched separately, so the chart lands on the same page as the
    # card's header rather than spilling onto its own near-empty page.
    skip_next_image = False
    for i, s in enumerate(sections):
        if skip_next_image and s.kind == "image":
            skip_next_image = False
            continue
        if s.kind == "pick_card":
            if not s.data:
                continue
            next_ticker = (
                sections[i + 1].image_ticker
                if i + 1 < len(sections) and sections[i + 1].kind == "image"
                else None
            )
            chart_data = chart_bytes.get(next_ticker) if next_ticker else None
            flow.extend(_pdf_pick_card(s.data, styles, chart_data=chart_data))
            if chart_data:
                skip_next_image = True
            continue
        render = _PDF_SECTION_RENDERERS.get(s.kind)
        if render is not None:
            flow.extend(render(s, chart_bytes, styles))
    return flow


def _spaced(flowable: Any, space: float) -> list[Any]:
    """The flowable and the gap under it; nothing when there is nothing."""
    return [] if flowable is None else [flowable, Spacer(1, space)]


def _pdf_image_section(s: Section, chart_bytes: dict[str, bytes], styles) -> list[Any]:
    data = chart_bytes.get(s.image_ticker) if s.image_ticker else None
    if not data:
        return []
    try:
        return _spaced(Image(BytesIO(data), width=6.5 * inch, height=3.5 * inch), 4)
    except Exception as e:  # noqa: BLE001 — the page is worth more than its chart
        logger.warning("Dropped the %s chart from the PDF (%s)", s.image_ticker, e)
        return []


def _pdf_table_section(s: Section, chart_bytes: dict[str, bytes], styles) -> list[Any]:
    if not (s.table_header and s.table_rows):
        return []
    tdata, col_widths = _fit_table(s.table_header, s.table_rows, styles)
    t = Table(tdata, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return _spaced(t, 4)


def _data_panel(build: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
    """A renderer for a kind whose builder takes (data, styles)."""
    return lambda s, chart_bytes, styles: build(s.data, styles) if s.data else []


def _data_chart(build: Callable[[Any], Any]) -> Callable[..., list[Any]]:
    """A renderer for a kind whose builder takes (data) and may decline."""
    return lambda s, chart_bytes, styles: _spaced(build(s.data), 6) if s.data else []


# One renderer per SectionKind (pick_card is handled in _pdf_flowables):
# (section, chart_bytes, styles) -> flowables.
_PDF_SECTION_RENDERERS: dict[str, Callable[..., list[Any]]] = {
    "heading": lambda s, c, st: _spaced(
        Paragraph(html.escape(s.text), st[f"Heading{min(s.level, 3)}"]), 4
    ),
    "para": lambda s, c, st: _spaced(Paragraph(html.escape(s.text), st["BodyText"]), 4),
    "preformatted": lambda s, c, st: _spaced(Preformatted(s.text, st["Code"]), 4),
    "blockquote": lambda s, c, st: _spaced(Paragraph(html.escape(s.text), st["Quote"]), 6),
    "image": _pdf_image_section,
    "table": _pdf_table_section,
    "status_banner": lambda s, c, st: _spaced(_pdf_status_banner(s.status, s.text, st), 6),
    "metric_strip": lambda s, c, st: _spaced(_pdf_metric_strip(s.metrics or [], st), 6),
    "holdings_dashboard": lambda s, c, st: (
        _spaced(_pdf_holdings_dashboard(s.holdings), 6) if s.holdings else []
    ),
    "sector_pie": lambda s, c, st: _spaced(_pdf_sector_pie(s.pie_data), 6) if s.pie_data else [],
    "allocation_table": _data_panel(_pdf_allocation_table),
    "rebalance_action_table": _data_panel(_pdf_rebalance_action_table),
    "holding_review_card": _data_panel(_pdf_holding_review_card),
    "market_themes_panel": _data_panel(_pdf_market_themes_panel),
    "premortem_panel": _data_panel(_pdf_premortem_panel),
    "factor_tilt_panel": _data_panel(_pdf_factor_tilt_panel),
    "equity_curve": _data_chart(_pdf_equity_curve),
    "bar_chart": _data_chart(_pdf_bar_chart),
    "premium_income": _data_panel(_pdf_premium_income),
    "round_lot_coverage": _data_panel(_pdf_round_lot_coverage),
    "premium_deployment": _data_panel(_pdf_premium_deployment),
    "page_break": lambda s, c, st: [PageBreak()],
}
