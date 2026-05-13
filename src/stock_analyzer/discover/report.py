"""Output renderers — HTML email body, PDF report, terminal summary.

Both HTML and PDF are generated from the same Section list so the layout
stays in sync. Email is now the only delivery surface (no markdown to disk):
  - HTML body for in-client reading (charts inline via cid: refs)
  - PDF attachment for archival / printing (same content, ReportLab-rendered)
"""
from __future__ import annotations

import html
import re
from datetime import date
from io import BytesIO
from typing import Any, Literal

from pydantic import BaseModel

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
from rich.console import Console
from rich.table import Table as RichTable

# --- pick / ticker block parsing --------------------------------------------

_PICK_RE = re.compile(
    r"^PICK\s+(\d+):\s+([A-Z][A-Z.\-]{0,5})\s+[—–-]\s+(.+)$", re.MULTILINE
)
_TICKER_BLOCK_RE = re.compile(r"^TICKER:\s*([A-Z][A-Z.\-]{0,5})\s*$", re.MULTILINE)


def parse_picks(ranker_text: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for m in _PICK_RE.finditer(ranker_text):
        out.append((int(m.group(1)), m.group(2), m.group(3).strip()))
    return out


def _split_by_ticker_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    matches = list(_TICKER_BLOCK_RE.finditer(text))
    for i, m in enumerate(matches):
        ticker = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[ticker] = text[start:end].strip()
    return blocks


def _split_by_pick_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    matches = list(_PICK_RE.finditer(text))
    for i, m in enumerate(matches):
        ticker = m.group(2)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[ticker] = text[start:end].strip()
    return blocks


# --- structured-output parsers (verdict/conf/action/status) ------------------

_VERDICT_RE = re.compile(r"^Verdict:\s*(HOLD|TRIM|SELL)\b", re.MULTILINE)
_CONFIDENCE_RE = re.compile(
    r"^Confidence\s*\(1-10\):\s*(\d+)", re.MULTILINE | re.IGNORECASE
)
_STATUS_RE = re.compile(
    r"^Status:\s*(NO ACTION RECOMMENDED|ACTION RECOMMENDED)", re.MULTILINE
)
_ACTION_RE = re.compile(
    r"^Action\s+\d+:\s+(SELL|TRIM|ADD|BUY)\s+([A-Z][A-Z.\-]{0,5})",
    re.MULTILINE,
)


def parse_verdict(review_text: str | None) -> str:
    if not review_text:
        return "HOLD"
    m = _VERDICT_RE.search(review_text)
    return m.group(1).upper() if m else "HOLD"


def parse_confidence(review_text: str | None) -> int | None:
    if not review_text:
        return None
    m = _CONFIDENCE_RE.search(review_text)
    return int(m.group(1)) if m else None


def parse_rebalance_status(rebalance_text_or_plan: object) -> str:
    """Return status as `NO_ACTION` / `ACTION` / `UNKNOWN`.

    Accepts either a structured `RebalancePlan` (preferred — read the
    field directly, no regex) OR a free-text plan (legacy / discover
    runs / older DB rows). The structured form removes the regex
    fragility that previously crashed the persist step on `None`."""
    # Lazy import to avoid a circular module load.
    from .rebalance_schema import RebalancePlan
    if isinstance(rebalance_text_or_plan, RebalancePlan):
        return rebalance_text_or_plan.status
    if not rebalance_text_or_plan:
        return "UNKNOWN"
    if not isinstance(rebalance_text_or_plan, str):
        return "UNKNOWN"
    m = _STATUS_RE.search(rebalance_text_or_plan)
    if not m:
        return "UNKNOWN"
    return "NO_ACTION" if "NO ACTION" in m.group(1) else "ACTION"


def parse_actions(rebalance_text_or_plan: object) -> list[tuple[str, str]]:
    """Return [(action_type, ticker), ...] preserving execution order.

    Reads from the structured RebalancePlan when given one (no regex);
    falls back to regex on free text for legacy/discover runs."""
    from .rebalance_schema import RebalancePlan
    if isinstance(rebalance_text_or_plan, RebalancePlan):
        return [(a.action, a.ticker) for a in rebalance_text_or_plan.actions]
    if not rebalance_text_or_plan or not isinstance(rebalance_text_or_plan, str):
        return []
    return [
        (m.group(1), m.group(2))
        for m in _ACTION_RE.finditer(rebalance_text_or_plan)
    ]


# --- visual constants -------------------------------------------------------

# Color palette for verdict/action badges. Used by both HTML (hex CSS) and
# PDF (ReportLab HexColor) renderers so they look identical.
_VERDICT_COLORS = {
    "HOLD": {"bg": "#e8f4f8", "fg": "#0c5e7c", "border": "#3b8fde"},
    "TRIM": {"bg": "#fff4e0", "fg": "#a36500", "border": "#e89c00"},
    "SELL": {"bg": "#fde4e4", "fg": "#9c1010", "border": "#d73030"},
    "ADD":  {"bg": "#ece8fb", "fg": "#4c1d95", "border": "#7c3aed"},
    "BUY":  {"bg": "#e6f4ea", "fg": "#0e6432", "border": "#1f9d55"},
}
_STATUS_COLORS = {
    "NO_ACTION": {"bg": "#e6f4ea", "fg": "#0e6432", "border": "#1f9d55"},
    "ACTION":    {"bg": "#fff4e0", "fg": "#8a4a00", "border": "#e89c00"},
    "UNKNOWN":   {"bg": "#f0f0f0", "fg": "#444", "border": "#888"},
}
# Categorical palette for sector pie slices.
_PIE_PALETTE = [
    "#3b8fde", "#1f9d55", "#e89c00", "#d73030", "#7c3aed",
    "#0891b2", "#65a30d", "#dc2626", "#ea580c", "#0284c7",
    "#16a34a", "#a16207",
]


# --- sections (unified IR for HTML + PDF) -----------------------------------


SectionKind = Literal[
    "heading", "para", "preformatted", "image", "blockquote", "table",
    "page_break", "status_banner", "metric_strip", "holdings_dashboard",
    "sector_pie",
]


class Section(BaseModel):
    kind: SectionKind
    text: str = ""
    level: int = 2
    image_ticker: str | None = None
    table_rows: list[list[str]] | None = None
    table_header: list[str] | None = None
    # Status banner: kind="status_banner", text=display label, level=color-key
    # ("NO_ACTION" | "ACTION" | "UNKNOWN")
    status: str = ""
    # Metric strip: list of (label, value) shown as colored cards
    metrics: list[tuple[str, str]] | None = None
    # Holdings dashboard: list of dicts with ticker, verdict, confidence,
    # pnl_pct, sector, concerns (parsed by build_sections)
    holdings: list[dict[str, Any]] | None = None
    # Sector pie data: list of (label, value, color)
    pie_data: list[tuple[str, float]] | None = None


def build_sections(
    *,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    candidates: list[dict[str, Any]],
    universe_size: int,
    holdings_summary: str,
    macro_summary: str = "",
    sector_rotation: dict[str, Any] | None = None,
    track_record_block: str = "",
) -> list[Section]:
    today = date.today().isoformat()
    pick_blocks = _split_by_pick_blocks(ranker_text)
    bear_blocks = _split_by_ticker_blocks(redteam_text)
    alloc_blocks = _split_by_ticker_blocks(sizer_text)
    pick_order = [t for _, t, _ in parse_picks(ranker_text)]
    survivors = [c for c in candidates if c["passed_filter"]]
    rejected = [c for c in candidates if not c["passed_filter"]]

    s: list[Section] = []

    s.append(Section(kind="heading", text=f"Stock discovery picks — {today}", level=1))
    s.append(Section(
        kind="para",
        text=(
            f"{universe_size} candidates considered, {len(survivors)} survived "
            f"hard filters, {len(pick_order)} picks."
        ),
    ))

    if track_record_block:
        s.append(Section(kind="heading", text="Track record", level=2))
        s.append(Section(kind="preformatted", text=track_record_block))

    if macro_summary:
        s.append(Section(kind="heading", text="Macro regime", level=2))
        s.append(Section(kind="blockquote", text=macro_summary))

    if sector_rotation and sector_rotation.get("leaders"):
        leaders = ", ".join(sector_rotation.get("leaders", []))
        laggards = ", ".join(sector_rotation.get("laggards", []))
        s.append(Section(kind="heading", text="Sector rotation (6-month returns)", level=2))
        s.append(Section(kind="para", text=f"Leaders: {leaders}"))
        s.append(Section(kind="para", text=f"Laggards: {laggards}"))

    s.append(Section(kind="heading", text="Current holdings (concentration context)", level=2))
    s.append(Section(kind="preformatted", text=holdings_summary or "(none)"))

    for ticker in pick_order:
        s.append(Section(kind="page_break"))
        s.append(Section(kind="heading", text=ticker, level=2))
        s.append(Section(kind="image",image_ticker=ticker))
        s.append(Section(kind="heading", text="Bull case", level=3))
        s.append(Section(kind="preformatted", text=pick_blocks.get(ticker, "(missing)")))
        s.append(Section(kind="heading", text="Bear case (red-team)", level=3))
        s.append(Section(kind="preformatted", text=bear_blocks.get(ticker, "(missing)")))
        s.append(Section(kind="heading", text="Position sizing", level=3))
        s.append(Section(kind="preformatted", text=alloc_blocks.get(ticker, "(missing)")))

    s.append(Section(kind="page_break"))
    s.append(Section(kind="heading", text="Ranker correlation notes", level=2))
    trailing = re.split(_PICK_RE, ranker_text)[-1].strip()
    s.append(Section(kind="preformatted", text=trailing or "(none)"))

    s.append(Section(kind="heading", text="Red-team summary", level=2))
    s.append(Section(kind="preformatted", text=redteam_text.split("---")[-1].strip() or "(none)"))

    s.append(Section(kind="heading", text="Sizer concentration warnings", level=2))
    s.append(Section(kind="preformatted", text=sizer_text.split("---")[-1].strip() or "(none)"))

    if survivors:
        s.append(Section(kind="page_break"))
        s.append(Section(kind="heading", text="All candidates that passed filters", level=2))
        rows: list[list[str]] = []
        for c in sorted(survivors, key=lambda x: x.get("score") or 0, reverse=True):
            comp = c.get("score_components") or {}
            rows.append([
                c["ticker"],
                f"{c.get('score') or '—'}",
                f"{comp.get('fundamentals', '—')}",
                f"{comp.get('trend', '—')}",
                f"{comp.get('conviction', '—')}",
                c.get("sector") or "—",
            ])
        s.append(Section(
            kind="table",
            table_header=["Ticker", "Score", "Fund.", "Trend", "Conv.", "Sector"],
            table_rows=rows,
        ))

    if rejected:
        s.append(Section(kind="heading", text="Rejected candidates", level=2))
        for c in sorted(rejected, key=lambda x: x["ticker"]):
            reasons = ", ".join(c.get("fail_reasons") or [])
            s.append(Section(kind="para", text=f"{c['ticker']}: {reasons}"))

    return s


# --- HTML renderer (for email body) -----------------------------------------


_HTML_HEAD = """<!DOCTYPE html><html><head><meta charset='utf-8'><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     max-width:820px;margin:24px auto;padding:0 16px;color:#1f2937;
     font-size:14px;line-height:1.55;background:#fafbfc}
h1{font-size:24px;margin-bottom:4px;color:#111827}
h2{font-size:18px;margin-top:32px;color:#111827;
   border-bottom:2px solid #e5e7eb;padding-bottom:6px}
h3{font-size:15px;margin-top:18px;color:#374151;font-weight:600}
pre{background:#f3f4f6;padding:14px;border-radius:6px;overflow-x:auto;
    white-space:pre-wrap;font-size:12.5px;line-height:1.45;
    border:1px solid #e5e7eb;color:#1f2937}
blockquote{margin:14px 0;padding:12px 16px;background:#eff6ff;
           border-left:4px solid #3b82f6;font-style:italic;border-radius:4px}
img{max-width:100%;border:1px solid #e5e7eb;border-radius:6px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border:1px solid #e5e7eb;padding:8px 12px;text-align:left}
th{background:#f3f4f6;font-weight:600;color:#374151}
hr{border:none;border-top:1px dashed #d1d5db;margin:24px 0}
.banner{margin:14px 0;padding:18px 22px;border-radius:8px;border-left:6px solid;
        font-weight:600;font-size:16px;letter-spacing:0.2px;
        box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.banner-sub{display:block;margin-top:6px;font-weight:400;font-size:13px;
            color:#374151;letter-spacing:0;line-height:1.5}
.metrics{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 8px 0}
.metric{flex:1;min-width:140px;padding:12px 14px;background:#fff;
        border:1px solid #e5e7eb;border-radius:6px;text-align:center;
        box-shadow:0 1px 2px rgba(0,0,0,0.03)}
.metric-label{font-size:11px;text-transform:uppercase;color:#6b7280;
              letter-spacing:0.4px;margin-bottom:4px}
.metric-value{font-size:18px;font-weight:600;color:#111827}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;
       font-size:11px;font-weight:600;letter-spacing:0.3px;border:1px solid}
.confbar{display:inline-block;width:60px;height:10px;background:#e5e7eb;
         border-radius:6px;overflow:hidden;vertical-align:middle;
         margin-left:6px}
.confbar-fill{display:block;height:100%}
.pl-pos{color:#16a34a;font-weight:600}
.pl-neg{color:#dc2626;font-weight:600}
.dashboard td{vertical-align:middle}
.pie-wrap{display:flex;gap:16px;align-items:center;margin:12px 0}
.pie-legend{flex:1;font-size:12px}
.pie-legend li{list-style:none;padding:3px 0;display:flex;align-items:center}
.pie-legend ul{margin:0;padding:0}
.legend-swatch{display:inline-block;width:12px;height:12px;border-radius:2px;
               margin-right:8px;flex-shrink:0}
</style></head><body>"""


def _badge_html(label: str) -> str:
    """Inline-styled badge for verdict/action keywords (HOLD/TRIM/SELL/etc.)."""
    colors_set = _VERDICT_COLORS.get(label.upper(), _VERDICT_COLORS["HOLD"])
    return (
        f"<span class='badge' style='background:{colors_set['bg']};"
        f"color:{colors_set['fg']};border-color:{colors_set['border']}'>"
        f"{html.escape(label.upper())}</span>"
    )


def _confidence_bar_html(conf: int | None) -> str:
    if conf is None:
        return ""
    # Color goes red (1) → amber (5) → green (10).
    if conf <= 4:
        fill_color = "#dc2626"
    elif conf <= 6:
        fill_color = "#e89c00"
    else:
        fill_color = "#16a34a"
    pct = max(10, min(100, conf * 10))
    return (
        f"<span class='confbar'><span class='confbar-fill' "
        f"style='width:{pct}%;background:{fill_color}'></span></span>"
        f"<span style='font-size:11px;color:#6b7280;margin-left:4px'>{conf}/10</span>"
    )


def _pl_class(pct: float | None) -> str:
    if pct is None:
        return ""
    return "pl-pos" if pct >= 0 else "pl-neg"


def _svg_pie(pie_data: list[tuple[str, float]], diameter: int = 180) -> str:
    """Render a donut-style pie via inline SVG with legend. No external deps."""
    if not pie_data:
        return ""
    total = sum(v for _, v in pie_data if v > 0)
    if total <= 0:
        return ""
    import math

    r = diameter / 2 - 4
    cx = cy = diameter / 2
    inner_r = r * 0.55  # donut hole

    paths: list[str] = []
    legend_items: list[str] = []
    start_angle = -math.pi / 2  # top
    for i, (label, value) in enumerate(pie_data):
        if value <= 0:
            continue
        fraction = value / total
        end_angle = start_angle + fraction * 2 * math.pi
        color = _PIE_PALETTE[i % len(_PIE_PALETTE)]

        x1 = cx + r * math.cos(start_angle)
        y1 = cy + r * math.sin(start_angle)
        x2 = cx + r * math.cos(end_angle)
        y2 = cy + r * math.sin(end_angle)
        x3 = cx + inner_r * math.cos(end_angle)
        y3 = cy + inner_r * math.sin(end_angle)
        x4 = cx + inner_r * math.cos(start_angle)
        y4 = cy + inner_r * math.sin(start_angle)
        large = 1 if fraction > 0.5 else 0
        d = (
            f"M {x1:.2f} {y1:.2f} A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f} "
            f"L {x3:.2f} {y3:.2f} A {inner_r} {inner_r} 0 {large} 0 "
            f"{x4:.2f} {y4:.2f} Z"
        )
        paths.append(
            f'<path d="{d}" fill="{color}" stroke="#fff" stroke-width="1.5"/>'
        )
        legend_items.append(
            f"<li><span class='legend-swatch' style='background:{color}'></span>"
            f"<span style='flex:1'>{html.escape(label)}</span>"
            f"<span style='color:#6b7280'>{fraction * 100:.1f}%</span></li>"
        )
        start_angle = end_angle

    return (
        "<div class='pie-wrap'>"
        f"<svg width='{diameter}' height='{diameter}' viewBox='0 0 {diameter} {diameter}'>"
        + "".join(paths)
        + "</svg>"
        "<div class='pie-legend'><ul>"
        + "".join(legend_items)
        + "</ul></div></div>"
    )


def render_html_email(sections: list[Section], chart_cids: dict[str, str]) -> str:
    parts: list[str] = [_HTML_HEAD]
    for s in sections:
        if s.kind == "heading":
            parts.append(f"<h{s.level}>{html.escape(s.text)}</h{s.level}>")

        elif s.kind == "para":
            parts.append(f"<p>{html.escape(s.text)}</p>")

        elif s.kind == "preformatted":
            parts.append(f"<pre>{html.escape(s.text)}</pre>")

        elif s.kind == "blockquote":
            parts.append(f"<blockquote>{html.escape(s.text)}</blockquote>")

        elif s.kind == "image" and s.image_ticker:
            cid = chart_cids.get(s.image_ticker)
            if cid:
                parts.append(
                    f"<img src='cid:{cid}' alt='{html.escape(s.image_ticker)} chart' />"
                )

        elif s.kind == "table" and s.table_header and s.table_rows:
            parts.append("<table><thead><tr>")
            for h in s.table_header:
                parts.append(f"<th>{html.escape(h)}</th>")
            parts.append("</tr></thead><tbody>")
            for row in s.table_rows:
                parts.append("<tr>")
                for cell in row:
                    parts.append(f"<td>{html.escape(cell)}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table>")

        elif s.kind == "status_banner":
            cs = _STATUS_COLORS.get(s.status, _STATUS_COLORS["UNKNOWN"])
            parts.append(
                f"<div class='banner' style='background:{cs['bg']};"
                f"color:{cs['fg']};border-left-color:{cs['border']}'>"
                f"{html.escape(s.text)}"
            )
            # Optional sub-text already in s.text using "\n" — second line as banner-sub
            parts.append("</div>")

        elif s.kind == "metric_strip" and s.metrics:
            parts.append("<div class='metrics'>")
            for label, value in s.metrics:
                parts.append(
                    f"<div class='metric'>"
                    f"<div class='metric-label'>{html.escape(label)}</div>"
                    f"<div class='metric-value'>{html.escape(value)}</div>"
                    f"</div>"
                )
            parts.append("</div>")

        elif s.kind == "holdings_dashboard" and s.holdings:
            parts.append("<table class='dashboard'><thead><tr>"
                         "<th>Ticker</th><th>Verdict</th><th>Conf</th>"
                         "<th>P/L</th><th>Sector</th><th>Forward note</th>"
                         "</tr></thead><tbody>")
            for h_row in s.holdings:
                verdict = (h_row.get("verdict") or "HOLD").upper()
                conf = h_row.get("confidence")
                pnl = h_row.get("pnl_pct")
                pnl_str = (
                    f"{pnl:+.1f}%" if isinstance(pnl, (int, float)) else "—"
                )
                pl_cls = _pl_class(pnl if isinstance(pnl, (int, float)) else None)
                parts.append(
                    f"<tr>"
                    f"<td><b>{html.escape(h_row.get('ticker', ''))}</b></td>"
                    f"<td>{_badge_html(verdict)}</td>"
                    f"<td>{_confidence_bar_html(conf)}</td>"
                    f"<td class='{pl_cls}'>{pnl_str}</td>"
                    f"<td style='color:#6b7280'>{html.escape(h_row.get('sector') or '—')}</td>"
                    f"<td style='color:#374151;font-size:12px'>{html.escape(h_row.get('note') or '')}</td>"
                    f"</tr>"
                )
            parts.append("</tbody></table>")

        elif s.kind == "sector_pie" and s.pie_data:
            parts.append(_svg_pie(s.pie_data))

        elif s.kind == "page_break":
            parts.append("<hr/>")
    parts.append("</body></html>")
    return "".join(parts)


# --- PDF renderer (ReportLab) -----------------------------------------------


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

        elif s.kind == "page_break":
            flow.append(PageBreak())

    doc.build(flow)
    return buf.getvalue()


# --- terminal summary -------------------------------------------------------


def print_terminal_summary(ranker_text: str, sizer_text: str) -> None:
    picks = parse_picks(ranker_text)
    sizer_blocks = _split_by_ticker_blocks(sizer_text)

    table = RichTable(title="Top picks for 6-12 month hold", show_lines=True)
    table.add_column("Rank", justify="right")
    table.add_column("Ticker", style="bold")
    table.add_column("Allocation")
    table.add_column("One-liner")
    for rank, ticker, one_liner in picks:
        alloc = "—"
        block = sizer_blocks.get(ticker)
        if block:
            m = re.search(r"Allocation:\s*(.+)", block)
            if m:
                alloc = m.group(1).strip()
        table.add_row(str(rank), ticker, alloc, one_liner[:80])
    Console().print(table)
