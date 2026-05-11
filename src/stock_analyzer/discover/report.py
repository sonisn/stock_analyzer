"""Output renderers — HTML email body, PDF report, terminal summary.

Both HTML and PDF are generated from the same Section list so the layout
stays in sync. Email is now the only delivery surface (no markdown to disk):
  - HTML body for in-client reading (charts inline via cid: refs)
  - PDF attachment for archival / printing (same content, ReportLab-rendered)
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Any, Literal

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


# --- sections (unified IR for HTML + PDF) -----------------------------------


SectionKind = Literal[
    "heading", "para", "preformatted", "image", "blockquote", "table", "page_break"
]


@dataclass
class Section:
    kind: SectionKind
    text: str = ""
    level: int = 2
    image_ticker: str | None = None
    table_rows: list[list[str]] | None = None
    table_header: list[str] | None = None


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
) -> list[Section]:
    today = date.today().isoformat()
    pick_blocks = _split_by_pick_blocks(ranker_text)
    bear_blocks = _split_by_ticker_blocks(redteam_text)
    alloc_blocks = _split_by_ticker_blocks(sizer_text)
    pick_order = [t for _, t, _ in parse_picks(ranker_text)]
    survivors = [c for c in candidates if c["passed_filter"]]
    rejected = [c for c in candidates if not c["passed_filter"]]

    s: list[Section] = []

    s.append(Section("heading", f"Stock discovery picks — {today}", level=1))
    s.append(Section(
        "para",
        f"{universe_size} candidates considered, {len(survivors)} survived "
        f"hard filters, {len(pick_order)} picks.",
    ))

    if macro_summary:
        s.append(Section("heading", "Macro regime", level=2))
        s.append(Section("blockquote", macro_summary))

    if sector_rotation and sector_rotation.get("leaders"):
        leaders = ", ".join(sector_rotation.get("leaders", []))
        laggards = ", ".join(sector_rotation.get("laggards", []))
        s.append(Section("heading", "Sector rotation (6-month returns)", level=2))
        s.append(Section("para", f"Leaders: {leaders}"))
        s.append(Section("para", f"Laggards: {laggards}"))

    s.append(Section("heading", "Current holdings (concentration context)", level=2))
    s.append(Section("preformatted", holdings_summary or "(none)"))

    for ticker in pick_order:
        s.append(Section("page_break"))
        s.append(Section("heading", ticker, level=2))
        s.append(Section("image", image_ticker=ticker))
        s.append(Section("heading", "Bull case", level=3))
        s.append(Section("preformatted", pick_blocks.get(ticker, "(missing)")))
        s.append(Section("heading", "Bear case (red-team)", level=3))
        s.append(Section("preformatted", bear_blocks.get(ticker, "(missing)")))
        s.append(Section("heading", "Position sizing", level=3))
        s.append(Section("preformatted", alloc_blocks.get(ticker, "(missing)")))

    s.append(Section("page_break"))
    s.append(Section("heading", "Ranker correlation notes", level=2))
    trailing = re.split(_PICK_RE, ranker_text)[-1].strip()
    s.append(Section("preformatted", trailing or "(none)"))

    s.append(Section("heading", "Red-team summary", level=2))
    s.append(Section("preformatted", redteam_text.split("---")[-1].strip() or "(none)"))

    s.append(Section("heading", "Sizer concentration warnings", level=2))
    s.append(Section("preformatted", sizer_text.split("---")[-1].strip() or "(none)"))

    if survivors:
        s.append(Section("page_break"))
        s.append(Section("heading", "All candidates that passed filters", level=2))
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
            "table",
            table_header=["Ticker", "Score", "Fund.", "Trend", "Conv.", "Sector"],
            table_rows=rows,
        ))

    if rejected:
        s.append(Section("heading", "Rejected candidates", level=2))
        for c in sorted(rejected, key=lambda x: x["ticker"]):
            reasons = ", ".join(c.get("fail_reasons") or [])
            s.append(Section("para", f"{c['ticker']}: {reasons}"))

    return s


# --- HTML renderer (for email body) -----------------------------------------


_HTML_HEAD = """<!DOCTYPE html><html><head><meta charset='utf-8'><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     max-width:780px;margin:24px auto;padding:0 16px;color:#222;font-size:14px;line-height:1.5}
h1{font-size:22px;border-bottom:2px solid #ddd;padding-bottom:6px}
h2{font-size:18px;margin-top:24px;border-bottom:1px solid #eee;padding-bottom:4px}
h3{font-size:15px;margin-top:14px;color:#555}
pre{background:#f5f5f5;padding:10px;border-radius:4px;overflow-x:auto;
    white-space:pre-wrap;font-size:12px}
blockquote{margin:10px 0;padding:8px 12px;background:#f0f7ff;border-left:3px solid #3b8fde;
           font-style:italic}
img{max-width:100%;border:1px solid #ddd;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}
th{background:#eee}
hr{border:none;border-top:1px dashed #ccc;margin:24px 0}
</style></head><body>"""


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
        elif s.kind == "page_break":
            parts.append("<hr/>")
    parts.append("</body></html>")
    return "".join(parts)


# --- PDF renderer (ReportLab) -----------------------------------------------


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles["Heading1"].fontSize = 18
    styles["Heading1"].textColor = colors.HexColor("#222222")
    styles["Heading2"].fontSize = 14
    styles["Heading3"].fontSize = 12
    styles["Heading3"].textColor = colors.HexColor("#555555")
    styles.add(ParagraphStyle(
        name="Quote",
        parent=styles["BodyText"],
        backColor=colors.HexColor("#f0f7ff"),
        borderColor=colors.HexColor("#3b8fde"),
        borderWidth=1,
        borderPadding=6,
        leftIndent=10,
    ))
    return styles


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
