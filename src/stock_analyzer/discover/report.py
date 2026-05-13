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


def parse_picks(ranker_text_or_output: object) -> list[tuple[int, str, str]]:
    """Return [(rank, ticker, one_liner), ...] sorted by rank.

    Accepts a structured `RankerOutput` (preferred — field reads) OR a
    free-text ranker output (legacy / discover-pipeline-output that
    hasn't been migrated yet)."""
    from .schemas import RankerOutput
    if isinstance(ranker_text_or_output, RankerOutput):
        return [
            (p.rank, p.ticker, p.one_liner)
            for p in sorted(ranker_text_or_output.picks, key=lambda p: p.rank)
        ]
    if not ranker_text_or_output or not isinstance(ranker_text_or_output, str):
        return []
    out: list[tuple[int, str, str]] = []
    for m in _PICK_RE.finditer(ranker_text_or_output):
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


def parse_verdict(review: object) -> str:
    """Return HOLD / TRIM / SELL.

    Accepts a `HoldingReview` (preferred — reads the field directly,
    no regex) OR a free-text review (legacy DB rows / partial runs).
    """
    from .schemas import HoldingReview
    if isinstance(review, HoldingReview):
        return review.verdict
    if not review or not isinstance(review, str):
        return "HOLD"
    m = _VERDICT_RE.search(review)
    return m.group(1).upper() if m else "HOLD"


def parse_confidence(review: object) -> int | None:
    """Return the 1-10 confidence integer.

    Accepts a `HoldingReview` (preferred) OR a free-text review."""
    from .schemas import HoldingReview
    if isinstance(review, HoldingReview):
        return review.confidence
    if not review or not isinstance(review, str):
        return None
    m = _CONFIDENCE_RE.search(review)
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
    # New structured-output kinds (Phase 4f) — renderer pulls fields from
    # `data` and produces a styled card / table instead of dumping prose.
    "pick_card", "allocation_table", "rebalance_action_table",
    "holding_review_card", "market_themes_panel",
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
    # Generic carrier for structured-output card kinds (pick_card, etc.)
    data: dict[str, Any] | None = None


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
    ranker_output: object = None,
    redteam_output: object = None,
    sizer_output: object = None,
    market_themes: object = None,
) -> list[Section]:
    # Prefer the structured Phase 4 objects when present; fall back to
    # parsing the free-text variants so legacy callers / partial runs
    # still render something.
    from .schemas import RankerOutput, RedTeamOutput, SizerOutput
    structured_ranker = ranker_output if isinstance(ranker_output, RankerOutput) else None
    structured_redteam = redteam_output if isinstance(redteam_output, RedTeamOutput) else None
    structured_sizer = sizer_output if isinstance(sizer_output, SizerOutput) else None

    today = date.today().isoformat()
    pick_blocks = _split_by_pick_blocks(ranker_text)
    bear_blocks = _split_by_ticker_blocks(redteam_text)
    alloc_blocks = _split_by_ticker_blocks(sizer_text)
    pick_order = [
        t for _, t, _ in parse_picks(structured_ranker or ranker_text)
    ]
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

    # Market themes panel — what's hot right now (drives ranker bias).
    from .schemas import MarketThemes
    if isinstance(market_themes, MarketThemes) and market_themes.themes:
        s.append(Section(kind="heading", text="Current market themes", level=2))
        s.append(Section(
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

    # Per-pick cards. When structured outputs are present, emit a single
    # rich pick_card section per ticker (renderer composes rank pill +
    # conviction badge + fragility chip + allocation + bull/bear prose).
    # Otherwise fall back to the legacy heading + preformatted layout.
    pick_by_ticker: dict[str, Any] = {}
    if structured_ranker:
        pick_by_ticker = {p.ticker: p for p in structured_ranker.picks}
    bear_by_ticker: dict[str, Any] = {}
    if structured_redteam:
        bear_by_ticker = {b.ticker: b for b in structured_redteam.bear_cases}
    alloc_by_ticker: dict[str, Any] = {}
    if structured_sizer:
        alloc_by_ticker = {a.ticker: a for a in structured_sizer.allocations}

    for ticker in pick_order:
        s.append(Section(kind="page_break"))
        if pick_by_ticker.get(ticker):
            pick = pick_by_ticker[ticker]
            bear = bear_by_ticker.get(ticker)
            alloc = alloc_by_ticker.get(ticker)
            s.append(Section(
                kind="pick_card",
                data={
                    "ticker": ticker,
                    "rank": pick.rank,
                    "one_liner": pick.one_liner,
                    "conviction": pick.conviction,
                    "time_horizon": pick.time_horizon,
                    "bull_thesis": pick.bull_thesis,
                    "what_youre_betting_on": pick.what_youre_betting_on,
                    "why_over_alternatives": pick.why_over_alternatives,
                    "sector_concentration_check": pick.sector_concentration_check,
                    "bear_case": bear.bear_case if bear else None,
                    "most_fragile_assumption": (
                        bear.most_fragile_assumption if bear else None
                    ),
                    "watch_metric": bear.watch_metric if bear else None,
                    "fragility_rank": bear.fragility_rank if bear else None,
                    "allocation_pct": (
                        alloc.allocation_pct if alloc else None
                    ),
                    "allocation_usd": (
                        alloc.allocation_usd if alloc else None
                    ),
                    "allocation_rationale": alloc.rationale if alloc else None,
                },
            ))
            s.append(Section(kind="image", image_ticker=ticker))
        else:
            s.append(Section(kind="heading", text=ticker, level=2))
            s.append(Section(kind="image", image_ticker=ticker))
            s.append(Section(kind="heading", text="Bull case", level=3))
            s.append(Section(kind="preformatted", text=pick_blocks.get(ticker, "(missing)")))
            s.append(Section(kind="heading", text="Bear case (red-team)", level=3))
            s.append(Section(kind="preformatted", text=bear_blocks.get(ticker, "(missing)")))
            s.append(Section(kind="heading", text="Position sizing", level=3))
            s.append(Section(kind="preformatted", text=alloc_blocks.get(ticker, "(missing)")))

    s.append(Section(kind="page_break"))
    # Structured allocation table when sizer ran in Phase 4e mode.
    if structured_sizer and structured_sizer.allocations:
        s.append(Section(kind="page_break"))
        s.append(Section(kind="heading", text="Allocation summary", level=2))
        s.append(Section(
            kind="allocation_table",
            data={
                "allocations": [
                    {
                        "ticker": a.ticker,
                        "pct": a.allocation_pct,
                        "usd": a.allocation_usd,
                        "rationale": a.rationale,
                    }
                    for a in structured_sizer.allocations
                ],
                "warnings": list(structured_sizer.concentration_warnings),
            },
        ))

    s.append(Section(kind="heading", text="Ranker correlation notes", level=2))
    if structured_ranker and structured_ranker.pairs_not_to_hold_together:
        for pair in structured_ranker.pairs_not_to_hold_together:
            s.append(Section(
                kind="para",
                text=(
                    f"{pair.ticker_a} + {pair.ticker_b}: "
                    f"{pair.shared_driver}"
                ),
            ))
    else:
        trailing = re.split(_PICK_RE, ranker_text)[-1].strip()
        s.append(Section(kind="preformatted", text=trailing or "(none)"))

    s.append(Section(kind="heading", text="Red-team summary", level=2))
    if structured_redteam:
        s.append(Section(
            kind="para",
            text=f"Single most fragile pick: {structured_redteam.single_most_fragile_pick}",
        ))
    else:
        s.append(Section(kind="preformatted", text=redteam_text.split("---")[-1].strip() or "(none)"))

    s.append(Section(kind="heading", text="Sizer concentration warnings", level=2))
    if structured_sizer:
        if structured_sizer.concentration_warnings:
            for w in structured_sizer.concentration_warnings:
                s.append(Section(kind="para", text=f"• {w}"))
        else:
            s.append(Section(kind="para", text="(none)"))
    else:
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


# Fragility-rank visual palette — 1 = most fragile (red), 5 = most resilient (green).
_FRAGILITY_COLORS: dict[int, dict[str, str]] = {
    1: {"bg": "#fde4e4", "fg": "#9c1010", "border": "#d73030"},
    2: {"bg": "#fde8d3", "fg": "#a3550b", "border": "#e89c00"},
    3: {"bg": "#fff4e0", "fg": "#8a4a00", "border": "#e89c00"},
    4: {"bg": "#e8f4f8", "fg": "#0c5e7c", "border": "#3b8fde"},
    5: {"bg": "#e6f4ea", "fg": "#0e6432", "border": "#1f9d55"},
}


def _conviction_swatch(score: int | None) -> str:
    """Return a hex color reflecting conviction strength (1=red → 10=green)."""
    if score is None:
        return "#9ca3af"
    if score >= 8:
        return "#0e6432"
    if score >= 6:
        return "#3b8fde"
    if score >= 4:
        return "#a3550b"
    return "#9c1010"


def _pick_card_html(d: dict[str, Any]) -> str:
    """Render a per-pick structured card. Uses pill badges for rank /
    conviction / fragility / allocation and stacks bull + bear prose."""
    ticker = html.escape(str(d.get("ticker", "")))
    rank = d.get("rank")
    conviction = d.get("conviction")
    fragility = d.get("fragility_rank")
    alloc_pct = d.get("allocation_pct")
    alloc_usd = d.get("allocation_usd")
    one_liner = html.escape(str(d.get("one_liner") or ""))
    bull = html.escape(str(d.get("bull_thesis") or ""))
    bear = html.escape(str(d.get("bear_case") or "")) if d.get("bear_case") else ""
    bet_on = html.escape(str(d.get("what_youre_betting_on") or ""))
    why_over = html.escape(str(d.get("why_over_alternatives") or ""))
    sector_concentration = html.escape(str(d.get("sector_concentration_check") or ""))
    most_fragile = html.escape(str(d.get("most_fragile_assumption") or ""))
    watch_metric = html.escape(str(d.get("watch_metric") or ""))
    alloc_rationale = html.escape(str(d.get("allocation_rationale") or ""))

    rank_html = (
        f"<span style='background:#1f2937;color:#fff;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:600'>#{rank}</span>"
        if rank is not None else ""
    )
    conv_color = _conviction_swatch(conviction if isinstance(conviction, int) else None)
    conv_html = (
        f"<span style='background:{conv_color};color:#fff;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:600'>"
        f"Conviction {conviction}/10</span>"
        if conviction is not None else ""
    )
    frag_html = ""
    if isinstance(fragility, int) and fragility in _FRAGILITY_COLORS:
        c = _FRAGILITY_COLORS[fragility]
        frag_html = (
            f"<span style='background:{c['bg']};color:{c['fg']};"
            f"border:1px solid {c['border']};padding:3px 10px;border-radius:12px;"
            f"font-size:12px;font-weight:600'>Fragility {fragility}/5</span>"
        )
    alloc_html = ""
    if alloc_pct is not None:
        alloc_html = (
            f"<span style='background:#ece8fb;color:#4c1d95;border:1px solid #7c3aed;"
            f"padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600'>"
            f"Allocation {alloc_pct:.1f}%</span>"
        )
    elif alloc_usd is not None:
        alloc_html = (
            f"<span style='background:#ece8fb;color:#4c1d95;border:1px solid #7c3aed;"
            f"padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600'>"
            f"Allocation ${alloc_usd:,.0f}</span>"
        )

    sections_html: list[str] = []
    if bull:
        sections_html.append(
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.5px;margin:10px 0 4px'>"
            f"Bull thesis</div><div>{bull}</div></div>"
        )
    if bet_on:
        sections_html.append(
            f"<div style='margin-top:6px;color:#374151;font-style:italic'>"
            f"You're betting on: {bet_on}</div>"
        )
    if bear:
        sections_html.append(
            f"<div><div style='font-size:11px;color:#9c1010;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.5px;margin:10px 0 4px'>"
            f"Bear case</div><div>{bear}</div></div>"
        )
    if most_fragile:
        sections_html.append(
            f"<div style='margin-top:4px;color:#374151'>"
            f"<b>Most fragile assumption:</b> {most_fragile}</div>"
        )
    if watch_metric:
        sections_html.append(
            f"<div style='margin-top:4px;color:#374151'>"
            f"<b>Watch:</b> {watch_metric}</div>"
        )
    if why_over:
        sections_html.append(
            f"<div><div style='font-size:11px;color:#6b7280;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.5px;margin:10px 0 4px'>"
            f"Why this over alternatives</div><div>{why_over}</div></div>"
        )
    if sector_concentration:
        sections_html.append(
            f"<div style='margin-top:4px;color:#374151;font-size:13px'>"
            f"Sector concentration: {sector_concentration}</div>"
        )
    if alloc_rationale:
        sections_html.append(
            f"<div style='margin-top:8px;background:#f3f4f6;padding:8px 12px;"
            f"border-left:3px solid #7c3aed;font-size:13px;color:#374151'>"
            f"<b>Sizing rationale:</b> {alloc_rationale}</div>"
        )

    return (
        f"<div style='border:1px solid #e5e7eb;border-radius:8px;padding:16px;"
        f"margin:16px 0;background:#fff'>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;gap:10px;"
        f"margin-bottom:10px'>"
        f"<h2 style='margin:0;border:none;padding:0'>{ticker}</h2>"
        f"{rank_html}{conv_html}{frag_html}{alloc_html}"
        f"</div>"
        f"<div style='color:#374151;margin-bottom:6px'>{one_liner}</div>"
        + "".join(sections_html)
        + "</div>"
    )


def _allocation_table_html(d: dict[str, Any]) -> str:
    """Render the structured sizing allocations as a clean numeric table."""
    allocations = d.get("allocations") or []
    warnings = d.get("warnings") or []
    if not allocations:
        return "<p>(no allocations)</p>"
    rows_html: list[str] = []
    for a in allocations:
        pct = a.get("pct")
        usd = a.get("usd")
        size_str = (
            f"{pct:.1f}%" if pct is not None
            else (f"${usd:,.0f}" if usd is not None else "—")
        )
        rows_html.append(
            f"<tr><td><b>{html.escape(str(a.get('ticker', '')))}</b></td>"
            f"<td style='text-align:right;color:#4c1d95;font-weight:600'>"
            f"{html.escape(size_str)}</td>"
            f"<td style='color:#374151;font-size:13px'>"
            f"{html.escape(str(a.get('rationale') or ''))}</td></tr>"
        )
    table_html = (
        "<table class='dashboard'><thead><tr>"
        "<th>Ticker</th><th style='text-align:right'>Size</th><th>Rationale</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
    )
    warnings_html = ""
    if warnings:
        items = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)
        warnings_html = (
            f"<div style='margin-top:8px;color:#8a4a00;background:#fff4e0;"
            f"padding:8px 12px;border-left:3px solid #e89c00'>"
            f"<b>Concentration warnings:</b><ul style='margin:4px 0 0 18px'>"
            f"{items}</ul></div>"
        )
    return table_html + warnings_html


_TREND_GLYPHS: dict[str, tuple[str, str]] = {
    "up":   ("▲", "#0e6432"),
    "flat": ("●", "#6b7280"),
    "down": ("▼", "#9c1010"),
}


def _theme_strength_color(strength: int | None) -> str:
    """Color-grade a 1-10 theme strength for visual pills."""
    if strength is None:
        return "#9ca3af"
    if strength >= 8:
        return "#0e6432"
    if strength >= 6:
        return "#3b8fde"
    if strength >= 4:
        return "#a3550b"
    return "#9c1010"


def _market_themes_panel_html(d: dict[str, Any]) -> str:
    """Render the detected market themes as a side-by-side grid of
    compact cards — one per theme with strength badge + trend arrow +
    description + member tickers."""
    themes = d.get("themes") or []
    if not themes:
        return ""
    cards: list[str] = []
    for theme in themes:
        name = html.escape(str(theme.get("name") or ""))
        description = html.escape(str(theme.get("description") or ""))
        strength = theme.get("strength")
        trending = str(theme.get("trending") or "flat")
        members = theme.get("member_tickers") or []
        glyph, glyph_color = _TREND_GLYPHS.get(trending, _TREND_GLYPHS["flat"])
        strength_color = _theme_strength_color(
            strength if isinstance(strength, int) else None
        )
        strength_pill = (
            f"<span style='background:{strength_color};color:#fff;"
            f"padding:2px 10px;border-radius:10px;font-size:11px;"
            f"font-weight:700'>{strength}/10</span>"
            if isinstance(strength, int) else ""
        )
        trend_pill = (
            f"<span style='color:{glyph_color};font-weight:700;"
            f"font-size:13px'>{glyph}</span>"
        )
        # Pack member tickers into a tight monospace strip.
        member_strip = ", ".join(
            f"<span style='font-family:ui-monospace,SFMono-Regular,"
            f"monospace;color:#374151'>{html.escape(str(t))}</span>"
            for t in members[:18]
        )
        if len(members) > 18:
            member_strip += f" <span style='color:#6b7280'>+{len(members)-18} more</span>"
        cards.append(
            f"<div style='border:1px solid #e5e7eb;border-radius:8px;"
            f"padding:12px 14px;margin:8px 0;background:#fafbfc'>"
            f"<div style='display:flex;flex-wrap:wrap;align-items:center;"
            f"gap:8px;margin-bottom:6px'>"
            f"<b style='font-size:14px;color:#111827'>{name}</b>"
            f"{strength_pill}{trend_pill}</div>"
            f"<div style='color:#374151;font-size:13px;margin-bottom:6px'>"
            f"{description}</div>"
            f"<div style='font-size:12px;color:#6b7280'>"
            f"<b>Members:</b> {member_strip}</div>"
            f"</div>"
        )
    return "".join(cards)


def _holding_review_card_html(d: dict[str, Any]) -> str:
    """Render a per-holding review (HoldingReview schema) as a structured
    card: ticker + verdict pill + confidence pill + position context,
    then labeled forward outlook / reasoning / what-would-change-your-mind
    paragraphs, with tax lot plan + wash-sale notice when present.

    Replaces the monospace `preformatted` dump that used to render
    HoldingReview.full_text verbatim."""
    ticker = html.escape(str(d.get("ticker", "")))
    verdict = str(d.get("verdict") or "HOLD").upper()
    confidence = d.get("confidence")
    trim_pct = d.get("trim_pct")
    position_context = html.escape(str(d.get("position_context") or ""))
    forward_outlook = html.escape(str(d.get("forward_outlook") or ""))
    reasoning = html.escape(str(d.get("reasoning") or ""))
    tax_lot_plan = d.get("tax_lot_plan") or []
    what_change = html.escape(str(d.get("what_would_change_mind") or ""))
    wash_sale_notice = d.get("wash_sale_notice")

    # Pill badges in header row.
    vc = _VERDICT_COLORS.get(verdict) or _VERDICT_COLORS["HOLD"]
    verdict_pill = (
        f"<span style='background:{vc['bg']};color:{vc['fg']};"
        f"border:1px solid {vc['border']};padding:3px 12px;"
        f"border-radius:12px;font-size:12px;font-weight:700;"
        f"letter-spacing:0.3px'>{verdict}</span>"
    )
    pills = [verdict_pill]
    if isinstance(confidence, int):
        cs = _conviction_swatch(confidence)
        pills.append(
            f"<span style='background:{cs};color:#fff;padding:3px 10px;"
            f"border-radius:12px;font-size:12px;font-weight:600'>"
            f"Conviction {confidence}/10</span>"
        )
    if verdict == "TRIM" and isinstance(trim_pct, (int, float)) and trim_pct > 0:
        pills.append(
            f"<span style='background:#fff;color:#a36500;"
            f"border:1px solid #e89c00;padding:3px 10px;border-radius:12px;"
            f"font-size:12px;font-weight:600'>Trim {trim_pct:.0f}%</span>"
        )

    # Build body sections (only include the ones present).
    body_parts: list[str] = []

    def _section(label: str, body: str, *, color: str = "#6b7280") -> None:
        if not body:
            return
        body_parts.append(
            f"<div style='margin-top:12px'>"
            f"<div style='font-size:11px;color:{color};font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px'>"
            f"{html.escape(label)}</div>"
            f"<div style='color:#1f2937'>{body}</div></div>"
        )

    _section("Forward outlook", forward_outlook)
    _section("Reasoning", reasoning)

    if tax_lot_plan:
        items = "".join(
            f"<li style='margin:3px 0'>{html.escape(str(line))}</li>"
            for line in tax_lot_plan
        )
        body_parts.append(
            f"<div style='margin-top:12px'>"
            f"<div style='font-size:11px;color:#4c1d95;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px'>"
            f"Tax lot plan</div>"
            f"<ul style='margin:4px 0 0 18px;padding:0;color:#1f2937'>"
            f"{items}</ul></div>"
        )

    if wash_sale_notice:
        body_parts.append(
            f"<div style='margin-top:12px;background:#fde4e4;"
            f"padding:8px 12px;border-left:3px solid #d73030;"
            f"color:#9c1010;font-size:13px'>"
            f"<b>Wash-sale notice:</b> "
            f"{html.escape(str(wash_sale_notice))}</div>"
        )

    if what_change:
        body_parts.append(
            f"<div style='margin-top:14px;padding-top:10px;"
            f"border-top:1px solid #e5e7eb;color:#6b7280;font-style:italic;"
            f"font-size:13px'>"
            f"<b style='color:#374151;font-style:normal'>"
            f"What would change my mind:</b> {what_change}</div>"
        )

    pos_html = (
        f"<div style='color:#6b7280;font-size:13px;margin-top:6px'>"
        f"{position_context}</div>"
        if position_context else ""
    )

    return (
        f"<div style='border:1px solid #e5e7eb;border-radius:8px;"
        f"padding:16px;margin:16px 0;background:#fff'>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;"
        f"gap:10px;margin-bottom:0'>"
        f"<h2 style='margin:0;border:none;padding:0;font-family:"
        f"ui-monospace,SFMono-Regular,monospace'>{ticker}</h2>"
        + "".join(pills)
        + f"</div>{pos_html}"
        + "".join(body_parts)
        + "</div>"
    )


def _rebalance_action_table_html(d: dict[str, Any]) -> str:
    """Render the rebalancer's structured actions list as a colored table:
    action-type badge (SELL/TRIM/ADD/BUY) + ticker + sizing."""
    actions = d.get("actions") or []
    summary = d.get("summary") or ""
    if not actions:
        return ""
    rows: list[str] = []
    for a in actions:
        action_type = str(a.get("action") or "")
        ticker = str(a.get("ticker") or "")
        sizing = str(a.get("sizing") or "")
        c = _VERDICT_COLORS.get(action_type) or _VERDICT_COLORS["HOLD"]
        badge = (
            f"<span style='background:{c['bg']};color:{c['fg']};"
            f"border:1px solid {c['border']};padding:3px 12px;"
            f"border-radius:12px;font-size:12px;font-weight:700;"
            f"letter-spacing:0.3px'>{html.escape(action_type)}</span>"
        )
        rows.append(
            f"<tr><td style='vertical-align:middle'>{badge}</td>"
            f"<td><b style='font-family:ui-monospace,SFMono-Regular,monospace'>"
            f"{html.escape(ticker)}</b></td>"
            f"<td style='color:#374151'>{html.escape(sizing)}</td></tr>"
        )
    parts: list[str] = []
    if summary:
        parts.append(
            f"<p style='color:#374151;font-style:italic;margin:8px 0 12px;"
            f"font-size:13px'>{html.escape(summary)}</p>"
        )
    parts.append(
        "<table class='dashboard'><thead><tr>"
        "<th style='width:1%'>Action</th>"
        "<th style='width:1%'>Ticker</th>"
        "<th>Sizing</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return "".join(parts)


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

        elif s.kind == "pick_card" and s.data:
            parts.append(_pick_card_html(s.data))

        elif s.kind == "allocation_table" and s.data:
            parts.append(_allocation_table_html(s.data))

        elif s.kind == "rebalance_action_table" and s.data:
            parts.append(_rebalance_action_table_html(s.data))

        elif s.kind == "holding_review_card" and s.data:
            parts.append(_holding_review_card_html(s.data))

        elif s.kind == "market_themes_panel" and s.data:
            parts.append(_market_themes_panel_html(s.data))

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
