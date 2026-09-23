"""HTML renderer for the report Section IR.

Generates the email body. Reads Section objects produced by
`report_sections.build_sections` (or by the rebalance pipeline's own
section builder) and emits a styled HTML document with inline images
referenced via `cid:` so the SMTP layer can attach the PNG charts.

Stays in sync with `report_pdf.py` because both pull palettes from
`report_sections` — see `_VERDICT_COLORS`, `_STATUS_COLORS`, etc.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable
from datetime import date
from typing import Any

from ..models.reports import Section
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
        paths.append(f'<path d="{d}" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
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
        "<div class='pie-legend'><ul>" + "".join(legend_items) + "</ul></div></div>"
    )


# Chart series colors — validated categorical slots 1/2 (light surface):
# lightness band, chroma floor, CVD + normal-vision separation, contrast.
_SERIES_1 = "#2a78d6"
_SERIES_2 = "#eb6834"
_REFERENCE_LINE = "#9ca3af"
_GRID = "#e5e7eb"
_INK = "#374151"
_INK_MUTED = "#6b7280"


def _money_short(v: float) -> str:
    if abs(v) >= 1000:
        return f"${v / 1000:.1f}k"
    return f"${v:,.0f}"


def _short_date(iso: str) -> str:
    try:
        return date.fromisoformat(iso).strftime("%b %d")
    except ValueError:
        return iso


def _nudge_apart(ys: list[float], min_gap: float) -> list[float]:
    """Spread label y-positions so none sit closer than `min_gap`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:], strict=False):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round-number ticks (1/2/2.5/5 x 10^k) spanning [lo, hi]."""
    span = (hi - lo) or 1.0
    raw = span / max(target - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    if ticks[-1] < hi:
        ticks.append(round(v, 10))
    return ticks


def pct_tick(t: float) -> str:
    return "0%" if t == 0 else f"{t:+g}%"


def ledger_chart_model(d: dict[str, Any]) -> dict[str, Any] | None:
    """Shared geometry for the paper-ledger chart (HTML + PDF).

    Plots return on invested capital, not dollar value: the dollar curve is
    dominated by each new tranche's contribution, which hides the gap the
    chart exists to show."""
    dates = d.get("dates") or []
    invested = d.get("invested") or []
    n = len(dates)
    if n < 2 or len(invested) != n:
        return None
    series = []
    for name, key, color in (("Picks", "strategy", _SERIES_1), ("SPY", "benchmark", _SERIES_2)):
        vals = d.get(key) or []
        if len(vals) != n:
            return None
        pct = [(v / inv - 1) * 100 if inv else 0.0 for v, inv in zip(vals, invested, strict=True)]
        series.append((name, pct, color))
    all_vals = [v for _, vals, _ in series for v in vals] + [0.0]
    ticks = _nice_ticks(min(all_vals), max(all_vals))
    return {"dates": dates, "series": series, "ticks": ticks, "lo": ticks[0], "hi": ticks[-1]}


def _equity_curve_svg(d: dict[str, Any]) -> str:
    """Picks vs SPY return on invested capital over time, a dashed
    break-even line, end-of-line labels, legend above."""
    model = ledger_chart_model(d)
    if model is None:
        return ""
    dates, series, lo, hi = model["dates"], model["series"], model["lo"], model["hi"]
    n = len(dates)
    w, h, left, right, top, bottom = 640, 240, 48, 96, 10, 28
    pw, ph = w - left - right, h - top - bottom

    def x(i: int) -> float:
        return left + pw * i / (n - 1)

    def y(v: float) -> float:
        return top + ph * (1 - (v - lo) / (hi - lo))

    parts = [
        f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px;display:block' "
        f"font-family='-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif' "
        f"role='img' aria-label='Paper portfolio return vs SPY'>",
        f"<title>{html.escape(_short_date(dates[-1]))}: "
        + ", ".join(f"{name} {vals[-1]:+.1f}%" for name, vals, _ in series)
        + "</title>",
    ]
    for t in model["ticks"]:
        parts.append(
            f"<line x1='{left}' x2='{w - right}' y1='{y(t):.1f}' y2='{y(t):.1f}' "
            f"stroke='{_GRID}' stroke-width='1'/>"
            f"<text x='{left - 6}' y='{y(t) + 4:.1f}' text-anchor='end' font-size='11' "
            f"fill='{_INK_MUTED}' style='font-variant-numeric:tabular-nums'>{pct_tick(t)}</text>"
        )
    parts.append(
        f"<line x1='{left}' x2='{w - right}' y1='{y(0):.1f}' y2='{y(0):.1f}' "
        f"stroke='{_REFERENCE_LINE}' stroke-width='1.5' stroke-dasharray='4 3'/>"
    )
    for i in sorted({0, (n - 1) // 2, n - 1}):
        anchor = "start" if i == 0 else "end" if i == n - 1 else "middle"
        parts.append(
            f"<text x='{x(i):.1f}' y='{h - 8}' text-anchor='{anchor}' font-size='11' "
            f"fill='{_INK_MUTED}'>{html.escape(_short_date(dates[i]))}</text>"
        )
    for _, vals, color in series:
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        parts.append(
            f"<polyline points='{points}' fill='none' stroke='{color}' "
            f"stroke-width='2' stroke-linejoin='round' stroke-linecap='round'/>"
        )
    label_ys = _nudge_apart([y(vals[-1]) for _, vals, _ in series], 14)
    for (name, vals, color), ly in zip(series, label_ys, strict=True):
        parts.append(
            f"<circle cx='{x(n - 1):.1f}' cy='{y(vals[-1]):.1f}' r='4' fill='{color}' "
            f"stroke='#fff' stroke-width='2'/>"
            f"<text x='{w - right + 10}' y='{ly + 4:.1f}' font-size='11' fill='{_INK}'>"
            f"{name} {vals[-1]:+.1f}%</text>"
        )
    parts.append("</svg>")
    legend_items = [
        f"<span style='display:inline-flex;align-items:center;gap:6px;margin-right:14px'>"
        f"<svg width='18' height='6'><line x1='0' x2='18' y1='3' y2='3' "
        f"stroke='{color}' stroke-width='2'{dash}/></svg>{label}</span>"
        for label, color, dash in [
            *((name, color, "") for name, _, color in series),
            ("Break-even", _REFERENCE_LINE, " stroke-dasharray='4 3'"),
        ]
    ]
    return (
        f"<div style='font-size:12px;color:{_INK};margin:6px 0'>"
        f"Return on invested capital — {''.join(legend_items)}</div>"
        f"<div style='overflow-x:auto'>{''.join(parts)}</div>"
    )


def _bar_chart_svg(d: dict[str, Any]) -> str:
    """Horizontal bars from a zero baseline (values may be negative), one
    hue, value labels in ink at the bar end."""
    bars = d.get("bars") or []
    if not bars:
        return ""
    unit = str(d.get("unit") or "")
    w, label_w, row_h, top = 640, 170, 30, 8
    values = [float(b.get("value") or 0.0) for b in bars]
    # Reserve room for the value text on whichever side of the bar it sits,
    # so a negative value never lands on top of its category label. ~6.8px
    # per character at 11px is a safe estimate for the email's sans stack.
    text_w = [
        6.8 * len(f"{v:+.1f}{unit}" + (f" {b.get('note')}" if b.get("note") else ""))
        for b, v in zip(bars, values, strict=True)
    ]
    neg_w = max((t for t, v in zip(text_w, values, strict=True) if v < 0), default=0.0)
    pos_w = max((t for t, v in zip(text_w, values, strict=True) if v >= 0), default=0.0)
    plot_l = label_w + 10 + (neg_w + 8 if neg_w else 0)
    plot_r = w - (pos_w + 8 if pos_w else 4)
    lo, hi = min(0.0, *values), max(0.0, *values)
    span = (hi - lo) or 1.0

    def x(v: float) -> float:
        return plot_l + (plot_r - plot_l) * (v - lo) / span

    h = top * 2 + row_h * len(bars)
    zero = x(0.0)
    parts = [
        f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px;display:block' "
        f"role='img' aria-label='{html.escape(str(d.get('title') or 'Bar chart'))}'>",
        f"<line x1='{zero:.1f}' x2='{zero:.1f}' y1='{top - 4}' y2='{h - top + 4}' "
        f"stroke='{_INK_MUTED}' stroke-width='1'/>",
    ]
    for i, (b, v) in enumerate(zip(bars, values, strict=True)):
        cy = top + row_h * i + row_h / 2
        bar_h = 14
        x0, x1 = sorted((zero, x(v)))
        length = x1 - x0
        r = min(4.0, length)
        # Square at the baseline, rounded at the data end.
        if v >= 0:
            path = (
                f"M{x0:.1f},{cy - bar_h / 2:.1f} H{x1 - r:.1f} "
                f"Q{x1:.1f},{cy - bar_h / 2:.1f} {x1:.1f},{cy - bar_h / 2 + r:.1f} "
                f"V{cy + bar_h / 2 - r:.1f} Q{x1:.1f},{cy + bar_h / 2:.1f} "
                f"{x1 - r:.1f},{cy + bar_h / 2:.1f} H{x0:.1f} Z"
            )
            value_x, anchor = x1 + 6, "start"
        else:
            path = (
                f"M{x1:.1f},{cy - bar_h / 2:.1f} H{x0 + r:.1f} "
                f"Q{x0:.1f},{cy - bar_h / 2:.1f} {x0:.1f},{cy - bar_h / 2 + r:.1f} "
                f"V{cy + bar_h / 2 - r:.1f} Q{x0:.1f},{cy + bar_h / 2:.1f} "
                f"{x0 + r:.1f},{cy + bar_h / 2:.1f} H{x1:.1f} Z"
            )
            value_x, anchor = x0 - 6, "end"
        note = html.escape(str(b.get("note") or ""))
        parts.append(
            f"<g><title>{html.escape(str(b.get('label')))}: {v:+.1f}{unit} {note}</title>"
            f"<text x='{label_w}' y='{cy + 4:.1f}' text-anchor='end' font-size='12' "
            f"fill='{_INK}'>{html.escape(str(b.get('label')))}</text>"
            + (f"<path d='{path}' fill='{_SERIES_1}'/>" if length > 0 else "")
            + f"<text x='{value_x:.1f}' y='{cy + 4:.1f}' text-anchor='{anchor}' "
            f"font-size='11' fill='{_INK}'>{v:+.1f}{unit}"
            + (f" <tspan fill='{_INK_MUTED}'>{note}</tspan>" if note else "")
            + "</text></g>"
        )
    parts.append("</svg>")
    return f"<div style='overflow-x:auto'>{''.join(parts)}</div>"


_CATALYST_DIRECTION_STYLE: dict[str, tuple[str, str]] = {
    "positive": ("▲", "#166534"),
    "negative": ("▼", "#9c1010"),
    "uncertain": ("◆", "#9a5b00"),
}


def _catalysts_html(catalysts: list[dict[str, Any]]) -> str:
    """Dated forward-event table shared by pick and holding-review cards."""
    if not catalysts:
        return ""
    rows: list[str] = []
    for c in catalysts:
        direction = str(c.get("direction") or "uncertain")
        glyph, color = _CATALYST_DIRECTION_STYLE.get(
            direction, _CATALYST_DIRECTION_STYLE["uncertain"]
        )
        rows.append(
            "<tr>"
            f"<td style='white-space:nowrap;font-variant-numeric:tabular-nums;"
            f"color:#374151'>{html.escape(str(c.get('expected_date') or 'Date TBD'))}</td>"
            f"<td style='white-space:nowrap;color:{color};font-weight:700'>"
            f"{glyph} {html.escape(direction)}</td>"
            f"<td style='white-space:nowrap;color:#6b7280'>"
            f"{html.escape(str(c.get('impact') or ''))} impact</td>"
            f"<td>{html.escape(str(c.get('event') or ''))}</td>"
            "</tr>"
        )
    return (
        "<div><div style='font-size:11px;color:#0e7490;font-weight:600;"
        "text-transform:uppercase;letter-spacing:0.5px;margin:10px 0 4px'>"
        "Upcoming catalysts</div>"
        "<div style='overflow-x:auto'><table style='font-size:13px;margin:4px 0'>"
        "<tbody>" + "".join(rows) + "</tbody></table></div></div>"
    )


def _pick_card_html(d: dict[str, Any]) -> str:
    """Render a per-pick structured card. Uses pill badges for rank /
    conviction / fragility / allocation and stacks bull + bear prose."""
    ticker = html.escape(str(d.get("ticker", "")))
    one_liner = html.escape(str(d.get("one_liner") or ""))
    return (
        f"<div style='border:1px solid #e5e7eb;border-radius:8px;padding:16px;"
        f"margin:16px 0;background:#fff'>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;gap:10px;"
        f"margin-bottom:10px'>"
        f"<h2 style='margin:0;border:none;padding:0'>{ticker}</h2>"
        f"{_pick_pills_html(d)}"
        f"</div>"
        f"<div style='color:#374151;margin-bottom:6px'>{one_liner}</div>"
        + "".join(_pick_body_html(d))
        + "</div>"
    )


def _pick_pills_html(d: dict[str, Any]) -> str:
    """Rank, conviction, fragility, allocation and consensus badges."""
    rank = d.get("rank")
    conviction = d.get("conviction")
    fragility = d.get("fragility_rank")
    alloc_pct = d.get("allocation_pct")
    alloc_usd = d.get("allocation_usd")
    agreement_ratio = d.get("agreement_ratio")
    voting_providers = d.get("voting_providers")

    rank_html = (
        f"<span style='background:#1f2937;color:#fff;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:600'>#{rank}</span>"
        if rank is not None
        else ""
    )
    conv_color = _conviction_swatch(conviction if isinstance(conviction, int) else None)
    conv_html = (
        f"<span style='background:{conv_color};color:#fff;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:600'>"
        f"Conviction {conviction}/10</span>"
        if conviction is not None
        else ""
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

    agreement_html = ""
    if agreement_ratio is not None:
        n = len(voting_providers) if voting_providers else 0
        total = round(n / agreement_ratio) if agreement_ratio else n
        is_unanimous = agreement_ratio >= 0.999
        bg, fg, border = (
            ("#e6f4ea", "#1a7f37", "#1a7f37") if is_unanimous else ("#fff4e5", "#9a5b00", "#9a5b00")
        )
        providers = ", ".join(voting_providers or [])
        agreement_html = (
            f"<span title='{html.escape(providers)}' style='background:{bg};color:{fg};"
            f"border:1px solid {border};padding:3px 10px;border-radius:12px;"
            f"font-size:12px;font-weight:600'>Consensus {n}/{total}</span>"
        )
    return f"{rank_html}{conv_html}{frag_html}{alloc_html}{agreement_html}"


def _pick_body_html(d: dict[str, Any]) -> list[str]:
    """Bull and bear prose, catalysts, alternatives and sizing rationale."""
    bull = html.escape(str(d.get("bull_thesis") or ""))
    bear = html.escape(str(d.get("bear_case") or "")) if d.get("bear_case") else ""
    bet_on = html.escape(str(d.get("what_youre_betting_on") or ""))
    why_over = html.escape(str(d.get("why_over_alternatives") or ""))
    sector_concentration = html.escape(str(d.get("sector_concentration_check") or ""))
    most_fragile = html.escape(str(d.get("most_fragile_assumption") or ""))
    watch_metric = html.escape(str(d.get("watch_metric") or ""))
    alloc_rationale = html.escape(str(d.get("allocation_rationale") or ""))

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
            f"<div style='margin-top:4px;color:#374151'><b>Watch:</b> {watch_metric}</div>"
        )
    if d.get("catalysts"):
        sections_html.append(_catalysts_html(d["catalysts"]))
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
    return sections_html


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
            f"{pct:.1f}%" if pct is not None else (f"${usd:,.0f}" if usd is not None else "—")
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


_FACTOR_TILT_COLORS: dict[str, str] = {
    "growth": "#0e7490",
    "value": "#166534",
    "quality": "#4c1d95",
    "momentum": "#b45309",
    "low_vol": "#374151",
}
_FACTOR_TILT_LABELS: dict[str, str] = {
    "growth": "Growth",
    "value": "Value",
    "quality": "Quality",
    "momentum": "Momentum",
    "low_vol": "Low-vol",
}
_FACTOR_TILT_ORDER = ["growth", "value", "quality", "momentum", "low_vol"]


def _factor_tilt_bar_html(bucket: str, value: float) -> str:
    color = _FACTOR_TILT_COLORS.get(bucket, "#6b7280")
    label = _FACTOR_TILT_LABELS.get(bucket, bucket)
    pct = max(0.0, min(100.0, value))
    return (
        f"<div style='margin:4px 0'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:12px;color:#374151'>"
        f"<span>{html.escape(label)}</span><span>{pct:.0f}</span></div>"
        f"<div style='background:#eef0f3;border-radius:4px;height:8px;"
        f"overflow:hidden'>"
        f"<div style='width:{pct:.0f}%;background:{color};height:100%'>"
        f"</div></div></div>"
    )


def _factor_tilt_panel_html(d: dict[str, Any]) -> str:
    """Render portfolio-level + per-pick style factor tilt as horizontal
    meter bars — growth/value/quality/momentum/low_vol, each 0-100."""
    portfolio = d.get("portfolio") or {}
    picks = d.get("picks") or []
    if not portfolio and not picks:
        return ""
    parts: list[str] = []
    if portfolio:
        parts.append(
            "<div style='border:1px solid #e5e7eb;border-radius:8px;"
            "padding:12px 14px;margin-bottom:10px;background:#fafbfc'>"
            "<b style='font-size:13px;color:#111827'>"
            "Portfolio (average across picks)</b>"
        )
        for bucket in _FACTOR_TILT_ORDER:
            if bucket in portfolio:
                parts.append(_factor_tilt_bar_html(bucket, portfolio[bucket]))
        parts.append("</div>")
    for pick in picks:
        tilt = pick.get("tilt") or {}
        if not tilt:
            continue
        ticker = html.escape(str(pick.get("ticker") or ""))
        parts.append(
            "<div style='border:1px solid #e5e7eb;border-radius:8px;"
            "padding:10px 14px;margin:6px 0;background:#fff'>"
            f"<b style='font-size:13px;color:#111827'>{ticker}</b>"
        )
        for bucket in _FACTOR_TILT_ORDER:
            if bucket in tilt:
                parts.append(_factor_tilt_bar_html(bucket, tilt[bucket]))
        parts.append("</div>")
    return "".join(parts)


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
        strength_color = _theme_strength_color(strength if isinstance(strength, int) else None)
        strength_pill = (
            f"<span style='background:{strength_color};color:#fff;"
            f"padding:2px 10px;border-radius:10px;font-size:11px;"
            f"font-weight:700'>{strength}/10</span>"
            if isinstance(strength, int)
            else ""
        )
        trend_pill = (
            f"<span style='color:{glyph_color};font-weight:700;font-size:13px'>{glyph}</span>"
        )
        # Pack member tickers into a tight monospace strip.
        member_strip = ", ".join(
            f"<span style='font-family:ui-monospace,SFMono-Regular,"
            f"monospace;color:#374151'>{html.escape(str(t))}</span>"
            for t in members[:18]
        )
        if len(members) > 18:
            member_strip += f" <span style='color:#6b7280'>+{len(members) - 18} more</span>"
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


def _premortem_panel_html(d: dict[str, Any]) -> str:
    """Render the plan-level pre-mortem: overall verdict banner + per-failure
    cards with likelihood × severity badges + narrative + early warning."""
    verdict = str(d.get("overall_verdict") or "proceed_with_caveat")
    summary = str(d.get("summary") or "")
    failures = d.get("failures") or []
    if not failures and not summary:
        return ""

    vp = _VERDICT_PALETTE_PREMORTEM.get(verdict, _VERDICT_PALETTE_PREMORTEM["proceed_with_caveat"])
    verdict_label = {
        "proceed_as_planned": "PROCEED AS PLANNED",
        "proceed_with_caveat": "PROCEED WITH CAVEAT",
        "reconsider": "RECONSIDER",
    }.get(verdict, verdict.upper())

    parts: list[str] = []
    parts.append(
        f"<div style='background:{vp['bg']};color:{vp['fg']};"
        f"border-left:4px solid {vp['border']};padding:12px 16px;"
        f"border-radius:6px;margin:12px 0;font-weight:600'>"
        f"Verdict: {html.escape(verdict_label)}</div>"
    )
    if summary:
        parts.append(f"<p style='color:#374151;margin:8px 0 16px'>{html.escape(summary)}</p>")
    for f in failures:
        likelihood = str(f.get("likelihood") or "medium").lower()
        severity = str(f.get("severity") or "moderate").lower()
        trig = html.escape(str(f.get("triggering_action") or ""))
        narrative = html.escape(str(f.get("failure_narrative") or ""))
        warning = html.escape(str(f.get("early_warning") or ""))
        like_color = _LIKELIHOOD_COLOR.get(likelihood, "#6b7280")
        sev_color = _SEVERITY_COLOR.get(severity, "#6b7280")
        parts.append(
            f"<div style='border:1px solid #e5e7eb;border-radius:8px;"
            f"padding:14px 16px;margin:10px 0;background:#fff'>"
            f"<div style='display:flex;flex-wrap:wrap;gap:8px;"
            f"margin-bottom:6px;align-items:center'>"
            f"<span style='background:{like_color};color:#fff;padding:2px 10px;"
            f"border-radius:10px;font-size:11px;font-weight:700'>"
            f"Likelihood: {html.escape(likelihood)}</span>"
            f"<span style='background:{sev_color};color:#fff;padding:2px 10px;"
            f"border-radius:10px;font-size:11px;font-weight:700'>"
            f"Severity: {html.escape(severity)}</span>"
            f"</div>"
            f"<div style='font-size:11px;color:#6b7280;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px'>"
            f"Triggering action</div>"
            f"<div style='color:#1f2937;font-style:italic;margin-bottom:8px'>"
            f"{trig}</div>"
            f"<div style='color:#1f2937;margin-bottom:8px'>{narrative}</div>"
            f"<div style='background:#fff4e0;border-left:3px solid #e89c00;"
            f"padding:6px 10px;color:#8a4a00;font-size:13px'>"
            f"<b>Early warning:</b> {warning}</div>"
            f"</div>"
        )
    return "".join(parts)


def _holding_review_card_html(d: dict[str, Any]) -> str:
    """Render a per-holding review (HoldingReview schema) as a structured
    card: ticker + verdict pill + confidence pill + position context,
    then labeled forward outlook / reasoning / what-would-change-your-mind
    paragraphs, with tax lot plan + wash-sale notice when present.

    Replaces the monospace `preformatted` dump that used to render
    HoldingReview.full_text verbatim."""
    ticker = html.escape(str(d.get("ticker", "")))
    position_context = html.escape(str(d.get("position_context") or ""))
    pos_html = (
        f"<div style='color:#6b7280;font-size:13px;margin-top:6px'>{position_context}</div>"
        if position_context
        else ""
    )

    return (
        f"<div style='border:1px solid #e5e7eb;border-radius:8px;"
        f"padding:16px;margin:16px 0;background:#fff'>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;"
        f"gap:10px;margin-bottom:0'>"
        f"<h2 style='margin:0;border:none;padding:0;font-family:"
        f"ui-monospace,SFMono-Regular,monospace'>{ticker}</h2>"
        + "".join(_review_pills_html(d))
        + f"</div>{pos_html}"
        + "".join(_review_body_html(d))
        + "</div>"
    )


def _review_pills_html(d: dict[str, Any]) -> list[str]:
    """Verdict, conviction and trim-size badges."""
    verdict = str(d.get("verdict") or "HOLD").upper()
    confidence = d.get("confidence")
    trim_pct = d.get("trim_pct")
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
    return pills


def _review_body_html(d: dict[str, Any]) -> list[str]:
    """Outlook, reasoning, tax-lot plan, catalysts, wash-sale notice and
    what would change the reviewer's mind — only the ones present."""
    forward_outlook = html.escape(str(d.get("forward_outlook") or ""))
    reasoning = html.escape(str(d.get("reasoning") or ""))
    tax_lot_plan = d.get("tax_lot_plan") or []
    what_change = html.escape(str(d.get("what_would_change_mind") or ""))
    wash_sale_notice = d.get("wash_sale_notice")

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
            f"<li style='margin:3px 0'>{html.escape(str(line))}</li>" for line in tax_lot_plan
        )
        body_parts.append(
            f"<div style='margin-top:12px'>"
            f"<div style='font-size:11px;color:#4c1d95;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px'>"
            f"Tax lot plan</div>"
            f"<ul style='margin:4px 0 0 18px;padding:0;color:#1f2937'>"
            f"{items}</ul></div>"
        )

    if d.get("catalysts"):
        body_parts.append(_catalysts_html(d["catalysts"]))

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
    return body_parts


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


def _render_premium_income(data: dict) -> str:
    rows_html = "".join(
        f"<tr>"
        f"<td>{r['ticker']}</td>"
        f"<td>{r.get('account', '—')}</td>"
        f"<td>${r['strike']:,.2f}</td>"
        f"<td>{r['expiry']}</td>"
        f"<td>{r['contracts']}</td>"
        f"<td>${r['premium_usd']:,.0f}</td>"
        f"<td>{r['delta']:.2f}</td>"
        f"<td>{r['assignment_pct']}%</td>"
        f"</tr>"
        for r in data.get("rows") or []
    )
    return (
        '<div style="border:1px solid #d1d5db; padding:12px; '
        'margin:16px 0; background:#f0fdfa;">'
        '<h3 style="margin:0 0 8px 0;">Premium Income</h3>'
        '<table style="width:100%; border-collapse:collapse;">'
        '<thead><tr style="text-align:left; border-bottom:1px solid #d1d5db;">'
        "<th>Ticker</th><th>Account</th><th>Strike</th><th>Expiry</th><th>Qty</th>"
        "<th>Premium</th><th>Δ</th><th>Assign %</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        f'<p style="margin:8px 0 0 0;">'
        f"Gross premium: <strong>${data.get('gross_premium_usd', 0):,.0f}</strong>"
        f" &nbsp; Slippage buffer (10%): -${data.get('slippage_buffer_usd', 0):,.0f}"
        f" &nbsp; Deployable: <strong>${data.get('deployable_premium_usd', 0):,.0f}</strong>"
        f"</p>"
        "</div>"
    )


def _render_round_lot_coverage(data: dict) -> str:
    rows = data.get("rows") or []
    rows_html = "".join(
        f"<tr>"
        f"<td>{r['ticker']}</td>"
        f"<td>{r['shares']}</td>"
        f"<td>{r['round_lots']} ({r['round_lot_shares']})</td>"
        f"<td>{r['stub_shares']}</td>"
        f"<td>${r['stub_dollar_value']:,.0f}</td>"
        f"<td>${r['to_next_lot_cost']:,.0f}</td>"
        f"</tr>"
        for r in rows
    )
    return (
        '<div style="border:1px solid #d1d5db; padding:12px; '
        'margin:16px 0; background:#fefce8;">'
        '<h3 style="margin:0 0 8px 0;">Round-Lot Coverage</h3>'
        '<table style="width:100%; border-collapse:collapse;">'
        '<thead><tr style="text-align:left; border-bottom:1px solid #d1d5db;">'
        "<th>Position</th><th>Shares</th><th>Round Lots</th>"
        "<th>Stub</th><th>Stub $</th><th>To-next-lot</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        f'<p style="margin:8px 0 0 0;">'
        f"Stub pool total: <strong>${data.get('stub_pool_total_usd', 0):,.0f}</strong>"
        f"</p>"
        "</div>"
    )


def _render_premium_deployment(data: dict) -> str:
    deps_html = "".join(
        f"<li>{d['action']} <strong>{d['ticker']}</strong> {d['sizing']}</li>"
        for d in (data.get("deployments") or [])
    )
    stub_usd = data.get("stub_consolidation_usd", 0)
    stub_row = f"<tr><td>Stub consolidation:</td><td>${stub_usd:,.0f}</td></tr>" if stub_usd else ""
    return (
        '<div style="border:1px solid #d1d5db; padding:12px; '
        'margin:16px 0; background:#eff6ff;">'
        '<h3 style="margin:0 0 8px 0;">Premium → Deployment</h3>'
        '<table style="margin:0;">'
        f"<tr><td>Deployable premium:</td>"
        f"<td>${data.get('deployable_premium_usd', 0):,.0f}</td></tr>"
        f"<tr><td>Existing cash:</td>"
        f"<td>${data.get('existing_cash_usd', 0):,.0f}</td></tr>"
        f"{stub_row}"
        f"<tr><td><strong>Total dry powder:</strong></td>"
        f"<td><strong>${data.get('total_dry_powder_usd', 0):,.0f}</strong></td></tr>"
        "</table>"
        f'<ul style="margin:8px 0 0 16px;">{deps_html}</ul>'
        "</div>"
    )


def render_html_email(sections: list[Section], chart_cids: dict[str, str]) -> str:
    parts: list[str] = [_HTML_HEAD]
    for s in sections:
        render = _HTML_SECTION_RENDERERS.get(s.kind)
        if render is not None:
            parts.append(render(s, chart_cids))
    parts.append("</body></html>")
    return "".join(parts)


def _image_html(s: Section, chart_cids: dict[str, str]) -> str:
    cid = chart_cids.get(s.image_ticker) if s.image_ticker else None
    return f"<img src='cid:{cid}' alt='{html.escape(s.image_ticker)} chart' />" if cid else ""


def _table_html(s: Section, chart_cids: dict[str, str]) -> str:
    if not (s.table_header and s.table_rows):
        return ""
    parts = ["<table><thead><tr>"]
    for h in s.table_header:
        parts.append(f"<th>{html.escape(h)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in s.table_rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{html.escape(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _status_banner_html(s: Section, chart_cids: dict[str, str]) -> str:
    cs = _STATUS_COLORS.get(s.status, _STATUS_COLORS["UNKNOWN"])
    return (
        f"<div class='banner' style='background:{cs['bg']};"
        f"color:{cs['fg']};border-left-color:{cs['border']}'>"
        f"{html.escape(s.text)}"
        "</div>"
    )


def _metric_strip_html(s: Section, chart_cids: dict[str, str]) -> str:
    if not s.metrics:
        return ""
    parts = ["<div class='metrics'>"]
    for label, value in s.metrics:
        parts.append(
            f"<div class='metric'>"
            f"<div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{html.escape(value)}</div>"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _holdings_dashboard_html(s: Section, chart_cids: dict[str, str]) -> str:
    if not s.holdings:
        return ""
    parts = [
        "<table class='dashboard'><thead><tr>"
        "<th>Ticker</th><th>Verdict</th><th>Conf</th>"
        "<th>P/L</th><th>Sector</th><th>Forward note</th>"
        "</tr></thead><tbody>"
    ]
    for h_row in s.holdings:
        verdict = (h_row.get("verdict") or "HOLD").upper()
        conf = h_row.get("confidence")
        pnl = h_row.get("pnl_pct")
        pnl_str = f"{pnl:+.1f}%" if isinstance(pnl, (int, float)) else "—"
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
    return "".join(parts)


def _data_html(build: Callable[[Any], str]) -> Callable[[Section, dict[str, str]], str]:
    """A renderer for a kind whose builder takes the section's data."""
    return lambda s, chart_cids: build(s.data) if s.data else ""


# One renderer per SectionKind: (section, chart_cids) -> HTML.
_HTML_SECTION_RENDERERS: dict[str, Callable[[Section, dict[str, str]], str]] = {
    "heading": lambda s, c: f"<h{s.level}>{html.escape(s.text)}</h{s.level}>",
    "para": lambda s, c: f"<p>{html.escape(s.text)}</p>",
    "preformatted": lambda s, c: f"<pre>{html.escape(s.text)}</pre>",
    "blockquote": lambda s, c: f"<blockquote>{html.escape(s.text)}</blockquote>",
    "image": _image_html,
    "table": _table_html,
    "status_banner": _status_banner_html,
    "metric_strip": _metric_strip_html,
    "holdings_dashboard": _holdings_dashboard_html,
    "sector_pie": lambda s, c: _svg_pie(s.pie_data) if s.pie_data else "",
    "pick_card": _data_html(_pick_card_html),
    "allocation_table": _data_html(_allocation_table_html),
    "rebalance_action_table": _data_html(_rebalance_action_table_html),
    "holding_review_card": _data_html(_holding_review_card_html),
    "market_themes_panel": _data_html(_market_themes_panel_html),
    "premortem_panel": _data_html(_premortem_panel_html),
    "factor_tilt_panel": _data_html(_factor_tilt_panel_html),
    "equity_curve": _data_html(_equity_curve_svg),
    "bar_chart": _data_html(_bar_chart_svg),
    "premium_income": _data_html(_render_premium_income),
    "round_lot_coverage": _data_html(_render_round_lot_coverage),
    "premium_deployment": _data_html(_render_premium_deployment),
    "page_break": lambda s, c: "<hr/>",
}
