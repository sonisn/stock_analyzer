"""Convert the analyst plain-text report into an email-friendly HTML body."""

from __future__ import annotations

import html
import re

from ..models.reports import TickerSection

TICKER_HEADER_RE = re.compile(r"^([A-Z][A-Z0-9.\-]{0,9})\s+-\s+(.+)$")
LABEL_LINE_RE = re.compile(r"^([A-Z][A-Za-z/0-9 &]+):\s*(.*)$")
DASHES_SPLIT_RE = re.compile(r"-{20,}")
PREAMBLE_LINE_RE = re.compile(
    r"^\s*(I'?ll\b|I will\b|Let me\b|Let's\b|Now let\b|Now I\b|I have\b"
    r"|Here is\b|Here's\b|I'?m\b)",
    re.IGNORECASE,
)
SENTIMENT_PREFIX = "Social/Economic Sentiment:"


def format_html(
    report: str,
    *,
    title: str = "Portfolio Analysis",
    chart_cids: dict[str, str] | None = None,
) -> str:
    """Render the analyst report as HTML.

    `chart_cids` maps ticker symbol → CID (without angle brackets). When present,
    the matching ticker section gets a <img src="cid:..."> rendered above the
    metrics table. CIDs must be attached as inline images by the SMTP layer.
    """
    sentiment, tickers = _parse(report)
    body_parts: list[str] = []
    if sentiment:
        body_parts.append(_render_sentiment(sentiment))
    body_parts.extend(_render_ticker(t, chart_cids or {}) for t in tickers)
    return _wrap_html(title, "\n".join(body_parts))


def _parse(report: str) -> tuple[str, list[TickerSection]]:
    idx = report.find(SENTIMENT_PREFIX)
    cleaned = report[idx:] if idx != -1 else report

    blocks = [b.strip() for b in DASHES_SPLIT_RE.split(cleaned) if b.strip()]

    sentiment = ""
    tickers: list[TickerSection] = []
    for block in blocks:
        if block.startswith(SENTIMENT_PREFIX):
            text = block[len(SENTIMENT_PREFIX):].strip()
            sentiment = _strip_preamble_lines(text)
        else:
            section = _parse_ticker_block(block)
            if section is not None:
                tickers.append(section)
    return sentiment, tickers


def _strip_preamble_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and PREAMBLE_LINE_RE.match(lines[0]):
        lines.pop(0)
    while lines and PREAMBLE_LINE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _parse_ticker_block(block: str) -> TickerSection | None:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]

    header_idx = next(
        (i for i, line in enumerate(lines) if TICKER_HEADER_RE.match(line)),
        None,
    )
    if header_idx is None:
        return None

    match = TICKER_HEADER_RE.match(lines[header_idx])
    assert match is not None
    section = TickerSection(symbol=match.group(1), name=match.group(2))

    current_label: str | None = None
    current_value: list[str] = []
    for line in lines[header_idx + 1 :]:
        if PREAMBLE_LINE_RE.match(line):
            continue
        label_match = LABEL_LINE_RE.match(line)
        if label_match:
            if current_label is not None:
                section.fields.append(
                    (current_label, " ".join(current_value).strip())
                )
            current_label = label_match.group(1).strip()
            current_value = [label_match.group(2).strip()]
        else:
            current_value.append(line.strip())
    if current_label is not None:
        section.fields.append((current_label, " ".join(current_value).strip()))
    return section


def _render_sentiment(text: str) -> str:
    return (
        '<section class="sentiment">'
        "<h2>Social / Economic Sentiment</h2>"
        f"<p>{html.escape(text)}</p>"
        "</section>"
    )


def _render_ticker(t: TickerSection, chart_cids: dict[str, str]) -> str:
    rows = "".join(
        f'<tr><th>{html.escape(label)}</th>'
        f'<td>{html.escape(value)}</td></tr>'
        for label, value in t.fields
    )
    cid = chart_cids.get(t.symbol)
    chart_img = (
        f'<img class="chart" src="cid:{html.escape(cid)}" '
        f'alt="{html.escape(t.symbol)} chart">'
        if cid
        else ""
    )
    return (
        '<section class="ticker">'
        f'<h2>{html.escape(t.symbol)} '
        f'<span class="company">{html.escape(t.name)}</span></h2>'
        f"{chart_img}"
        f"<table>{rows}</table>"
        "</section>"
    )


def format_insider_html(report: str, *, title: str = "Insider & Political Trades") -> str:
    """Render the insider/political trade report as HTML.

    Expected input shape:
        === HEADER LINE ===
        Section Name:
        - bullet
        - bullet
        Other Section:
        - bullet
    """
    lines = [line.rstrip() for line in report.splitlines()]
    parts: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            continue
        # Header line wrapped in === ... ===
        if stripped.startswith("===") and stripped.endswith("==="):
            close_list()
            text = stripped.strip("= ").strip()
            parts.append(f"<h2>{html.escape(text)}</h2>")
        # Bullet
        elif stripped.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{html.escape(stripped[2:])}</li>")
        # Section label (line ending with ":")
        elif stripped.endswith(":"):
            close_list()
            parts.append(f"<h3>{html.escape(stripped[:-1])}</h3>")
        else:
            close_list()
            parts.append(f"<p>{html.escape(stripped)}</p>")
    close_list()
    return _wrap_html(title, "\n".join(parts))


def _wrap_html(title: str, body: str) -> str:
    style = (
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,"
        "sans-serif;color:#1a1a1a;max-width:720px;margin:24px auto;padding:0 16px;"
        "line-height:1.55;}"
        "h1{font-size:22px;margin:0 0 16px;}"
        "h2{font-size:18px;margin:28px 0 8px;padding-bottom:6px;"
        "border-bottom:1px solid #e5e7eb;}"
        ".ticker h2 .company{font-size:14px;color:#6b7280;font-weight:400;}"
        ".sentiment p{background:#f9fafb;padding:12px 14px;"
        "border-left:3px solid #4f46e5;border-radius:4px;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{padding:6px 10px;text-align:left;vertical-align:top;"
        "border-bottom:1px solid #f3f4f6;font-size:14px;}"
        "th{width:140px;color:#6b7280;font-weight:600;}"
        "h3{font-size:15px;margin:18px 0 6px;color:#374151;}"
        "ul{margin:6px 0 14px 20px;padding:0;}"
        "li{margin:3px 0;font-size:14px;}"
        "img.chart{display:block;max-width:100%;height:auto;margin:8px 0 14px;"
        "border:1px solid #1f2937;border-radius:6px;background:#111;}"
    )
    return (
        "<!doctype html>"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        f"<style>{style}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
    )
