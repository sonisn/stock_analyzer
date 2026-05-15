"""Section IR + parsing helpers + shared visual palettes for the report.

The renderers (`report_html.py` and `report_pdf.py`) both build off the
same `Section` list, so this module owns the schema, the LLM-output
parsers (verdict / confidence / status / actions), and any palette
constants / helper functions shared between HTML and PDF.

The `report.py` public surface re-exports from this module so existing
callers (`cli/discover.py`, `cli/rebalance.py`, tests) keep working.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from rich.console import Console
from rich.table import Table as RichTable

from ..models.reports import Section

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
    from ..models.llm import RankerOutput
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
    from ..models.llm import HoldingReview
    if isinstance(review, HoldingReview):
        return review.verdict
    if not review or not isinstance(review, str):
        return "HOLD"
    m = _VERDICT_RE.search(review)
    return m.group(1).upper() if m else "HOLD"


def parse_confidence(review: object) -> int | None:
    """Return the 1-10 confidence integer.

    Accepts a `HoldingReview` (preferred) OR a free-text review."""
    from ..models.llm import HoldingReview
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
    from ..models.rebalance import RebalancePlan
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
    from ..models.rebalance import RebalancePlan
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
# Fragility-rank visual palette — 1 = most fragile (red), 5 = most resilient (green).
_FRAGILITY_COLORS: dict[int, dict[str, str]] = {
    1: {"bg": "#fde4e4", "fg": "#9c1010", "border": "#d73030"},
    2: {"bg": "#fde8d3", "fg": "#a3550b", "border": "#e89c00"},
    3: {"bg": "#fff4e0", "fg": "#8a4a00", "border": "#e89c00"},
    4: {"bg": "#e8f4f8", "fg": "#0c5e7c", "border": "#3b8fde"},
    5: {"bg": "#e6f4ea", "fg": "#0e6432", "border": "#1f9d55"},
}
# Trend arrows for the market-themes panel (HTML + PDF).
_TREND_GLYPHS: dict[str, tuple[str, str]] = {
    "up":   ("▲", "#0e6432"),
    "flat": ("●", "#6b7280"),
    "down": ("▼", "#9c1010"),
}
# Pre-mortem palettes — verdict banner + per-failure pills.
_VERDICT_PALETTE_PREMORTEM = {
    "proceed_as_planned":  {"bg": "#e6f4ea", "fg": "#0e6432", "border": "#1f9d55"},
    "proceed_with_caveat": {"bg": "#fff4e0", "fg": "#8a4a00", "border": "#e89c00"},
    "reconsider":          {"bg": "#fde4e4", "fg": "#9c1010", "border": "#d73030"},
}
_LIKELIHOOD_COLOR = {"high": "#9c1010", "medium": "#a3550b", "low": "#0e6432"}
_SEVERITY_COLOR = {"severe": "#9c1010", "moderate": "#a3550b", "mild": "#0e6432"}


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


# --- sections (unified IR for HTML + PDF) -----------------------------------


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
    from ..models.llm import RankerOutput, RedTeamOutput, SizerOutput
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
    from ..models.llm import MarketThemes
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
