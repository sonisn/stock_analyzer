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
from ..models.track_record import PickReturn, TrackRecord

# --- pick / ticker block parsing --------------------------------------------

_PICK_RE = re.compile(r"^PICK\s+(\d+):\s+([A-Z][A-Z.\-]{0,5})\s+[—–-]\s+(.+)$", re.MULTILINE)
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
_CONFIDENCE_RE = re.compile(r"^Confidence\s*\(1-10\):\s*(\d+)", re.MULTILINE | re.IGNORECASE)
_STATUS_RE = re.compile(r"^Status:\s*(NO ACTION RECOMMENDED|ACTION RECOMMENDED)", re.MULTILINE)
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
    return [(m.group(1), m.group(2)) for m in _ACTION_RE.finditer(rebalance_text_or_plan)]


# --- visual constants -------------------------------------------------------

# Color palette for verdict/action badges. Used by both HTML (hex CSS) and
# PDF (ReportLab HexColor) renderers so they look identical.
_VERDICT_COLORS = {
    "HOLD": {"bg": "#e8f4f8", "fg": "#0c5e7c", "border": "#3b8fde"},
    "TRIM": {"bg": "#fff4e0", "fg": "#a36500", "border": "#e89c00"},
    "SELL": {"bg": "#fde4e4", "fg": "#9c1010", "border": "#d73030"},
    "ADD": {"bg": "#ece8fb", "fg": "#4c1d95", "border": "#7c3aed"},
    "BUY": {"bg": "#e6f4ea", "fg": "#0e6432", "border": "#1f9d55"},
}
_STATUS_COLORS = {
    "NO_ACTION": {"bg": "#e6f4ea", "fg": "#0e6432", "border": "#1f9d55"},
    "ACTION": {"bg": "#fff4e0", "fg": "#8a4a00", "border": "#e89c00"},
    "UNKNOWN": {"bg": "#f0f0f0", "fg": "#444", "border": "#888"},
}
# Categorical palette for sector pie slices.
_PIE_PALETTE = [
    "#3b8fde",
    "#1f9d55",
    "#e89c00",
    "#d73030",
    "#7c3aed",
    "#0891b2",
    "#65a30d",
    "#dc2626",
    "#ea580c",
    "#0284c7",
    "#16a34a",
    "#a16207",
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
    "up": ("▲", "#0e6432"),
    "flat": ("●", "#6b7280"),
    "down": ("▼", "#9c1010"),
}
# Pre-mortem palettes — verdict banner + per-failure pills.
_VERDICT_PALETTE_PREMORTEM = {
    "proceed_as_planned": {"bg": "#e6f4ea", "fg": "#0e6432", "border": "#1f9d55"},
    "proceed_with_caveat": {"bg": "#fff4e0", "fg": "#8a4a00", "border": "#e89c00"},
    "reconsider": {"bg": "#fde4e4", "fg": "#9c1010", "border": "#d73030"},
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


# Fixed reason templates emitted by screen.py::passes_hard_filter /
# passes_trend_gate — matched by prefix since the numeric values differ per
# ticker. Order matters: first match wins, so a ticker missing both
# fundamentals and technicals still gets one clear primary reason instead of
# the two "no ... data" strings colliding with the same prefix.
_REJECT_REASON_LABELS: list[tuple[str, str]] = [
    ("no fundamentals data", "No fundamentals data"),
    ("no technicals data", "No technicals data"),
    ("market_cap=", "Market cap too small"),
    ("revenue_growth=", "Revenue growth too slow"),
    ("operating_cash_flow=", "Negative operating cash flow"),
    ("debt_to_equity=", "Too much debt"),
    ("price not above 200DMA", "Below 200-day average"),
    ("50DMA not above 200DMA", "No moving-average uptrend"),
    ("rs_6mo=", "Weak 6-month relative strength"),
    ("52w drawdown", "Too far below 52-week high"),
]


def _primary_reject_reason(reasons: list[str]) -> str:
    """First (most upstream) hard-filter failure, mapped to a short label
    for grouping — the raw reason strings embed per-ticker numbers, so they
    can't be grouped on directly."""
    if not reasons:
        return "Unknown"
    first = reasons[0]
    for prefix, label in _REJECT_REASON_LABELS:
        if first.startswith(prefix):
            return label
    return first[:40]


_DIRECTION_LABELS = {"buy": "Buy", "hold": "Hold", "trim": "Trim", "sell": "Sell"}
_MAX_DECISION_ROWS = 12


def _pct_or_dash(v: float | None) -> str:
    return f"{v:+.1f}%" if v is not None else "—"


def _decision_row(p: PickReturn) -> list[str]:
    return [
        _DIRECTION_LABELS.get(p.direction, p.direction),
        p.ticker,
        p.pick_date,
        _pct_or_dash(p.pick_return_pct),
        _pct_or_dash(p.spy_return_pct),
        _pct_or_dash(p.alpha_pct),
        _pct_or_dash(p.beta_adjusted_alpha_pct),
    ]


def append_track_record_section(
    sections: list[Section],
    record: TrackRecord | None,
    fallback_block: str = "",
) -> None:
    """'Track record' — headline sentence, alpha-by-call bar chart, per-horizon
    stats table, provider/model breakdown, scored decisions, pending and
    unmeasurable decisions. Falls back to the preformatted prompt block when
    no structured record is available (legacy callers)."""
    if not isinstance(record, TrackRecord):
        if fallback_block:
            sections.append(Section(kind="heading", text="Track record", level=2))
            sections.append(Section(kind="preformatted", text=fallback_block))
        return
    if record.n_picks_total == 0:
        return

    sections.append(Section(kind="heading", text="Track record", level=2))
    _track_record_headline(sections, record)
    _track_record_tables(sections, record)
    _scored_calls_table(sections, record)
    _pending_calls_table(sections, record)
    _unscorable_calls_note(sections, record)


def _track_record_headline(sections: list[Section], record: TrackRecord) -> None:
    """One sentence on the mature calls, and alpha by call as a bar chart."""
    from .track_record import format_track_record_summary

    horizon = record.reported_horizon_days
    by_direction = [
        ("buy", record.buy_stats),
        ("hold", record.hold_stats),
        ("trim", record.trim_stats),
        ("sell", record.sell_stats),
    ]
    if record.n_mature == 0:
        sections.append(Section(kind="para", text=format_track_record_summary(record)))
    else:
        beta_adj = [
            (s.mean_beta_adjusted_alpha_pct, s.n_beta_adjusted)
            for _, s in by_direction
            if s.mean_beta_adjusted_alpha_pct is not None and s.n_beta_adjusted
        ]
        beta_bit = ""
        if beta_adj:
            n_beta = sum(n for _, n in beta_adj)
            mean_beta = sum(v * n for v, n in beta_adj) / n_beta
            beta_bit = f" ({mean_beta:+.1f}% after removing market beta)"
        sections.append(
            Section(
                kind="para",
                text=(
                    f"Over a {horizon}-day horizon, {record.n_mature} scored calls averaged "
                    f"{_pct_or_dash(record.mean_alpha_pct)} alpha vs SPY{beta_bit}: "
                    f"{record.winners} right, {record.losers} wrong, {record.flats} flat. "
                    f"Positive alpha means the call was right — buys and holds beat SPY, "
                    f"trims and sells lagged it."
                ),
            )
        )
        bars = [
            {
                "label": f"{_DIRECTION_LABELS[d]} · {s.n_mature} scored",
                "value": s.mean_alpha_pct,
                "note": (
                    f"beta-adj {s.mean_beta_adjusted_alpha_pct:+.1f}%"
                    if s.mean_beta_adjusted_alpha_pct is not None
                    else ""
                ),
            }
            for d, s in by_direction
            if s.n_mature and s.mean_alpha_pct is not None
        ]
        if bars:
            sections.append(
                Section(
                    kind="bar_chart",
                    data={
                        "title": f"Mean alpha vs SPY by call, {horizon}d",
                        "unit": "%",
                        "bars": bars,
                    },
                )
            )


def _track_record_tables(sections: list[Section], record: TrackRecord) -> None:
    """Per-horizon stats by call, and buy alpha by model and provider."""
    stat_rows: list[list[str]] = []
    breakdown_rows: list[list[str]] = []
    for h in record.horizons:
        for d, s in [
            ("buy", h.buy_stats),
            ("hold", h.hold_stats),
            ("trim", h.trim_stats),
            ("sell", h.sell_stats),
        ]:
            if not s.n_mature:
                continue
            stat_rows.append(
                [
                    f"{h.horizon_days}d",
                    _DIRECTION_LABELS[d],
                    str(s.n_mature),
                    _pct_or_dash(s.mean_alpha_pct),
                    _pct_or_dash(s.mean_beta_adjusted_alpha_pct),
                    f"{s.winners}/{s.losers}/{s.flats}",
                    f"{s.sharpe:.2f}" if s.sharpe is not None else "n/a",
                ]
            )
        for m in h.model_breakdown:
            breakdown_rows.append(
                [
                    f"{h.horizon_days}d",
                    f"Model: {m.opus_model}",
                    str(m.n_mature),
                    _pct_or_dash(m.mean_alpha_pct),
                    f"{m.sharpe:.2f}" if m.sharpe is not None else "n/a",
                ]
            )
        for p in h.provider_breakdown:
            breakdown_rows.append(
                [
                    f"{h.horizon_days}d",
                    f"Provider: {p.provider}",
                    str(p.n_mature),
                    _pct_or_dash(p.mean_alpha_pct),
                    f"{p.sharpe:.2f}" if p.sharpe is not None else "n/a",
                ]
            )
    if stat_rows:
        sections.append(
            Section(
                kind="table",
                table_header=[
                    "Horizon",
                    "Call",
                    "Scored",
                    "Mean alpha",
                    "Beta-adj alpha",
                    "W/L/F",
                    "Sharpe",
                ],
                table_rows=stat_rows,
            )
        )
    if breakdown_rows:
        sections.append(Section(kind="heading", text="Buy alpha by model and provider", level=3))
        sections.append(
            Section(
                kind="table",
                table_header=["Horizon", "Source", "Picks", "Mean alpha", "Sharpe"],
                table_rows=breakdown_rows,
            )
        )


def _scored_calls_table(sections: list[Section], record: TrackRecord) -> None:
    """Scored calls best to worst, capped at the best and worst few."""
    horizon = record.reported_horizon_days
    scored = sorted(
        (p for p in record.picks if p.alpha_pct is not None),
        key=lambda p: p.alpha_pct or 0.0,
        reverse=True,
    )
    if scored:
        half = _MAX_DECISION_ROWS // 2
        shown = scored if len(scored) <= _MAX_DECISION_ROWS else scored[:half] + scored[-half:]
        title = f"Scored calls, {horizon}d, best to worst"
        if len(shown) < len(scored):
            title += f" ({half} best and {half} worst of {len(scored)})"
        sections.append(Section(kind="heading", text=title, level=3))
        sections.append(
            Section(
                kind="table",
                table_header=["Call", "Ticker", "Date", "Return", "SPY", "Alpha", "Beta-adj"],
                table_rows=[_decision_row(p) for p in shown],
            )
        )


def _pending_calls_table(sections: list[Section], record: TrackRecord) -> None:
    """The five newest calls too young to score."""
    if not record.pending:
        return
    pending = sorted(record.pending, key=lambda p: p.pick_date, reverse=True)[:5]
    sections.append(
        Section(
            kind="heading",
            text=f"Too young to score ({record.n_pending}; live mark only)",
            level=3,
        )
    )
    sections.append(
        Section(
            kind="table",
            table_header=["Call", "Ticker", "Date", "Age", "Live return"],
            table_rows=[
                [
                    _DIRECTION_LABELS.get(p.direction, p.direction),
                    p.ticker,
                    p.pick_date,
                    f"{p.age_days}d",
                    _pct_or_dash(p.pick_return_pct),
                ]
                for p in pending
            ],
        )
    )


def _unscorable_calls_note(sections: list[Section], record: TrackRecord) -> None:
    """Calls with no forward price are flagged, not dropped."""
    no_data = sorted(
        (u for u in record.unmeasurable if u.reason == "no_price_data"),
        key=lambda u: u.pick_date,
    )
    if no_data:
        names = ", ".join(
            f"{u.ticker} ({_DIRECTION_LABELS.get(u.direction, u.direction).lower()}, {u.pick_date})"
            for u in no_data
        )
        sections.append(
            Section(
                kind="para",
                text=(
                    f"No forward price for {len(no_data)} call(s) — delisted or a bad symbol, "
                    f"and likely a loss, so they are flagged rather than dropped: {names}."
                ),
            )
        )


def append_thesis_check_section(
    sections: list[Section], checks: list[dict[str, Any]] | None
) -> None:
    """'Open picks: thesis check' — a one-line tally, then a table of every
    pick that needs attention (broken, past its bull target, or on watch).
    Intact picks are only named, to keep the section short."""
    if not checks:
        return
    by_status: dict[str, list[dict[str, Any]]] = {}
    for c in checks:
        by_status.setdefault(c["status"], []).append(c)
    tally = ", ".join(
        f"{len(by_status[k])} {label}"
        for k, label in (
            ("BROKEN", "broken"),
            ("TARGET HIT", "past the bull target"),
            ("WATCH", "on watch"),
            ("INTACT", "intact"),
        )
        if by_status.get(k)
    )
    from .thesis_tracker import OPEN_WINDOW_DAYS

    sections.append(Section(kind="heading", text="Open picks: thesis check", level=2))
    text = (
        f"{len(checks)} picks from the last {OPEN_WINDOW_DAYS} days, re-checked against their own "
        f"bear/bull targets, the 200-day trend, SPY, their catalysts and EPS "
        f"revisions: {tally}."
    )
    intact = sorted(c["ticker"] for c in by_status.get("INTACT", []))
    if intact:
        text += f" Intact: {', '.join(intact)}."
    sections.append(Section(kind="para", text=text))
    flagged = [c for c in checks if c["status"] != "INTACT"]
    if not flagged:
        return

    def targets(c: dict[str, Any]) -> str:
        bear, bull = c.get("bear_target_pct"), c.get("bull_target_pct")
        if bear is None and bull is None:
            return "—"
        per = "/yr" if c.get("annualized") else ""
        return f"{_pct_or_dash(bear)}{per} / {_pct_or_dash(bull)}{per}"

    sections.append(
        Section(
            kind="table",
            table_header=["Status", "Ticker", "Picked", "Return", "vs SPY", "Bear / bull", "Why"],
            table_rows=[
                [
                    c["status"],
                    c["ticker"],
                    c["pick_date"],
                    _pct_or_dash(c.get("return_pct")),
                    f"{c['excess_pct']:+.1f} pts" if c.get("excess_pct") is not None else "—",
                    targets(c),
                    "; ".join(s["text"] for s in c["signals"] if s["severity"] != "info") or "—",
                ]
                for c in flagged
            ],
        )
    )
    upcoming = [
        (c["ticker"], s["text"]) for c in checks for s in c["signals"] if s["severity"] == "info"
    ]
    if upcoming:
        sections.append(
            Section(
                kind="para",
                text="Coming up: " + "; ".join(f"{t}: {msg}" for t, msg in upcoming) + ".",
            )
        )


def append_paper_ledger_section(sections: list[Section], ledger: dict[str, Any] | None) -> None:
    """'Paper portfolio vs SPY' — headline sentence, equity curve, per-run table."""
    if not ledger or not ledger.get("dates"):
        return
    invested = ledger["invested"][-1]
    picks_value = ledger["strategy"][-1]
    spy_value = ledger["benchmark"][-1]
    picks_ret = ledger.get("strategy_return_pct") or 0.0
    spy_ret = ledger.get("benchmark_return_pct") or 0.0
    verdict = "ahead of" if picks_ret > spy_ret else "behind"
    sections.append(Section(kind="heading", text="Paper portfolio vs SPY", level=2))
    sections.append(
        Section(
            kind="para",
            text=(
                f"If every past run's picks had received $1,000 at the Sizer's weights, "
                f"the ${invested:,.0f} invested would be worth ${picks_value:,.0f} "
                f"({picks_ret:+.1f}%) — {abs(picks_ret - spy_ret):.1f} points {verdict} the "
                f"${spy_value:,.0f} ({spy_ret:+.1f}%) the same money earned in SPY. "
                f"Dividend-adjusted; no costs, taxes or trims."
            ),
        )
    )
    sections.append(Section(kind="equity_curve", data=ledger))
    rows = [
        [
            t["run_date"],
            ", ".join(t["tickers"]),
            f"{t['strategy_return_pct']:+.1f}%",
            f"{t['benchmark_return_pct']:+.1f}%",
            f"{t['strategy_return_pct'] - t['benchmark_return_pct']:+.1f}",
        ]
        for t in reversed(ledger.get("tranches") or [])
    ]
    if rows:
        sections.append(
            Section(
                kind="table",
                table_header=["Run", "Picks", "Picks return", "SPY return", "Excess (pts)"],
                table_rows=rows,
            )
        )


def append_at_a_glance(
    sections: list[Section],
    *,
    structured_ranker: Any,
    structured_sizer: Any,
    pick_order: list[str],
    thesis_checks: list[dict[str, Any]] | None,
    data_warnings: list[str] | None,
    usage: dict[str, Any] | None,
) -> None:
    """'At a glance' — the few lines that decide this run: each pick with
    its size, conviction and consensus, then anything that needs attention
    (open picks whose thesis broke or hit target, deterministic trims,
    data warnings, cost-cap cuts). Everything else is detail below."""
    if not pick_order and not thesis_checks:
        return
    sections.append(Section(kind="heading", text="At a glance", level=2))
    picks = {p.ticker: p for p in getattr(structured_ranker, "picks", None) or []}
    allocs = {a.ticker: a for a in getattr(structured_sizer, "allocations", None) or []}
    rows = []
    for t in pick_order:
        p, a = picks.get(t), allocs.get(t)
        size = "—"
        if a is not None and a.allocation_pct is not None:
            size = f"{a.allocation_pct:.0f}%"
        elif a is not None and a.allocation_usd is not None:
            size = f"${a.allocation_usd:,.0f}"
        # agreement_ratio = agreeing rounds / rounds run; voting_providers
        # lists the agreeing ones.
        agree = len(getattr(p, "voting_providers", None) or [])
        consensus = (
            f"{agree}/{round(agree / p.agreement_ratio)}"
            if p is not None and p.agreement_ratio and agree
            else "—"
        )
        why = (p.one_liner if p is not None else "") or ""
        rows.append(
            [
                t,
                size,
                f"{p.conviction}/10" if p is not None else "—",
                consensus,
                why if len(why) <= 110 else why[:107].rstrip() + "…",
            ]
        )
    if rows:
        sections.append(
            Section(
                kind="table",
                table_header=["Buy", "Size", "Conviction", "Models agree", "Why"],
                table_rows=rows,
            )
        )
    flags: list[str] = []
    for c in thesis_checks or []:
        if c["status"] == "BROKEN":
            flags.append(
                f"Open pick {c['ticker']}: thesis broken ({c['return_pct']:+.1f}% since pick)."
            )
        elif c["status"] == "TARGET HIT":
            flags.append(
                f"Open pick {c['ticker']}: past its bull target ({c['return_pct']:+.1f}%) — "
                f"re-underwrite or take profits."
            )
    for w in getattr(structured_sizer, "concentration_warnings", None) or []:
        if w.startswith(("SECTOR CAP", "CORRELATION CAP", "EARNINGS BLACKOUT")):
            flags.append(w.split(" — ")[0].rstrip(".") + " (sizes above already reflect it).")
    if data_warnings:
        flags.append(f"{len(data_warnings)} data warning(s) — see Data warnings below.")
    for note in ((usage or {}).get("budget") or {}).get("notes") or []:
        flags.append(f"Cost cap: {note}.")
    if not flags:
        sections.append(Section(kind="para", text="Nothing else needs attention this run."))
        return
    sections.append(Section(kind="para", text="Needs attention:"))
    sections.extend(Section(kind="para", text=f"• {f}") for f in flags)


def append_usage_section(sections: list[Section], usage: dict[str, Any] | None) -> None:
    """'Model usage this run' table from usage.UsageTracker.report_data()."""
    if not usage or not usage.get("rows"):
        return
    rows = [
        [
            r["stage"],
            r["model"],
            str(r["calls"]),
            f"{r['input_tokens']:,}",
            f"{r['output_tokens']:,}",
            f"${r['cost_usd']:.2f}" if r["cost_usd"] is not None else "n/a",
        ]
        for r in usage["rows"]
    ]
    total = f"${usage['total_cost_usd']:.2f}"
    if not usage.get("cost_complete"):
        total += " + unpriced non-Claude calls"
    sections.append(Section(kind="heading", text="Model usage this run", level=2))
    sections.append(
        Section(
            kind="table",
            table_header=["Stage", "Model", "Calls", "Tokens in", "Tokens out", "Est. cost"],
            table_rows=rows,
        )
    )
    sections.append(Section(kind="para", text=f"Estimated Claude cost: {total}"))
    budget = usage.get("budget")
    if budget:
        cuts = budget.get("notes") or []
        text = (
            f"Cost cap ${budget['cap_usd']:.2f}: estimated priced spend "
            f"${budget['spent_usd']:.2f}. "
        )
        text += (
            "To stay under it this run: " + "; ".join(cuts) + "."
            if cuts
            else "Nothing had to be cut."
        )
        sections.append(Section(kind="para", text=text))


# --- sections (unified IR for HTML + PDF) -----------------------------------


def build_sections(
    *,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    candidates: list[dict[str, Any]],
    universe_size: int,
    holdings_summary: str,
    holdings_rows: list[list[str]] | None = None,
    macro_summary: str = "",
    sector_rotation: dict[str, Any] | None = None,
    track_record_block: str = "",
    track_record: TrackRecord | None = None,
    ranker_output: object = None,
    redteam_output: object = None,
    sizer_output: object = None,
    market_themes: object = None,
    data_warnings: list[str] | None = None,
    pick_tilts: dict[str, dict[str, float]] | None = None,
    portfolio_tilt: dict[str, float] | None = None,
    pick_catalysts: dict[str, list[dict[str, Any]]] | None = None,
    usage: dict[str, Any] | None = None,
    paper_ledger: dict[str, Any] | None = None,
    thesis_checks: list[dict[str, Any]] | None = None,
) -> list[Section]:
    # Prefer the structured Phase 4 objects when present; fall back to
    # parsing the free-text variants so legacy callers / partial runs
    # still render something.
    from ..models.llm import RankerOutput, RedTeamOutput, SizerOutput

    structured_ranker = ranker_output if isinstance(ranker_output, RankerOutput) else None
    structured_redteam = redteam_output if isinstance(redteam_output, RedTeamOutput) else None
    structured_sizer = sizer_output if isinstance(sizer_output, SizerOutput) else None

    today = date.today().isoformat()
    pick_order = [t for _, t, _ in parse_picks(structured_ranker or ranker_text)]
    survivors = [c for c in candidates if c["passed_filter"]]
    rejected = [c for c in candidates if not c["passed_filter"]]

    s: list[Section] = []

    s.append(Section(kind="heading", text=f"Stock discovery picks — {today}", level=1))
    s.append(
        Section(
            kind="para",
            text=(
                f"{universe_size} candidates considered, {len(survivors)} survived "
                f"hard filters, {len(pick_order)} picks."
            ),
        )
    )

    append_at_a_glance(
        s,
        structured_ranker=structured_ranker,
        structured_sizer=structured_sizer,
        pick_order=pick_order,
        thesis_checks=thesis_checks,
        data_warnings=data_warnings,
        usage=usage,
    )

    # Context (themes, macro, rotation, holdings) is collected here and
    # placed after the picks: the report leads with what to act on.
    ctx = _context_sections(
        market_themes=market_themes,
        macro_summary=macro_summary,
        data_warnings=data_warnings,
        sector_rotation=sector_rotation,
        holdings_rows=holdings_rows,
        holdings_summary=holdings_summary,
    )

    append_pick_cards(
        s,
        pick_order=pick_order,
        structured_ranker=structured_ranker,
        structured_redteam=structured_redteam,
        structured_sizer=structured_sizer,
        ranker_text=ranker_text,
        redteam_text=redteam_text,
        sizer_text=sizer_text,
        pick_catalysts=pick_catalysts,
    )

    s.append(Section(kind="page_break"))
    append_allocation_table(s, structured_sizer)
    append_correlation_notes(s, structured_ranker, ranker_text)
    append_redteam_summary(s, structured_redteam, redteam_text)
    append_sizer_warnings(s, structured_sizer, sizer_text)

    append_thesis_check_section(s, thesis_checks)
    s.extend(ctx)
    append_track_record_section(s, track_record, track_record_block)
    append_paper_ledger_section(s, paper_ledger)

    append_survivor_table(s, survivors)
    append_factor_tilt(s, portfolio_tilt, pick_tilts)
    append_rejected_candidates(s, rejected, n_candidates=len(candidates))

    append_usage_section(s, usage)
    return s


def market_themes_sections(market_themes: object) -> list[Section]:
    """The 'Current market themes' heading and panel, when there are any."""
    from ..models.llm import MarketThemes

    if not (isinstance(market_themes, MarketThemes) and market_themes.themes):
        return []
    return [
        Section(kind="heading", text="Current market themes", level=2),
        Section(
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
        ),
    ]


def _context_sections(
    *,
    market_themes: object,
    macro_summary: str,
    data_warnings: list[str] | None,
    sector_rotation: dict[str, Any] | None,
    holdings_rows: list[list[str]] | None,
    holdings_summary: str,
) -> list[Section]:
    """Themes, macro, data warnings, rotation and holdings — the context
    that follows the picks."""
    # Market themes panel — what's hot right now (drives ranker bias).
    ctx: list[Section] = market_themes_sections(market_themes)

    if macro_summary:
        ctx.append(Section(kind="heading", text="Macro regime", level=2))
        ctx.append(Section(kind="blockquote", text=macro_summary))

    if data_warnings:
        ctx.append(Section(kind="heading", text="Data warnings", level=2))
        ctx.append(
            Section(
                kind="preformatted",
                text="\n".join(f"- {w}" for w in data_warnings),
            )
        )

    if sector_rotation and sector_rotation.get("leaders"):
        leaders = ", ".join(sector_rotation.get("leaders", []))
        laggards = ", ".join(sector_rotation.get("laggards", []))
        ctx.append(Section(kind="heading", text="Sector rotation (6-month returns)", level=2))
        ctx.append(Section(kind="para", text=f"Leaders: {leaders}"))
        ctx.append(Section(kind="para", text=f"Laggards: {laggards}"))

    ctx.append(Section(kind="heading", text="Current holdings (concentration context)", level=2))
    if holdings_rows:
        ctx.append(
            Section(
                kind="table",
                table_header=["Ticker", "Shares", "Avg cost", "Cost basis"],
                table_rows=holdings_rows,
            )
        )
    else:
        ctx.append(Section(kind="preformatted", text=holdings_summary or "(none)"))
    return ctx


def append_pick_cards(
    s: list[Section],
    *,
    pick_order: list[str],
    structured_ranker: Any,
    structured_redteam: Any,
    structured_sizer: Any,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    pick_catalysts: dict[str, list[dict[str, Any]]] | None,
) -> None:
    """Per-pick cards. When structured outputs are present, emit a single
    rich pick_card section per ticker (renderer composes rank pill +
    conviction badge + fragility chip + allocation + bull/bear prose).
    Otherwise fall back to the legacy heading + preformatted layout."""
    pick_blocks = _split_by_pick_blocks(ranker_text)
    bear_blocks = _split_by_ticker_blocks(redteam_text)
    alloc_blocks = _split_by_ticker_blocks(sizer_text)
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
            s.append(
                Section(
                    kind="pick_card",
                    data=_pick_card_data(
                        ticker,
                        pick_by_ticker[ticker],
                        bear_by_ticker.get(ticker),
                        alloc_by_ticker.get(ticker),
                        (pick_catalysts or {}).get(ticker, []),
                    ),
                )
            )
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


def _pick_card_data(
    ticker: str, pick: Any, bear: Any, alloc: Any, catalysts: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
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
        "most_fragile_assumption": (bear.most_fragile_assumption if bear else None),
        "watch_metric": bear.watch_metric if bear else None,
        "fragility_rank": bear.fragility_rank if bear else None,
        "allocation_pct": (alloc.allocation_pct if alloc else None),
        "allocation_usd": (alloc.allocation_usd if alloc else None),
        "allocation_rationale": alloc.rationale if alloc else None,
        "agreement_ratio": pick.agreement_ratio,
        "voting_providers": pick.voting_providers,
        "catalysts": catalysts,
    }


def append_allocation_table(s: list[Section], structured_sizer: Any) -> None:
    """Structured allocation table when sizer ran in Phase 4e mode."""
    if not (structured_sizer and structured_sizer.allocations):
        return
    s.append(Section(kind="heading", text="Allocation summary", level=2))
    s.append(
        Section(
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
        )
    )


def append_correlation_notes(s: list[Section], structured_ranker: Any, ranker_text: str) -> None:
    s.append(Section(kind="heading", text="Ranker correlation notes", level=2))
    if structured_ranker is not None:
        # Structured output ran — pairs_not_to_hold_together is the
        # authoritative source. Never fall back to regex-parsing full_text
        # here: an empty list means "no flagged pairs", not "look for
        # trailing prose" — that fallback was picking up unrelated stray
        # text from the end of the last pick's block.
        if structured_ranker.pairs_not_to_hold_together:
            for pair in structured_ranker.pairs_not_to_hold_together:
                s.append(
                    Section(
                        kind="para",
                        text=(f"{pair.ticker_a} + {pair.ticker_b}: {pair.shared_driver}"),
                    )
                )
        else:
            s.append(Section(kind="para", text="(none)"))
    else:
        trailing = re.split(_PICK_RE, ranker_text)[-1].strip()
        s.append(Section(kind="preformatted", text=trailing or "(none)"))


def append_redteam_summary(s: list[Section], structured_redteam: Any, redteam_text: str) -> None:
    s.append(Section(kind="heading", text="Red-team summary", level=2))
    if structured_redteam:
        s.append(
            Section(
                kind="para",
                text=f"Single most fragile pick: {structured_redteam.single_most_fragile_pick}",
            )
        )
    else:
        s.append(
            Section(kind="preformatted", text=redteam_text.split("---")[-1].strip() or "(none)")
        )


def append_sizer_warnings(s: list[Section], structured_sizer: Any, sizer_text: str) -> None:
    s.append(Section(kind="heading", text="Sizer concentration warnings", level=2))
    if structured_sizer:
        if structured_sizer.concentration_warnings:
            for w in structured_sizer.concentration_warnings:
                s.append(Section(kind="para", text=f"• {w}"))
        else:
            s.append(Section(kind="para", text="(none)"))
    else:
        s.append(Section(kind="preformatted", text=sizer_text.split("---")[-1].strip() or "(none)"))


def append_survivor_table(s: list[Section], survivors: list[dict[str, Any]]) -> None:
    if not survivors:
        return
    s.append(Section(kind="page_break"))
    s.append(Section(kind="heading", text="All candidates that passed filters", level=2))
    rows: list[list[str]] = []
    for c in sorted(survivors, key=lambda x: x.get("score") or 0, reverse=True):
        comp = c.get("score_components") or {}
        rows.append(
            [
                c["ticker"],
                f"{c.get('score') or '—'}",
                f"{comp.get('fundamentals', '—')}",
                f"{comp.get('trend', '—')}",
                f"{comp.get('conviction', '—')}",
                c.get("sector") or "—",
            ]
        )
    s.append(
        Section(
            kind="table",
            table_header=["Ticker", "Score", "Fund.", "Trend", "Conv.", "Sector"],
            table_rows=rows,
        )
    )


def append_factor_tilt(
    s: list[Section],
    portfolio_tilt: dict[str, float] | None,
    pick_tilts: dict[str, dict[str, float]] | None,
) -> None:
    if not (portfolio_tilt or pick_tilts):
        return
    s.append(Section(kind="heading", text="Style factor tilt", level=2))
    s.append(
        Section(
            kind="factor_tilt_panel",
            data={
                "portfolio": portfolio_tilt or {},
                "picks": [
                    {"ticker": ticker, "tilt": tilt} for ticker, tilt in (pick_tilts or {}).items()
                ],
            },
        )
    )


def append_rejected_candidates(
    s: list[Section], rejected: list[dict[str, Any]], *, n_candidates: int
) -> None:
    if not rejected:
        return
    s.append(Section(kind="page_break"))
    s.append(Section(kind="heading", text="Rejected candidates", level=2))
    s.append(
        Section(
            kind="para",
            text=(
                f"{len(rejected)} of {n_candidates} candidates were "
                f"eliminated by the hard filter, grouped below by their "
                f"primary reason."
            ),
        )
    )
    by_reason: dict[str, list[str]] = {}
    for c in rejected:
        label = _primary_reject_reason(c.get("fail_reasons") or [])
        by_reason.setdefault(label, []).append(c["ticker"])
    ordered_reasons = sorted(by_reason.items(), key=lambda kv: len(kv[1]), reverse=True)
    # Reusing the generic sector_pie renderer (label, value) pairs — it
    # has no sector-specific logic, just a labeled pie chart.
    s.append(
        Section(
            kind="sector_pie",
            pie_data=[(label, float(len(tickers))) for label, tickers in ordered_reasons],
        )
    )
    for label, tickers in ordered_reasons:
        s.append(
            Section(
                kind="para",
                text=f"{label} ({len(tickers)}): " + ", ".join(sorted(tickers)),
            )
        )


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
