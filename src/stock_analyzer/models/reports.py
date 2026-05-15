"""Pydantic models for the unified report intermediate representation.

The HTML and PDF renderers both consume the same ``Section`` list, so
this module owns the section IR (``SectionKind`` + ``Section``), the
per-ticker email block (``TickerSection``), and the plan-level
pre-mortem models (``PreMortem`` + ``PreMortemFailure``) that drive
the pre-mortem panel section.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SectionKind = Literal[
    "heading", "para", "preformatted", "image", "blockquote", "table",
    "page_break", "status_banner", "metric_strip", "holdings_dashboard",
    "sector_pie",
    # New structured-output kinds (Phase 4f) — renderer pulls fields from
    # `data` and produces a styled card / table instead of dumping prose.
    "pick_card", "allocation_table", "rebalance_action_table",
    "holding_review_card", "market_themes_panel", "premortem_panel",
    # Covered-call sections (cli/rebalance.py CC extension).
    "premium_income", "round_lot_coverage", "premium_deployment",
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


class TickerSection(BaseModel):
    symbol: str
    name: str
    fields: list[tuple[str, str]] = Field(default_factory=list)


class PreMortemFailure(BaseModel):
    """One specific way the plan could fail."""

    model_config = ConfigDict(frozen=True)

    likelihood: Literal["low", "medium", "high"] = Field(
        ...,
        description=(
            "How likely is this failure mode given current data? "
            "high = >25% probability; medium = 10-25%; low = <10%."
        ),
    )
    severity: Literal["mild", "moderate", "severe"] = Field(
        ...,
        description=(
            "If this failure plays out, how much portfolio damage? "
            "severe = double-digit % loss across the plan; moderate = "
            "single-digit; mild = uncomfortable but recoverable."
        ),
    )
    triggering_action: str = Field(
        ...,
        description=(
            "Which specific action in the plan triggers this failure mode? "
            "Quote it (e.g. 'Action 2: ADD GOOGL ~$3,400')."
        ),
    )
    failure_narrative: str = Field(
        ...,
        description=(
            "2-3 sentence imagined post-mortem in past tense: "
            "'In April, GOOGL fell 18% after the DOJ Chrome divestiture "
            "ruling. The plan's ADD added to losses; in hindsight the "
            "wash-sale window on the MRVL trim made the timing worse.' "
            "Cite specific named events or metrics."
        ),
    )
    early_warning: str = Field(
        ...,
        description=(
            "ONE metric or event the user could watch in the next 30 days "
            "that would tell them this failure mode is materializing."
        ),
    )


class PreMortem(BaseModel):
    """Structured output of the PreMortem agent."""

    model_config = ConfigDict(frozen=True)

    overall_verdict: Literal["proceed_as_planned", "proceed_with_caveat", "reconsider"] = Field(
        ...,
        description=(
            "After examining the failure modes, would you recommend the user "
            "execute this plan as-is, execute with smaller sizes / staged "
            "entry, or reconsider entirely?"
        ),
    )
    summary: str = Field(
        ...,
        description=(
            "One paragraph summarizing the pre-mortem: what's the single "
            "most likely way this plan goes wrong and why."
        ),
    )
    failures: list[PreMortemFailure] = Field(
        ...,
        min_length=1,
        max_length=6,
        description=(
            "2-4 specific failure modes ranked by likelihood × severity. "
            "Concrete, not generic — must reference actual actions in the "
            "plan, not 'market downturn'."
        ),
    )
    full_text: str = Field(
        ...,
        description=(
            "Plain-text rendering: VERDICT line + summary + per-failure "
            "blocks. What the PDF/email renders."
        ),
    )


__all__ = [
    "SectionKind",
    "Section",
    "TickerSection",
    "PreMortemFailure",
    "PreMortem",
]
