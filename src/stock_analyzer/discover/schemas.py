"""Pydantic schemas for every LLM stage's structured output.

The discover + rebalance pipelines used to glue stages together via
free-text + regex parsing — that was the source of recurring fragility
(empty content from rate-limit retries → parsers raised on None,
labeled prose drifting between runs, etc.).

Now every LLM stage emits a validated Pydantic instance. Each schema
keeps a `full_text` field carrying the prose rendering of that stage's
output: PDF/email renderers continue to use it, AND downstream LLM
stages still receive prompt-ready text without needing to know about
this refactor.

Structured fields are the source of truth for parsers, dashboards,
analytics, and the modernized report layout (per-pick cards, verdict
badges, fragility chips, sizing tables).

Schemas live here (one module) so the LLM contracts are easy to find
and review together. The pre-existing `rebalance_schema.RebalancePlan`
stays in its own file for historical continuity.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["HOLD", "TRIM", "SELL"]
ActionType = Literal["SELL", "TRIM", "ADD", "BUY"]
FragilityRank = Literal[1, 2, 3, 4, 5]


# --- Reviewer ---------------------------------------------------------------


class HoldingReview(BaseModel):
    """Per-holding review emitted by the Reviewer agent.

    Replaces the prior free-text output that downstream parsers
    (parse_verdict / parse_confidence) regex-scanned for verdict +
    confidence — those reads are now trivial field accesses.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    verdict: Verdict = Field(..., description="HOLD / TRIM / SELL — default HOLD when uncertain.")
    confidence: int = Field(..., ge=1, le=10, description="1-10 conviction score.")
    trim_pct: float | None = Field(
        default=None,
        description="Only set for TRIM verdicts: 25 / 33 / 50.",
    )
    position_context: str = Field(
        ...,
        description="One line: N shares, avg cost $X, current $Y, P/L +/-Z%.",
    )
    forward_outlook: str = Field(
        ...,
        description=(
            "1-2 sentences citing forward EPS, target price vs current, "
            "analyst stance, calendar catalysts. Not trailing performance."
        ),
    )
    reasoning: str = Field(
        ...,
        description=(
            "2-3 sentences citing specific FORWARD-LOOKING evidence for the "
            "verdict. Past performance is supporting evidence only."
        ),
    )
    tax_lot_plan: list[str] = Field(
        default_factory=list,
        description=(
            "SELL/TRIM only. Per-lot recommendations as strings: "
            "'2024-01-15: sell 50 shares (held 400 days, long-term), "
            "realizes ~$5,200 gain'."
        ),
    )
    what_would_change_mind: str = Field(
        ...,
        description="One sentence — the specific evidence that would flip the verdict.",
    )
    wash_sale_notice: str | None = Field(
        default=None,
        description=(
            "Only set if you recommend SELL at a loss. Warn the user not to "
            "re-buy this security or a substantially identical one within "
            "30 days of the sale."
        ),
    )

    # Always required — the prose rendering for the PDF/email + the
    # rebalancer prompt (which feeds the full text of every review into
    # its own structured-output call).
    full_text: str = Field(
        ...,
        description=(
            "Human-readable rendering of this review matching the prose "
            "format in your instructions (TICKER: ... / Verdict: ... / "
            "Forward outlook: ... etc.)."
        ),
    )


# --- Ranker ---------------------------------------------------------------


class RankerPick(BaseModel):
    """One of the Ranker's top-N picks, structured."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(..., ge=1, description="1-indexed pick rank, descending by conviction.")
    ticker: str
    one_liner: str = Field(..., description="Single sentence thesis.")
    why_over_alternatives: str = Field(
        ...,
        description=(
            "2-3 sentences citing other candidates that lost out and why. "
            "Name the alternatives by ticker."
        ),
    )
    conviction: int = Field(..., ge=1, le=10)
    time_horizon: str = Field(default="6-12 months")
    sector_concentration_check: str = Field(
        ...,
        description=(
            "Does this overlap with current holdings? Flag concentration risk."
        ),
    )
    bull_thesis: str = Field(
        ...,
        description="3-4 sentences synthesizing fundamentals + trend + catalysts.",
    )
    what_youre_betting_on: str = Field(
        ...,
        description="1-2 sentences making the core assumption explicit.",
    )


class CorrelatedPair(BaseModel):
    """Two picks that share too much factor exposure to size as independent bets."""

    model_config = ConfigDict(frozen=True)

    ticker_a: str
    ticker_b: str
    shared_driver: str = Field(
        ...,
        description="Brief: 'same hyperscaler AI capex cycle', 'same gold-price tape', etc.",
    )


class RankerOutput(BaseModel):
    """Structured output of the Ranker agent. Replaces _PICK_RE regex
    in the consensus-run majority logic and parse_picks downstream."""

    model_config = ConfigDict(frozen=True)

    picks: list[RankerPick] = Field(
        ...,
        min_length=1,
        description="Top N picks ordered by conviction descending (rank 1 = highest).",
    )
    pairs_not_to_hold_together: list[CorrelatedPair] = Field(
        default_factory=list,
        description=(
            "Highly-correlated pairs among your picks (same sector + similar "
            "drivers). Empty if no problematic pairs."
        ),
    )
    full_text: str = Field(
        ...,
        description=(
            "Plain-text rendering matching the prose template ('---' "
            "separators, 'PICK N: TICKER — ...' headers, etc.). What the "
            "RedTeam / Sizer / Rebalancer read as prompt input."
        ),
    )


# --- Red team -------------------------------------------------------------


class BearCase(BaseModel):
    """Adversarial bear case for one pick."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    bear_case: str = Field(
        ...,
        description=(
            "3-4 sentences naming concrete failure modes — earnings miss, "
            "margin compression, competitor wins, valuation re-rating, "
            "regulatory action. Cite specific numbers when possible."
        ),
    )
    most_fragile_assumption: str = Field(
        ...,
        description=(
            "Single sentence identifying the load-bearing assumption that, "
            "if wrong, breaks the bull thesis."
        ),
    )
    watch_metric: str = Field(
        ...,
        description=(
            "One concrete number to watch, e.g. 'Q2 revenue growth — if "
            "below 15%, thesis is wrong'."
        ),
    )
    fragility_rank: FragilityRank = Field(
        ...,
        description=(
            "1 = most fragile (highest probability of disappointment), "
            "5 = most resilient."
        ),
    )


class RedTeamOutput(BaseModel):
    """Structured output of the RedTeam agent."""

    model_config = ConfigDict(frozen=True)

    bear_cases: list[BearCase] = Field(..., min_length=1)
    single_most_fragile_pick: str = Field(
        ...,
        description=(
            "Ticker of the pick most likely to disappoint, with a one-sentence "
            "explanation. Format: 'TICKER — <reason>'."
        ),
    )
    full_text: str = Field(
        ...,
        description=(
            "Plain-text rendering: '---' separators between BearCase blocks, "
            "trailing 'Single most fragile pick:' line. What the Sizer / "
            "Rebalancer read as prompt input."
        ),
    )


# --- Market themes --------------------------------------------------------


class MarketTheme(BaseModel):
    """One macro/sector theme that's currently driving prices."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        description=(
            "Short theme name, e.g. 'AI compute capex', 'GLP-1 weight-loss', "
            "'energy transition', 'defense rearmament', 'cybersecurity'."
        ),
    )
    description: str = Field(
        ...,
        description="1-2 sentences on why this theme is hot now and what's driving it.",
    )
    strength: int = Field(
        ...,
        ge=1,
        le=10,
        description=(
            "Theme strength right now. 10 = dominant secular trend with "
            "broad price + earnings tailwind; 5 = real but contested; "
            "1 = waning / topping out."
        ),
    )
    trending: Literal["up", "flat", "down"] = Field(
        ...,
        description=(
            "Direction over the LAST 30 DAYS. 'up' = accelerating, "
            "'down' = decelerating / rolling over, 'flat' = sideways."
        ),
    )
    member_tickers: list[str] = Field(
        ...,
        min_length=3,
        description=(
            "10-25 tickers that meaningfully benefit from this theme. "
            "Be liberal — include direct beneficiaries (e.g. NVDA for AI "
            "compute) AND adjacent ones (e.g. ANET for AI networking, "
            "VST/CEG for AI power)."
        ),
    )


class MarketThemes(BaseModel):
    """Snapshot of dominant market themes for one pipeline run."""

    model_config = ConfigDict(frozen=True)

    themes: list[MarketTheme] = Field(
        ...,
        min_length=3,
        max_length=10,
        description=(
            "5-8 themes that are materially moving stocks right now. "
            "Skip stale/long-faded themes; include rising ones."
        ),
    )
    full_text: str = Field(
        ...,
        description=(
            "Plain-text rendering of all themes for downstream LLM prompts: "
            "one named theme per block with strength + trend + description "
            "+ exemplar tickers. The ranker and rebalancer read this verbatim."
        ),
    )


# --- Sizer ----------------------------------------------------------------


class Allocation(BaseModel):
    """Sizing for one pick."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    allocation_pct: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Percentage of new capital. Set this when no dollar budget was "
            "given. Either allocation_pct OR allocation_usd must be set."
        ),
    )
    allocation_usd: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Dollar amount when a cash budget was given. Either "
            "allocation_pct OR allocation_usd must be set."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "1-2 sentences citing conviction, fragility rank, correlation "
            "to existing holdings."
        ),
    )


class SizerOutput(BaseModel):
    """Structured output of the Sizer agent."""

    model_config = ConfigDict(frozen=True)

    allocations: list[Allocation] = Field(..., min_length=1)
    concentration_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Sector or theme where new picks + existing holdings would "
            "exceed 30% combined. Empty if no warnings."
        ),
    )
    full_text: str = Field(
        ...,
        description=(
            "Plain-text rendering: '---' separators between Allocation "
            "blocks, trailing 'Concentration warnings:' block. What the "
            "PDF/email render reads."
        ),
    )


# --- Analyst ---------------------------------------------------------------


class AnalystReport(BaseModel):
    """Per-candidate analyst report emitted by the Analyst agent.

    Replaces the prior free-text format ('TICKER: AAPL / Score: 7 / ...').
    Downstream (Ranker) reads full_text from its prompt input; the report
    layer and any analytics read the structured fields directly.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    score: int = Field(..., ge=1, le=10, description="1-10 conviction score.")
    one_liner: str = Field(..., description="Single sentence summary, no fluff.")
    competitive_position: str = Field(
        ...,
        description="1-2 sentences on moat / market position / what's hard to replicate.",
    )
    growth_runway: str = Field(
        ...,
        description="1-2 sentences on 3-5 year revenue/profit drivers from the data.",
    )
    top_risks: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "Up to 5 concrete risks extracted from 10-K risk factors or news. "
            "Not generic boilerplate."
        ),
    )
    valuation_context: str = Field(
        ...,
        description="1-2 sentences: PE / FCF yield vs peers or historical; is it stretched?",
    )
    catalyst_calendar: str = Field(
        ...,
        description="Next earnings date if known + any product / regulatory items from news.",
    )
    full_text: str = Field(
        ...,
        description=(
            "Plain-text rendering matching the prose template: TICKER: / "
            "Score: / One-liner: / Competitive position: / ... — what the "
            "ranker reads as prompt input."
        ),
    )


__all__ = [
    "Verdict",
    "ActionType",
    "FragilityRank",
    "HoldingReview",
    "AnalystReport",
    "RankerPick",
    "CorrelatedPair",
    "RankerOutput",
    "BearCase",
    "RedTeamOutput",
    "Allocation",
    "SizerOutput",
    "MarketTheme",
    "MarketThemes",
]
