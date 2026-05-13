"""Deterministic EV computation on probability-weighted scenarios.

The Sizer ranks picks by E[return] = Σ(p × return). The LLM never
computes this directly — it produces scenarios; we compute the
weighted average and use that for sizing. A wrong sum / sign flip
here would silently mis-rank every position.
"""
from __future__ import annotations

from stock_analyzer.discover.schemas import RankerPick, Scenario, expected_return_pct


def _pick(scenarios: list[Scenario]) -> RankerPick:
    """Minimal RankerPick — only `scenarios` matters for expected_return_pct,
    but the other fields are required by the schema."""
    return RankerPick.model_construct(
        rank=1, ticker="NVDA", one_liner="x",
        why_over_alternatives="x", conviction=8, time_horizon="6-12mo",
        sector_concentration_check="x", bull_thesis="x",
        what_youre_betting_on="x", scenarios=scenarios,
    )


def test_expected_return_is_probability_weighted_sum():
    """30% × 60% + 50% × 8% + 20% × -25% = 18 + 4 + -5 = 17."""
    pick = _pick([
        Scenario(label="bull", probability=0.30, target_return_pct=60.0,
                 rationale="ai capex"),
        Scenario(label="base", probability=0.50, target_return_pct=8.0,
                 rationale="steady"),
        Scenario(label="bear", probability=0.20, target_return_pct=-25.0,
                 rationale="recession"),
    ])
    assert expected_return_pct(pick) == 17.0


def test_returns_none_when_probabilities_dont_sum_to_one():
    """The LLM occasionally emits scenarios where probabilities sum to
    e.g. 0.7 — that's a math error, not a valid distribution. Reject
    so the Sizer doesn't size on bogus EV."""
    pick = _pick([
        Scenario(label="bull", probability=0.20, target_return_pct=60.0,
                 rationale="x"),
        Scenario(label="base", probability=0.30, target_return_pct=8.0,
                 rationale="x"),
        Scenario(label="bear", probability=0.20, target_return_pct=-25.0,
                 rationale="x"),
    ])
    # Sum = 0.70, well outside the [0.95, 1.05] tolerance.
    assert expected_return_pct(pick) is None


def test_returns_none_for_legacy_picks_without_scenarios():
    """Picks created before Phase 5c have empty scenarios list — the
    function must return None, not 0, so the Sizer can ignore them
    rather than treating them as 0% EV."""
    pick = _pick([])
    assert expected_return_pct(pick) is None


# --- MarketTheme min_length regression -----------------------------------


def test_market_theme_accepts_under_populated_member_tickers():
    """The LLM occasionally emits a theme with one member ticker (we saw
    `['AU']` in production). The schema must NOT reject the whole
    MarketThemes response — that triggers all-parsing-attempts-failed
    and the entire theme block is lost. The auto-correct in
    cli/discover.py drops <3-member themes at runtime; the schema's job
    is structural validity, not quality enforcement."""
    from stock_analyzer.discover.schemas import MarketTheme, MarketThemes
    theme = MarketTheme(
        name="Gold miners",
        description="Bullion ride",
        strength=4,
        trending="up",
        member_tickers=["AU"],  # under-populated; auto-correct will drop
    )
    # Must NOT raise.
    output = MarketThemes(themes=[theme], full_text="...")
    assert output.themes[0].member_tickers == ["AU"]
