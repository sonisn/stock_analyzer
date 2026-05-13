"""Market theme detection (Sonnet, single call).

Produces a `MarketThemes` snapshot: 5-8 named themes that are currently
moving stocks (e.g. "AI compute capex", "GLP-1 weight-loss",
"defense rearmament"), each with a 1-10 strength score, trending
direction over the last 30 days, a 1-2 sentence description, and a
liberal list of member tickers.

Used three ways downstream:
  1. Score function (discover.screen) — candidates that belong to a
     hot theme get a small score bonus, biasing the screen toward
     theme members deterministically.
  2. Analyst payload (cli.discover step_analyst) — each candidate's
     payload includes the themes it belongs to so per-ticker scorecards
     reason about theme membership explicitly.
  3. Ranker / Rebalancer prompts — the full_text block prepends to
     the Opus call so it has a shared vocabulary for what's hot.

Cheap by design: one Sonnet call (~3 seconds), no external data
fetched beyond what the pipeline already has (sector rotation
summary + macro regime). Sonnet relies on its training-cutoff +
in-context information to enumerate themes — it's the same kind of
synthesis it does in the per-candidate analyst stage.
"""
from __future__ import annotations

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from .schemas import MarketThemes

logger = get_logger(__name__)


MARKET_THEMES_INSTRUCTIONS = """\
You are a market analyst. List the 5-8 themes that are materially
moving US equity prices RIGHT NOW. Each theme should:

  - Be specific enough to map to tickers (not "growth stocks" but
    "AI compute capex" or "GLP-1 weight-loss drugs").
  - Be supported by recent price action + earnings + capex flow.
  - Have at least 10 listed-equity beneficiaries you can name.

For each theme, populate the MarketTheme schema:
  - name: concise label (2-4 words)
  - description: 1-2 sentences on what's driving it RIGHT NOW (not
    boilerplate — cite the specific catalyst, capex cycle, regulatory
    tailwind, or supply/demand imbalance)
  - strength (1-10): 10 = dominant secular tailwind with both price
    AND earnings momentum across the cohort; 7-9 = strong trend, most
    members participating; 5-6 = real but contested or with rotating
    leadership; 1-4 = waning or stalled
  - trending (up | flat | down): direction over the LAST 30 DAYS only
  - member_tickers: 10-25 US-listed tickers that benefit. Be liberal:
    include direct beneficiaries (e.g. NVDA for AI compute), the
    derivatives layer (ANET for AI networking, VST/CEG for AI power,
    APP for AI ads), AND the second-order plays (suppliers, real-estate
    landlords, etc.). Tickers must be valid US listings.

DO NOT make tool calls. Use ONLY your training + the macro context
(sector rotation, macro regime) the user provides. If a theme is
overhyped but actually rolling over, you can flag it with
strength <= 4 and trending=down — those are useful warnings too.

Skip themes that are perennial generic ("rising rates", "consumer
spending"). Stick to themes a portfolio manager would actively be
positioning around this quarter.

Output ONLY the structured MarketThemes object. The `full_text` field
should render the themes as plain text with this format per theme:

THEME: <name> [strength X/10, trending <up|flat|down>]
<description>
Members: <ticker, ticker, ticker, ...>

Separate themes with a blank line. This text is what downstream LLM
calls (ranker, rebalancer) read as context — keep it concise.\
"""


class MarketThemesAgent:
    def __init__(
        self, provider: Provider, model: str
    ):
        self.agent = AgnoAgent(
            "MarketThemes",
            provider,
            model,
            model_kwargs={
                "temperature": 0,
                "retries": 3,
                "exponential_backoff": True,
                "delay_between_retries": 10,
            },
            instructions=MARKET_THEMES_INSTRUCTIONS,
            output_schema=MarketThemes,
        )

    def detect(
        self,
        macro_summary: str = "",
        sector_rotation: dict | None = None,
    ) -> MarketThemes | None:
        sector_block = ""
        if sector_rotation:
            leaders = ", ".join(sector_rotation.get("leaders", []))
            laggards = ", ".join(sector_rotation.get("laggards", []))
            sector_block = (
                f"Sector rotation (6-month returns):\n"
                f"  Leaders: {leaders or '(none)'}\n"
                f"  Laggards: {laggards or '(none)'}\n\n"
            )
        macro_block = (
            f"Macro regime context:\n{macro_summary}\n\n"
            if macro_summary else ""
        )
        prompt = (
            f"{macro_block}"
            f"{sector_block}"
            f"List the dominant equity themes RIGHT NOW per your "
            f"instructions."
        )
        logger.info("Detecting market themes")
        result = self.agent.run(prompt).content
        if result is None:
            logger.warning("MarketThemes returned no content")
            return None
        if isinstance(result, MarketThemes):
            return result
        if isinstance(result, str):
            try:
                return MarketThemes.model_validate_json(result)
            except Exception as e:
                logger.warning(
                    "MarketThemes returned a string that wasn't valid "
                    "MarketThemes JSON: %s", e,
                )
                return None
        logger.warning(
            "MarketThemes returned unexpected type %s", type(result).__name__,
        )
        return None


def themes_by_ticker(themes: MarketThemes | None) -> dict[str, list[dict]]:
    """Invert the {theme -> tickers} mapping into {ticker -> [theme info]}.

    Each entry holds enough to render in the analyst payload + score the
    candidate by theme strength.
    """
    out: dict[str, list[dict]] = {}
    if themes is None:
        return out
    for theme in themes.themes:
        for ticker in theme.member_tickers:
            out.setdefault(ticker.upper(), []).append({
                "name": theme.name,
                "strength": theme.strength,
                "trending": theme.trending,
            })
    return out


def theme_score_bonus(
    ticker: str, by_ticker: dict[str, list[dict]]
) -> tuple[float, dict[str, object]]:
    """Compute a small additive score bonus for a ticker based on the
    max theme strength it's a member of.

    Returns (bonus, breakdown). bonus is 0..6 — bounded so it nudges
    rather than overrides the fundamentals/trend/conviction signal.
    Up-trending themes get a half-point extra; down-trending themes
    get penalized by half a point.
    """
    matches = by_ticker.get((ticker or "").upper()) or []
    if not matches:
        return 0.0, {"theme_count": 0, "max_strength": 0, "best_theme": None}
    # Sort by strength descending so the strongest theme wins; up/flat/down
    # adjusts the contribution.
    best = max(matches, key=lambda m: m["strength"])
    base = float(best["strength"]) * 0.5  # 1..5 score points per theme strength
    trend_adj = {"up": 1.0, "flat": 0.0, "down": -0.5}[best["trending"]]
    bonus = max(0.0, min(6.0, base + trend_adj))
    return round(bonus, 2), {
        "theme_count": len(matches),
        "max_strength": best["strength"],
        "best_theme": best["name"],
        "trending": best["trending"],
    }


__all__ = [
    "MarketThemesAgent",
    "themes_by_ticker",
    "theme_score_bonus",
]
