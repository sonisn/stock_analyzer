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
from ..models.llm import MarketThemes

logger = get_logger(__name__)


MARKET_THEMES_INSTRUCTIONS = """\
You are a market analyst. The user provides EXPLICIT, RECENT market
data — top and bottom performers in their universe, EPS-revision
direction per ticker, sector ETF returns, macro regime. Your job:
**identify themes that are visible in THIS data**, NOT from your
training memory.

Each theme should:
  - Be specific enough to map to tickers (not "growth stocks" but
    "AI compute capex" or "GLP-1 weight-loss drugs").
  - Be supported by tickers in the TOP_PERFORMERS list (for up themes)
    or BOTTOM_PERFORMERS list (for down themes) provided to you.
  - Have at least 5 named members that all appear in the user's data.

For each theme, populate the MarketTheme schema:
  - name: concise label (2-4 words)
  - description: 1-2 sentences citing the specific tickers from the
    user's data that support this theme. e.g. "NVDA +18% / AVGO +12% /
    AMD +15% all in top performers + raising EPS revisions confirm
    AI-compute capex tailwind."
  - strength (1-10): based on what fraction of the named members
    appear in the TOP_PERFORMERS list AND have raising EPS revisions.
    10 = nearly all members are top performers with raising revisions;
    1-3 = members are mostly bottom performers (theme is rolling over).
  - trending (up | flat | down): direction based on whether named
    members are dominantly in TOP, mixed, or BOTTOM performers.
  - member_tickers: 10-25 US-listed tickers. Prioritize tickers from
    the provided TOP/BOTTOM/REVISIONS lists. You may add 3-5 obvious
    related names not in those lists if needed for theme coherence,
    but MUST clearly identify them as "[add]" via a comment in the
    description.

ANTI-HALLUCINATION RULES (these are hard constraints):
  - Every ticker you name as a member MUST be either (a) in one of
    the lists the user provided, or (b) an obvious adjacent name in
    the same sub-industry (and you must mention so in the description).
  - Every numerical claim in the description MUST reference data the
    user gave you. Don't invent percentages.
  - If you cannot find 3+ themes supported by the provided data,
    output fewer themes. Do not invent themes that aren't visible.

If a theme that's well-known (e.g. "AI compute") has NO supporting
tickers in the user's data, DO NOT include it. The data is the source
of truth, not your priors.

Output ONLY the structured MarketThemes object. The `full_text` field
should render the themes as plain text with this format per theme:

THEME: <name> [strength X/10, trending <up|flat|down>]
<description with specific tickers + numbers from the data>
Members: <ticker, ticker, ticker, ...>

Separate themes with a blank line. This text is what downstream LLM
calls (ranker, rebalancer) read as context — keep it concise.\
"""


def _format_top_performers(
    technicals: dict, fundamentals: dict, *, top_n: int = 20
) -> str:
    """List top-N tickers by rs_6mo with actual relative-return numbers.

    Format:
      TOP_PERFORMERS (sorted by 6-month relative strength vs SPY):
        NVDA  +24.5%  Tech / Semiconductors
        AVGO  +18.2%  Tech / Semiconductors
        ...
    """
    rows: list[tuple[str, float, str, str]] = []
    for ticker, t in technicals.items():
        rs6 = t.get("rs_6mo")
        if rs6 is None:
            continue
        f = (fundamentals.get(ticker) or {})
        sector = str(f.get("sector") or "—")
        industry = str(f.get("industry") or "—")
        rows.append((ticker, float(rs6), sector, industry))
    rows.sort(key=lambda r: r[1], reverse=True)
    if not rows:
        return "TOP_PERFORMERS: (no relative-strength data available)\n"
    out = "TOP_PERFORMERS (sorted by 6-month return vs SPY):\n"
    for ticker, rs6, sector, industry in rows[:top_n]:
        out += f"  {ticker:6s}  {rs6*100:+6.1f}%  {sector} / {industry}\n"
    return out


def _format_bottom_performers(
    technicals: dict, fundamentals: dict, *, bottom_n: int = 15
) -> str:
    """List bottom-N tickers by rs_6mo. Used to flag rolling-over themes."""
    rows: list[tuple[str, float, str, str]] = []
    for ticker, t in technicals.items():
        rs6 = t.get("rs_6mo")
        if rs6 is None:
            continue
        f = (fundamentals.get(ticker) or {})
        sector = str(f.get("sector") or "—")
        industry = str(f.get("industry") or "—")
        rows.append((ticker, float(rs6), sector, industry))
    rows.sort(key=lambda r: r[1])
    if not rows:
        return "BOTTOM_PERFORMERS: (no relative-strength data available)\n"
    out = "BOTTOM_PERFORMERS (sorted by 6-month return vs SPY, worst first):\n"
    for ticker, rs6, sector, industry in rows[:bottom_n]:
        out += f"  {ticker:6s}  {rs6*100:+6.1f}%  {sector} / {industry}\n"
    return out


def _format_revisions_summary(eps_revisions: dict) -> str:
    """List tickers with raising and lowering EPS revisions in last 30 days."""
    raising: list[str] = []
    lowering: list[str] = []
    for ticker, r in eps_revisions.items():
        direction = r.get("direction_30d")
        net = r.get("net_revisions_30d", 0)
        if direction == "raising":
            raising.append(f"{ticker} (+{net})")
        elif direction == "lowering":
            lowering.append(f"{ticker} ({net})")
    out = "EPS_REVISIONS_30D (net analyst ups - downs):\n"
    out += f"  Raising: {', '.join(raising[:30]) if raising else '(none)'}\n"
    out += f"  Lowering: {', '.join(lowering[:15]) if lowering else '(none)'}\n"
    return out


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
        technicals: dict | None = None,
        fundamentals: dict | None = None,
        eps_revisions: dict | None = None,
    ) -> MarketThemes | None:
        """Identify themes that are visible in the user's actual data.

        We hand the LLM:
          - Top-20 performers by rs_6mo with their actual returns
          - Bottom-15 performers (rolling-over themes)
          - Tickers with raising EPS revisions (forward-thesis confirmation)
          - Tickers with lowering EPS revisions (rolling over)
          - Sector ETF returns
          - Macro regime snippet
        Sonnet then summarizes the themes ALREADY VISIBLE in this data
        rather than recalling them from training memory.
        """
        top_block = _format_top_performers(technicals or {}, fundamentals or {}, top_n=20)
        bottom_block = _format_bottom_performers(technicals or {}, fundamentals or {}, bottom_n=15)
        revisions_block = _format_revisions_summary(eps_revisions or {})
        sector_block = ""
        if sector_rotation:
            leaders = ", ".join(sector_rotation.get("leaders", []))
            laggards = ", ".join(sector_rotation.get("laggards", []))
            sector_block = (
                f"SECTOR ROTATION (6-month sector ETF returns):\n"
                f"  Leaders: {leaders or '(none)'}\n"
                f"  Laggards: {laggards or '(none)'}\n\n"
            )
        macro_block = (
            f"MACRO REGIME:\n{macro_summary}\n\n"
            if macro_summary else ""
        )
        prompt = (
            f"{macro_block}"
            f"{sector_block}"
            f"{top_block}\n"
            f"{bottom_block}\n"
            f"{revisions_block}\n"
            f"Based on THIS data — what themes are visible? Identify "
            f"3-8 themes that are supported by the actual tickers + "
            f"numbers above. Do not include themes that have no "
            f"supporting evidence in this data."
        )
        logger.info("Detecting market themes (grounded in %d performers, %d revisions)",
                    len(technicals or {}), len(eps_revisions or {}))
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
