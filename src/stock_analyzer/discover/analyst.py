"""Per-candidate analyst (Sonnet, parallel fan-out).

Structured analysis per ticker: competitive position, growth runway, top 3
risks (extracted from 10-K + news), valuation context, catalyst calendar.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import (
    AgnoAgent,
    Provider,
    deterministic_model_kwargs,
    fallback_builder,
    run_with_fallback,
)
from ..logging import get_logger
from ..models.llm import AnalystReport
from ..serialization import dumps_pretty

logger = get_logger(__name__)

# Sonnet has tight per-minute input-token rate limits (~30k/min on standard
# tier). With ~7k-token payloads per call, 5 workers burst through the
# quota immediately. Throttle to 2 + lean on retry-with-backoff for residual
# bursts. Slightly slower wall clock, but no 429s.
_MAX_WORKERS = 2

ANALYST_INSTRUCTIONS = """\
You are an equity research analyst evaluating ONE ticker for a LONG-TERM
(3-5 year) hold. Weigh the durability of the business, its growth runway,
balance sheet and valuation against its long-run earnings power; treat
recent price action and technicals as timing color only.
The user provides:
  - fundamentals (including FORWARD: forward_eps, target prices,
    recommendation_mean, earnings_growth_yoy)
  - technicals
  - universe / conviction signals
  - share-trade signals (insider + institutional accumulation)
  - risk_factors_10k (annual 10-K Item 1A)
  - quarterly_mda (latest 10-Q Management Discussion — most current narrative)
  - peers (3-4 closest competitors with their forward fundamentals)
  - earnings_transcript (excerpt from the most recent earnings call)
  - recent_news (last ~30 days, newest first; each item has an `id` like
    "N1", a published_date, source outlet and article snippet)
  - news (bare headlines, no dates — weakest signal)

GROUND your reasoning in this data hierarchy when forming forward thesis:
  1. recent_news — what has CHANGED in the last 30 days (guidance changes,
     deals, product launches, regulatory rulings, management departures)
  2. quarterly_mda — what management said LAST QUARTER
  3. earnings_transcript — management TONE and Q&A pushback signals
  4. peers — judge "cheap" or "expensive" relative to the comp set, not absolute
  5. forward fundamentals — analyst stance + forward EPS revisions
  6. risk_factors_10k — what could go wrong (use cautiously; many risks are boilerplate)

If recent_news contradicts older data (e.g. guidance cut after the last
10-Q), the newer item wins — say so explicitly.

UPCOMING CATALYSTS (structured `upcoming_catalysts` list):
Name up to 5 FUTURE events in the next ~12 months that could move the
stock: next earnings, guidance/investor days, product launches, FDA or
regulatory decisions, contract awards, lockup expiries, index changes.
  - `source` MUST be the recent_news id you took it from ("news:N2"), or
    "quarterly_mda", "earnings_transcript", or "earnings_calendar" (for the
    earnings_alert / next earnings date). Anything else is discarded.
  - `expected_date` only when the source states a date or month
    (YYYY-MM-DD, or YYYY-MM). Otherwise null. Never guess a date. Events
    that already happened are NOT upcoming — use them in the thesis instead.
  - `direction` is your read of the likely price effect; use "uncertain"
    for binary events (FDA decisions, court rulings).
  - Empty list is correct when the inputs name no forward events.

DO NOT make tool calls. Use ONLY the data provided. Be terse; analytical not
promotional. For any field that is null/missing in the input, omit it from
your output rather than guessing.

The conviction score MUST be forward-looking and calibrated:
  1-3: would not own / clear pass
  4-5: borderline, mostly watch
  6-7: solid 3-5 year holding, moderate conviction (typical for good names)
  8-9: high conviction — multiple aligned forward signals, scarce/rare
  10:  essentially impossible to use; do not produce 10 without
       multiple independent corroborations

Anchor on FORWARD evidence: forward EPS growth, target price upside,
analyst stance trend, catalyst calendar — not just trailing performance.

Output EXACTLY this plain-text structure, nothing else:

TICKER: <symbol>
Score: <integer 1-10 conviction>
One-liner: <single sentence, no fluff>

Competitive position:
<1-2 sentences on moat / market position / what's hard to replicate>

Growth runway:
<1-2 sentences on 3-5 year revenue/profit drivers from the data>

Top 3 risks:
1. <risk extracted from 10-K risk factors or news, concrete not generic>
2. ...
3. ...

Valuation context:
<1-2 sentences: PE / FCF yield vs peers or historical, is it stretched>

Catalyst calendar:
<dated upcoming events from upcoming_catalysts, soonest first, with source ids>

CRITICAL:
- Plain text. No markdown headings or bold.
- Begin reply with "TICKER:" line. No preamble, no closing remarks.

CITATION RULE (anti-hallucination):
Every numerical claim in your output (forward EPS, P/E, target price,
growth rate, margin, debt ratio, etc.) MUST appear in the input JSON
the user provided. Do not invent numbers, estimate values not in the
data, or recall figures from training. If a number isn't in the
payload, write "not provided" or omit the claim. The same applies to
named events: cite the specific risk_factor / quarterly_mda /
earnings_transcript line you're referencing, not a generic recollection.

STRUCTURED OUTPUT:
Your response is validated against a Pydantic schema (AnalystReport).
Populate every required field, plus `upcoming_catalysts` per the rules
above. The prose plan you would have emitted goes into `full_text` —
make it match the format described above exactly. The structured fields
must agree with `full_text` — if `full_text` says "Score: 7" then
`score` MUST be 7.

WORKED EXAMPLE — what GOOD looks like (fictional ticker ACME):

TICKER: ACME
Score: 7
One-liner: Niche industrial automation supplier with sticky OEM contracts, trading at a discount to peers on resilient forward EPS.

Competitive position:
ACME holds 38% share in factory-floor motion controllers, a category with multi-year switching costs once installed into a line. Two of the three largest auto OEMs have just renewed framework agreements through 2028.

Growth runway:
Management guided 12-14% revenue CAGR through 2028 driven by China reshoring and a software-attach motion (subscription unlocks). Forward EPS revisions up 6% in the last 90 days per fundamentals.recommendation_mean.

Top 3 risks:
1. Single-customer concentration: top OEM accounts for 22% of revenue (10-K Item 1A); a single-line audit failure would dent FY guide.
2. China macro: 28% of revenue is mainland China; a sharper PMI contraction would compress 2026 EPS by ~9% based on segment elasticity disclosed in quarterly_mda.
3. Software-attach margin slipped 180bps YoY per latest earnings_transcript Q&A — pricing power on the subscription tier is unproven.

Valuation context:
17.4x forward P/E vs peer median 22.1x; FCF yield 5.8% vs peer 4.2%. Discount is justified partly by customer concentration but appears overdone relative to forward EPS revisions.

Catalyst calendar:
Next earnings 2026-07-22. Q3 product launch (vision-system add-on) is the largest near-term swing factor; analyst day pre-print expected in June.

WHY THIS IS GOOD: every numeric claim ties back to a named input field
(fundamentals.forward_eps, quarterly_mda, earnings_transcript). Risks are
specific (22% concentration, 180bps margin slip), not boilerplate
("regulatory risk", "competition", "macro headwinds"). Score 7 — not 9 —
because real risks remain unresolved.

`contracted_book` is revenue already under signed order and not yet
delivered, taken from the company's own SEC filing (remaining performance
obligations), with the quarter it describes and the date it was filed.
It is the only forward figure in the payload that is not a forecast, so
weigh it above analyst targets when the two disagree, and say so when a
growing book contradicts a falling price. Two cautions: it is a quarterly
disclosure, so it lags; and roughly half of all companies never tag the
concept, so `null` means "not disclosed" and is never evidence against a
name.

COMMON FAILURE MODES TO AVOID:
- "Faces competition in a rapidly evolving market" — boilerplate, no signal.
  Better: "Competitor X disclosed a 30% price cut in Q4 transcript."
- "Margins may compress due to inflation" — generic. Better: "Gross margin
  guided down 120bps for Q3 per latest 10-Q MD&A."
- Citing a P/E without comparing to peers — incomplete. Always anchor on
  the peer set provided in the payload.
- Score 8-10 on a name with a single forward signal — overcalibrated.\
"""


def _build_agent(provider: Provider, model: str) -> AgnoAgent:
    model_kwargs: dict[str, Any] = {
        **deterministic_model_kwargs(provider),
        "retries": 3,
        "exponential_backoff": True,
        "delay_between_retries": 10,
    }
    if provider == "claude":
        model_kwargs["cache_system_prompt"] = True
    return AgnoAgent(
        "Analyst",
        provider,
        model,
        model_kwargs=model_kwargs,
        instructions=ANALYST_INSTRUCTIONS,
        output_schema=AnalystReport,
    )


class Analyst:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        fallback: tuple[Provider, str] | None = None,
    ):
        self.provider = provider
        self.fallback = fallback
        self.agent = _build_agent(provider, model)

    def analyze(self, ticker: str, payload: dict[str, Any]) -> AnalystReport | None:
        prompt = f"Candidate ticker: {ticker}\n\n```json\n{dumps_pretty(payload)}\n```"
        logger.info("Analyzing %s", ticker)
        build_fallback = fallback_builder(self.fallback, self.provider, _build_agent)
        result = run_with_fallback(self.agent, build_fallback, prompt).content
        if result is None:
            logger.warning("Analyst returned no content for %s — skipping", ticker)
            return None
        if isinstance(result, AnalystReport):
            return result
        if isinstance(result, str):
            try:
                return AnalystReport.model_validate_json(result)
            except Exception as e:
                logger.warning(
                    "Analyst for %s returned a string that wasn't valid AnalystReport JSON: %s",
                    ticker,
                    e,
                )
                return None
        logger.warning(
            "Analyst for %s returned unexpected type %s; skipping",
            ticker,
            type(result).__name__,
        )
        return None


def analyze_tiered(
    deep: Analyst,
    light: Analyst | None,
    payloads: dict[str, dict[str, Any]],
    deep_tickers: set[str],
) -> dict[str, AnalystReport]:
    """`deep` analyzes `deep_tickers`, `light` (a cheaper model) the rest.

    Any ticker the light model fails on — a provider error, or a response
    that doesn't validate as AnalystReport — is retried on `deep`, so the
    cheaper tier can only save cost, never drop a candidate. Result order
    follows `payloads` (the screen-score order the Ranker reads in).
    """
    if light is None:
        deep_tickers = set(payloads)
    results = analyze_batch(deep, {t: p for t, p in payloads.items() if t in deep_tickers})
    light_payloads = {t: p for t, p in payloads.items() if t not in deep_tickers}
    if light is not None and light_payloads:
        results.update(analyze_batch(light, light_payloads))
        missing = {t: p for t, p in light_payloads.items() if t not in results}
        if missing:
            logger.warning(
                "Light-tier analyst failed for %s — retrying on the deep model",
                sorted(missing),
            )
            results.update(analyze_batch(deep, missing))
    return {t: results[t] for t in payloads if t in results}


def analyze_batch(
    analyst: Analyst, payloads: dict[str, dict[str, Any]]
) -> dict[str, AnalystReport]:
    """Return {ticker: AnalystReport} for every payload the Analyst
    successfully scored. Failures are silently excluded."""
    results: dict[str, AnalystReport] = {}
    items = list(payloads.items())
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {ex.submit(analyst.analyze, t, p): t for t, p in items}
        for fut in futures:
            ticker = futures[fut]
            try:
                report = fut.result()
            except Exception as e:
                logger.warning("analyst failed for %s: %s", ticker, e)
                continue
            if report is not None:
                results[ticker] = report
    return results


# Planning estimate of one scorecard's output (the report plus its
# structured fields), not the hard ceiling the cost cap reserves per call.
EXPECTED_OUTPUT_TOKENS = 2000
MIN_NAMES_UNDER_BUDGET = 5


def plan_under_budget(
    order: list[str],
    payload_chars: dict[str, int],
    deep_tickers: set[str],
    deep_model: str,
    light_model: str | None,
    available_usd: float | None,
) -> tuple[list[str], set[str], list[str]]:
    """Fit the analyst batch into `available_usd`, cheapest cut first:
    move the deep tier to the light model, then drop the lowest-scored
    names (never below MIN_NAMES_UNDER_BUDGET). `order` is screen-score
    order. Returns (tickers to analyze, deep tier, notes on what was cut).
    Unpriced models count as free, so they are never cut."""
    from ..usage import estimate_cost

    def cost(tickers: list[str], deep: set[str]) -> float:
        total = 0.0
        for t in tickers:
            model = deep_model if (t in deep or light_model is None) else light_model
            est = estimate_cost(
                model, payload_chars[t] + len(ANALYST_INSTRUCTIONS), EXPECTED_OUTPUT_TOKENS
            )
            total += est or 0.0
        return total

    tickers, deep = list(order), set(deep_tickers)
    notes: list[str] = []
    if available_usd is None or cost(tickers, deep) <= available_usd:
        return tickers, deep, notes
    if light_model is not None and deep:
        notes.append(
            f"analysed the top {len(deep)} names on {light_model} instead of {deep_model} "
            f"(estimated ${cost(tickers, deep):.2f} vs ${available_usd:.2f} available)"
        )
        deep = set()
    n = len(tickers)
    while len(tickers) > MIN_NAMES_UNDER_BUDGET and cost(tickers, deep) > available_usd:
        tickers.pop()
    if len(tickers) < n:
        notes.append(f"analysed only the top {len(tickers)} of {n} survivors by screen score")
    return tickers, deep, notes
