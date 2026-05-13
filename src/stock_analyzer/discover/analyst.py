"""Per-candidate analyst (Sonnet, parallel fan-out).

Structured analysis per ticker: competitive position, growth runway, top 3
risks (extracted from 10-K + news), valuation context, catalyst calendar.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from ..serialization import dumps_pretty
from .schemas import AnalystReport

logger = get_logger(__name__)

# Sonnet has tight per-minute input-token rate limits (~30k/min on standard
# tier). With ~7k-token payloads per call, 5 workers burst through the
# quota immediately. Throttle to 2 + lean on retry-with-backoff for residual
# bursts. Slightly slower wall clock, but no 429s.
_MAX_WORKERS = 2

ANALYST_INSTRUCTIONS = """\
You are an equity research analyst evaluating ONE ticker for a 6-12 month hold.
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
  - news

GROUND your reasoning in this data hierarchy when forming forward thesis:
  1. quarterly_mda — what management said LAST QUARTER (most current)
  2. earnings_transcript — management TONE and Q&A pushback signals
  3. peers — judge "cheap" or "expensive" relative to the comp set, not absolute
  4. forward fundamentals — analyst stance + forward EPS revisions
  5. risk_factors_10k — what could go wrong (use cautiously; many risks are boilerplate)

DO NOT make tool calls. Use ONLY the data provided. Be terse; analytical not
promotional. For any field that is null/missing in the input, omit it from
your output rather than guessing.

The conviction score MUST be forward-looking and calibrated:
  1-3: would not own / clear pass
  4-5: borderline, mostly watch
  6-7: solid 6-12mo bet, moderate conviction (typical for good names)
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
<next earnings date if known, any product/regulatory items from news>

CRITICAL:
- Plain text. No markdown headings or bold.
- Begin reply with "TICKER:" line. No preamble, no closing remarks.

STRUCTURED OUTPUT:
Your response is validated against a Pydantic schema (AnalystReport).
Populate every required field. The prose plan you would have emitted
goes into `full_text` — make it match the format described above
exactly. The structured fields must agree with `full_text` — if
`full_text` says "Score: 7" then `score` MUST be 7.\
"""


class Analyst:
    def __init__(self, provider: Provider, model: str):
        self.agent = AgnoAgent(
            "Analyst",
            provider,
            model,
            model_kwargs={
                "temperature": 0,
                "retries": 3,
                "exponential_backoff": True,
                "delay_between_retries": 10,
            },
            instructions=ANALYST_INSTRUCTIONS,
            output_schema=AnalystReport,
        )

    def analyze(self, ticker: str, payload: dict[str, Any]) -> AnalystReport | None:
        prompt = (
            f"Candidate ticker: {ticker}\n\n"
            f"```json\n{dumps_pretty(payload)}\n```"
        )
        logger.info("Analyzing %s", ticker)
        result = self.agent.run(prompt).content
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
                    "Analyst for %s returned a string that wasn't valid "
                    "AnalystReport JSON: %s", ticker, e,
                )
                return None
        logger.warning(
            "Analyst for %s returned unexpected type %s; skipping",
            ticker, type(result).__name__,
        )
        return None


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
