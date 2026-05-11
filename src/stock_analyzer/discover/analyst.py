"""Per-candidate analyst (Sonnet, parallel fan-out).

Structured analysis per ticker: competitive position, growth runway, top 3
risks (extracted from 10-K + news), valuation context, catalyst calendar.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import AgnoAgent, Provider
from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 5

ANALYST_INSTRUCTIONS = """\
You are an equity research analyst evaluating ONE ticker for a 6-12 month hold.
The user provides fundamentals, technicals, conviction signals, risk factors
extracted from the latest 10-K, and news.

DO NOT make tool calls. Use ONLY the data provided. Be terse; analytical not
promotional. For any field that is null/missing in the input, omit it from
your output rather than guessing.

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
- Begin reply with "TICKER:" line. No preamble, no closing remarks.\
"""


class Analyst:
    def __init__(self, provider: Provider, model: str):
        self.agent = AgnoAgent(
            "Analyst", provider, model, instructions=ANALYST_INSTRUCTIONS
        )

    def analyze(self, ticker: str, payload: dict[str, Any]) -> str:
        prompt = (
            f"Candidate ticker: {ticker}\n\n"
            f"```json\n{json.dumps(payload, default=str, indent=2)}\n```"
        )
        logger.info("Analyzing %s", ticker)
        return self.agent.run(prompt).content


def analyze_batch(
    analyst: Analyst, payloads: dict[str, dict[str, Any]]
) -> dict[str, str]:
    results: dict[str, str] = {}
    items = list(payloads.items())
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {ex.submit(analyst.analyze, t, p): t for t, p in items}
        for fut in futures:
            ticker = futures[fut]
            try:
                results[ticker] = fut.result()
            except Exception as e:
                logger.warning("analyst failed for %s: %s", ticker, e)
    return results
