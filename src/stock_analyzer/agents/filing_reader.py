"""Read a 10-Q/10-K/20-F into structured facts with an open model.

The layer below the final verdict. Two readers, chosen against Claude on
the same ten filings (2026-09-27):

  - GLM-5.3 (92% field agreement with Claude, ~$0.0125 a filing) for the
    stocks acted on: holdings, picks, the shortlist, market leaders;
  - GLM-5.3-Flash (83%, ~$0.002) for the rest of the universe. A Flash
    read that reports a serious event is read again on GLM-5.3, which is
    what replaces it (cli/filings.py), so nothing reaches a decision on
    the cheaper read alone.

Every field is backed by a quote, checked against the filing text in code
— a quote that isn't there is how an invented number shows up.

Nothing here decides anything. The facts feed the evidence the deciding
models (Claude, Gemini, OpenAI) receive.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..http_client import HttpClientError
from ..logging import get_logger
from ..openrouter import OpenRouter, parse_json_object

logger = get_logger(__name__)

# ~26k tokens in all.
SECTION_CHARS = {"mda": 60_000, "risks": 30_000}
# GLM reasons before it answers: at 3,000 tokens GLM-5.2 ran out on 10 of
# 27 filings and returned no answer at all (billed anyway).
READER_MAX_TOKENS = 8_000
# fp8 or better, cheapest host first. Compressed (fp4) copies are cheaper
# still but not what was tested; fp8 routing scored best (33/36) and cost
# a third less than letting OpenRouter pick.
READER_EXTRA: dict[str, Any] = {
    "reasoning": {"effort": "low"},
    "provider": {
        "quantizations": ["fp8", "bf16", "fp16"],
        "sort": "price",
        "allow_fallbacks": True,
    },
}
# Second try after an empty answer: thinking off, so it can't run out again.
RETRY_EXTRA: dict[str, Any] = {**READER_EXTRA, "reasoning": {"enabled": False}}
# Some hosts refuse to turn thinking off (HTTP 400 "Reasoning is mandatory");
# there the retry keeps it on with twice the room instead.
RETRY_MANDATORY_REASONING_TOKENS = 2 * READER_MAX_TOKENS
# A caveat escalates only when it names an event, not a standing risk.
CAVEAT_CATEGORIES = (
    "going_concern",
    "material_weakness",
    "restatement",
    "auditor_change",
    "investigation",
    "customer_loss",
    "covenant",
    "impairment",
    "guidance_cut",
    "demand_drop",
    "margin_drop",
    "major_litigation",
)
# Below this share of quotes found in the filing, a bulk read escalates.
MIN_QUOTE_HIT_RATE = 0.8

READER_INSTRUCTIONS = """\
You extract facts from a US company's SEC filing for a long-term (3-5 year)
investor. You do NOT give advice or a verdict. Report only what the filing
says; if it does not address a field, say so ("not_disclosed" / null) rather
than inferring.

Every "quote" must be copied VERBATIM from the filing text you are given —
one sentence or table row, at most 40 words. Quotes are checked by machine
against the filing; a paraphrase counts as a fabrication. Use "" when there
is nothing to quote.

Return ONE JSON object, no prose, with exactly these keys:
{
  "guidance": {"direction": "raised|maintained|lowered|withdrawn|none_given",
               "detail": "<=30 words", "quote": "..."},
  "demand": {"direction": "accelerating|steady|slowing|unclear",
             "detail": "<=30 words: revenue drivers, segment growth", "quote": "..."},
  "margins": {"direction": "expanding|stable|contracting|unclear",
              "detail": "<=30 words", "quote": "..."},
  "backlog": {"value_usd_millions": <number or null>,
              "change": "growing|flat|shrinking|not_disclosed",
              "detail": "<=30 words: backlog, RPO, bookings", "quote": "..."},
  "liquidity": {"concern": true|false, "detail": "<=30 words: cash, debt,
                covenants, going concern", "quote": "..."},
  "capital_return": {"detail": "<=25 words: buybacks, dividends", "quote": "..."},
  "key_risks": [{"risk": "<=20 words, specific to THIS company, not
                 boilerplate", "quote": "..."}],          (at most 3)
  "one_offs": [{"item": "<=20 words: impairment, restructuring, litigation
               charge, tax item", "quote": "..."}],       (at most 3)
  "caveats": [{"issue": "<=25 words", "category": "{categories}",
              "severity": "high|medium", "quote": "..."}],
  "tone": "positive|neutral|cautious|negative",
  "summary": "<=2 sentences: what changed this period"
}

"caveats" are EVENTS reported in this filing that an owner must not miss,
each in one of the categories above: going-concern doubt, material weakness
in controls, restatement, auditor change, a disclosed SEC/DOJ investigation,
loss of a major customer, covenant breach, a large impairment, a guidance cut
or withdrawal, sharp demand or margin deterioration, litigation that could
cost a material share of earnings. "high" means it could change the
investment case.

Standing risk-factor language is NOT a caveat, however serious it sounds:
macro conditions, geopolitics, tariffs, regulation, competition, supply
chain, customer concentration, cybersecurity "could" statements. Those go
in key_risks. Most filings have no caveats; an empty list is the normal
answer.\
""".replace("{categories}", "|".join(CAVEAT_CATEGORIES))

# --- quote checking ----------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """Lowercase letters and digits, single-spaced: survives the curly
    quotes, dashes, table pipes and line breaks the text conversion adds."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def quote_found(quote: str, norm_source: str) -> bool:
    """Is `quote` in the filing? Exact after normalising, or — for a quote
    that stitches two table cells or drops a word — most of its five-word
    runs are there. A made-up sentence shares almost none."""
    q = normalise(quote)
    if not q:
        return True
    if q in norm_source:
        return True
    words = q.split()
    if len(words) < 6:
        return False
    runs = [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]
    return sum(r in norm_source for r in runs) / len(runs) >= 0.6


# What a host's privacy filter leaves in place of a name or place
# ("[ADDRESS]", "<PERSON>"). A filing never contains one, so a reply that
# does was read from altered input (seen 2026-09-28 on two GLM-5.3 hosts).
_PLACEHOLDER = re.compile(
    r"[\[<](?:ADDRESS|NAME|PERSON|LOCATION|EMAIL|PHONE|ORGANI[SZ]ATION|REDACTED)[^\]>]{0,20}[\]>]"
)


def has_placeholder(reply: Any) -> bool:
    return bool(_PLACEHOLDER.search(json.dumps(reply)))


def _quotes(value: Any) -> list[str]:
    """Every "quote" string anywhere in a reply."""
    if isinstance(value, dict):
        out = [value["quote"]] if isinstance(value.get("quote"), str) else []
        for k, v in value.items():
            if k != "quote":
                out.extend(_quotes(v))
        return out
    if isinstance(value, list):
        return [q for item in value for q in _quotes(item)]
    return []


def check_quotes(reply: dict[str, Any], norm_source: str) -> tuple[int, int, list[str]]:
    """(quotes checked, quotes found, the ones not found). Empty quotes
    (nothing to quote) are not counted."""
    quotes = [q for q in _quotes(reply) if q.strip()]
    missing = [q for q in quotes if not quote_found(q, norm_source)]
    return len(quotes), len(quotes) - len(missing), missing


# --- the read ----------------------------------------------------------------


def filing_prompt(filing: dict[str, Any], sections: dict[str, str]) -> str:
    parts = [
        f"Company: {filing['ticker']}  Form: {filing['form']}  "
        f"Period ended: {filing.get('period_end') or 'n/a'}  Filed: {filing['filed_on']}"
    ]
    if filing.get("income_drops"):
        # Tagged figures from this filing (data/income_drop): say what drove them.
        parts.append(
            "Income as filed (XBRL): "
            + "; ".join(d.removeprefix("filed ") for d in filing["income_drops"])
        )
    if "mda" in sections:
        parts.append("=== MANAGEMENT'S DISCUSSION AND ANALYSIS ===\n" + sections["mda"])
    if "risks" in sections:
        parts.append("=== RISK FACTORS ===\n" + sections["risks"])
    return "\n\n".join(parts)


def flag_reasons(facts: dict[str, Any] | None, checked: int, found: int) -> list[str]:
    """Why a bulk read goes to the better reader; empty = it stands."""
    if facts is None:
        return ["reader reply was not usable JSON"]
    reasons = []
    for c in facts.get("caveats") or []:
        if (
            isinstance(c, dict)
            and c.get("severity") == "high"
            and c.get("category") in CAVEAT_CATEGORIES
        ):
            reasons.append(f"high-severity caveat: {str(c.get('issue', ''))[:80]}")
    direction = (facts.get("guidance") or {}).get("direction")
    if direction in ("lowered", "withdrawn"):
        reasons.append(f"guidance {direction}")
    if (facts.get("liquidity") or {}).get("concern") is True:
        reasons.append("liquidity concern")
    if checked and found / checked < MIN_QUOTE_HIT_RATE:
        reasons.append(f"only {found}/{checked} quotes found in the filing")
    if has_placeholder(facts):
        reasons.append("reply has a redaction placeholder: the host altered the filing text")
    return reasons


@dataclass
class FilingRead:
    filing: dict[str, Any]
    reader_model: str
    facts: dict[str, Any] | None
    quotes_checked: int = 0
    quotes_found: int = 0
    missing_quotes: list[str] = field(default_factory=list)
    flag_reasons: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    provider: str | None = None
    # Set when a bulk read was flagged and this read replaced it.
    escalated_from: str | None = None

    @property
    def flagged(self) -> bool:
        return bool(self.flag_reasons)


def read_filing(
    client: OpenRouter,
    filing: dict[str, Any],
    sections: dict[str, str],
    *,
    model: str,
    retry: bool = True,
    host: str | None = None,
) -> FilingRead:
    """One read on `model`, retried once (unless `retry` is off) if the
    answer comes back empty or unparseable: with thinking off, or — on a
    host that won't allow that — with twice the output room. Raises only
    when a call is refused (over
    the daily cap) or fails."""
    prompt = filing_prompt(filing, sections)
    cost = 0.0
    facts = None
    provider = None

    def attempt(extra: dict[str, Any], max_tokens: int) -> dict[str, Any] | None:
        nonlocal cost, provider
        if host:  # pinned to one host: the known-answer check (openrouter_hosts)
            extra = {
                **extra,
                "provider": {**extra["provider"], "only": [host], "allow_fallbacks": False},
            }
        reply = client.complete(
            "FilingReader", model, READER_INSTRUCTIONS, prompt, max_tokens=max_tokens, extra=extra
        )
        cost += reply.cost_usd
        provider = reply.provider
        return parse_json_object(reply.text)

    facts = attempt(READER_EXTRA, READER_MAX_TOKENS)
    if facts is None and retry:
        logger.info("%s on %s: no usable answer, retrying", filing["ticker"], model)
        try:
            facts = attempt(RETRY_EXTRA, READER_MAX_TOKENS)
        except HttpClientError as e:
            if "mandatory" not in str(e).lower():
                raise
            facts = attempt(READER_EXTRA, RETRY_MANDATORY_REASONING_TOKENS)
    checked, found, missing = check_quotes(facts, normalise(prompt)) if facts else (0, 0, [])
    return FilingRead(
        filing=filing,
        reader_model=model,
        facts=facts,
        quotes_checked=checked,
        quotes_found=found,
        missing_quotes=missing,
        flag_reasons=flag_reasons(facts, checked, found),
        cost_usd=cost,
        provider=provider,
    )


# --- 8-Ks: the filings that can't wait for the weekly read ------------------

# The 8-K items worth an email on a stock you hold. Routine ones —
# shareholder votes (5.07), Reg FD slides (7.01), debt paperwork (2.03),
# "other events" (8.01) alone — are left to the weekly email.
MATERIAL_8K_ITEMS = {
    "1.01": "material agreement",
    "1.02": "agreement terminated",
    "1.03": "bankruptcy",
    "2.01": "acquisition or sale completed",
    "2.02": "results",
    "2.05": "restructuring",
    "2.06": "impairment",
    "3.01": "delisting notice",
    "4.01": "auditor change",
    "4.02": "restatement",
    "5.02": "executive change",
}
EIGHTK_CHARS = {"body": 20_000, "exhibit": 40_000}

EIGHTK_INSTRUCTIONS = """\
You summarize a US company's 8-K (a report of a material event) and its
attached press release for a long-term (3-5 year) shareholder. You do NOT
give advice. Report only what the filing says.

Every "quote" must be copied VERBATIM from the text you are given — one
sentence or table row, at most 40 words. Quotes are checked by machine.
Use "" when there is nothing to quote.

Return ONE JSON object, no prose:
{
  "headline": "<=15 words: what happened",
  "what_happened": "<=60 words",
  "guidance": {"direction": "raised|maintained|lowered|withdrawn|initiated|none_given",
               "detail": "<=30 words", "quote": "..."},
  "numbers": [{"metric": "e.g. revenue", "value": "...", "vs_prior": "e.g. +22% y/y",
               "quote": "..."}],                     (at most 5; results only)
  "events": [{"issue": "<=25 words", "category": "{categories}",
              "severity": "high|medium", "quote": "..."}],
  "tone": "positive|neutral|cautious|negative"
}
"events" are facts an owner must not miss (see the categories); a routine
filing has none. An executive departure is "medium" unless the filing
ties it to a disagreement, an investigation or the results.\
""".replace("{categories}", "|".join(CAVEAT_CATEGORIES))


def eightk_prompt(filing: dict[str, Any], body: str, exhibit: str | None) -> str:
    items = ", ".join(f"{i} ({MATERIAL_8K_ITEMS.get(i, 'other')})" for i in filing["items"])
    parts = [
        f"Company: {filing['ticker']}  Form: 8-K  Filed: {filing['filed_on']}  Items: {items}",
        "=== 8-K ===\n" + body[: EIGHTK_CHARS["body"]],
    ]
    if exhibit:
        parts.append("=== EXHIBIT 99 (press release) ===\n" + exhibit[: EIGHTK_CHARS["exhibit"]])
    return "\n\n".join(parts)


@dataclass
class EightKRead:
    filing: dict[str, Any]
    model: str
    summary: dict[str, Any] | None
    quotes_checked: int = 0
    quotes_found: int = 0
    cost_usd: float = 0.0


def read_8k(
    client: OpenRouter,
    filing: dict[str, Any],
    body: str,
    exhibit: str | None,
    *,
    model: str,
) -> EightKRead:
    prompt = eightk_prompt(filing, body, exhibit)
    reply = client.complete(
        "EightKReader",
        model,
        EIGHTK_INSTRUCTIONS,
        prompt,
        max_tokens=READER_MAX_TOKENS,
        extra=READER_EXTRA,
    )
    summary = parse_json_object(reply.text)
    checked, found, _ = check_quotes(summary, normalise(prompt)) if summary else (0, 0, [])
    return EightKRead(filing, model, summary, checked, found, reply.cost_usd)
