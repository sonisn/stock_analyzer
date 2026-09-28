"""Short reads of SEC event filings (data/sec_events): what a Schedule 13D
holder is after, and what a 424B5 offering sells. Open reader model, same
host routing and daily cap as the filing reader; a reply that isn't
usable JSON is simply left out of the alert."""

from __future__ import annotations

from typing import Any

from ..logging import get_logger
from ..openrouter import OpenRouter, parse_json_object
from .filing_reader import READER_EXTRA, normalise, quote_found

logger = get_logger(__name__)

ACTIVIST_INSTRUCTIONS = """\
You classify a Schedule 13D (a 5%+ holder's filing) for a long-term
investor. Use ONLY the text given; do not guess at intentions it does not
state. Return ONE JSON object, no prose:
{
  "stance": "activist|strategic|founder_or_insider|merger_related|passive|unclear",
  "headline": "<=15 words: who holds how much and why",
  "demands": "<=40 words: board seats, sale, buybacks, strategy change... or \\"none stated\\"",
  "quote": "one sentence copied verbatim from the text that shows the stance, or \\"\\""
}
"activist" = the holder names or pursues SPECIFIC changes at the company:
board seats, a sale or break-up, a buyback or dividend, replacing
management, a strategy change, opposing a deal. Routine "engagement" on
governance or sustainability by an asset manager, a holder with "no
current plans", a founder, the company's own insiders and a merger
counterparty are not activist — "passive", "founder_or_insider" or
"merger_related".\
"""

OFFERING_INSTRUCTIONS = """\
You read the cover and summary of a 424B5 prospectus supplement for an
existing shareholder. Use ONLY the text given. Return ONE JSON object:
{
  "security": "common|preferred|debt|convertible|units|warrants|other",
  "at_the_market": true|false,
  "amount_usd_millions": <number or null>,
  "shares_millions": <number or null>,
  "price_per_share": <number or null>,
  "use_of_proceeds": "<=25 words",
  "headline": "<=15 words",
  "quote": "the sentence, copied verbatim, that states the amount or share count"
}
"at_the_market" is true for an equity distribution / at-the-market program
(shares sold over time at market prices).\
"""
OFFERING_CHARS = 14_000
EVENT_MAX_TOKENS = 4000


def _read(
    client: OpenRouter, stage: str, model: str, system: str, text: str
) -> dict[str, Any] | None:
    reply = client.complete(
        stage, model, system, text, max_tokens=EVENT_MAX_TOKENS, extra=READER_EXTRA
    )
    return parse_json_object(reply.text)


def read_13d(
    client: OpenRouter, filing: dict[str, Any], parsed: dict[str, Any], *, model: str
) -> dict[str, Any] | None:
    text = (
        f"Company: {filing['ticker']}  Form: {filing['form']}  Filed: {filing['filed_on']}\n"
        f"Holders: {', '.join(parsed.get('holders') or []) or 'n/a'}\n"
        f"Percent of class: {parsed.get('percent')}\n\n"
        f"Item 4 (purpose): {parsed.get('purpose') or '(not in this filing)'}\n\n"
        f"Items 1-7: {parsed.get('items_text') or ''}"
    )
    out = _read(client, "ActivistReader", model, ACTIVIST_INSTRUCTIONS, text)
    if out is None:
        return None
    if out.get("quote") and not quote_found(out["quote"], normalise(text)):
        out["quote"] = ""  # an unverifiable quote is dropped, not shown
    return out


def read_offering(
    client: OpenRouter, filing: dict[str, Any], body: str, *, model: str
) -> dict[str, Any] | None:
    text = (
        f"Company: {filing['ticker']}  Form: {filing['form']}  Filed: {filing['filed_on']}\n\n"
        + body[:OFFERING_CHARS]
    )
    out = _read(client, "OfferingReader", model, OFFERING_INSTRUCTIONS, text)
    if out is None:
        return None
    if out.get("quote") and not quote_found(out["quote"], normalise(text)):
        out["quote"] = ""
    return out
