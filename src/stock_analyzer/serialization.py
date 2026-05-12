"""orjson-backed JSON serialization for LLM prompt payloads.

orjson is 3-5x faster than stdlib `json` on the deeply-nested
fundamentals + tax-lot + technicals dicts we push into reviewer/analyst
prompts. It falls back to stdlib `json.dumps` only when orjson rejects
the payload (it strictly rejects NaN/Inf floats, which yfinance
occasionally emits for missing data — stdlib silently accepts them).
"""
from __future__ import annotations

import json
from typing import Any

import orjson


def dumps_pretty(payload: Any) -> str:
    """Pretty-print a payload to JSON, matching the old stdlib behavior.

    orjson serializes `date`, `datetime`, `UUID`, and `dataclasses`
    natively; anything else falls back to `str()` (so Decimal, Pydantic
    models, np scalars etc. still round-trip rather than raising)."""
    try:
        return orjson.dumps(
            payload,
            default=str,
            option=orjson.OPT_INDENT_2,
        ).decode("utf-8")
    except (TypeError, ValueError):
        # NaN/Inf or some other type orjson refuses — stdlib accepts both
        # and the LLM doesn't care, so preserve the old behavior.
        return json.dumps(payload, default=str, indent=2)
