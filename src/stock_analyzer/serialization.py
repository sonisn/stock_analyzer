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
    except TypeError, ValueError:
        # NaN/Inf or some other type orjson refuses — stdlib accepts both
        # and the LLM doesn't care, so preserve the old behavior.
        return json.dumps(payload, default=str, indent=2)


def finite(payload: Any) -> Any:
    """Replace NaN/Infinity with None, recursively.

    yfinance emits NaN for missing data and stdlib `json` writes it out as
    a bare `NaN` token, which is not valid JSON — it happens to parse as
    JavaScript, so a page embedding it works by luck rather than by
    contract. Turning it into null first means the output is always valid
    JSON, and null is what a reader has to handle anyway.
    """
    if isinstance(payload, float):
        return None if (payload != payload or payload in (float("inf"), float("-inf"))) else payload
    if isinstance(payload, dict):
        return {k: finite(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [finite(v) for v in payload]
    return payload


def dumps_compact(payload: Any) -> str:
    """JSON with no whitespace, for embedding rather than reading.

    Missing numbers become null before serializing, so orjson never has to
    refuse the payload and the result is always parseable — which matters
    when it is being written into a page someone else's browser will read.
    """
    return orjson.dumps(finite(payload), default=str).decode("utf-8")


def _trimmed(payload: Any) -> Any:
    """finite(), plus floats cut to six significant digits: 0.123456789
    reads the same to a model as 0.123457 and costs fewer tokens."""
    if isinstance(payload, float):
        if payload != payload or payload in (float("inf"), float("-inf")):
            return None
        return float(f"{payload:.6g}")
    if isinstance(payload, dict):
        return {k: _trimmed(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_trimmed(v) for v in payload]
    return payload


def dumps_prompt(payload: Any) -> str:
    """JSON for an LLM prompt: no indentation, no float noise, valid always.

    Indentation and 15-digit floats are tokens the model reads and we pay
    for on every per-ticker call, with nothing gained; and the NaN fallback
    of `dumps_pretty` put bare `NaN` tokens (not JSON) into the prompt.
    """
    return orjson.dumps(_trimmed(payload), default=str).decode("utf-8")


def loads(raw: str | bytes) -> Any:
    """orjson.loads, which takes str and bytes alike."""
    return orjson.loads(raw)
