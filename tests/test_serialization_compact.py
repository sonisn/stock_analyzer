"""Embedding JSON in a page someone else's browser will parse.

stdlib `json` writes NaN and Infinity as bare tokens, which are not valid
JSON. They happen to parse as JavaScript, so a page carrying them works
by luck rather than by contract — and yfinance emits NaN for missing data
often enough that it is not a hypothetical.
"""

from __future__ import annotations

import json

from stock_analyzer.serialization import dumps_compact, dumps_pretty, dumps_prompt, finite, loads

NAN, INF = float("nan"), float("inf")


def test_missing_numbers_become_null_not_bare_tokens():
    payload = {"a": NAN, "b": [1.0, INF], "c": {"d": -INF}, "e": 2.5}
    out = dumps_compact(payload)
    assert out == '{"a":null,"b":[1.0,null],"c":{"d":null},"e":2.5}'
    # What stdlib would have written into the page instead:
    assert "NaN" in json.dumps(payload)
    # And the result survives a strict parser, which NaN would not.
    assert json.loads(out) == {"a": None, "b": [1.0, None], "c": {"d": None}, "e": 2.5}


def test_compact_means_no_whitespace():
    """The page carries this inline; indentation is bytes for nothing."""
    payload = {"holdings": [{"ticker": "NVDA", "value": 89233}]}
    assert " " not in dumps_compact(payload)
    assert len(dumps_compact(payload)) < len(dumps_pretty(payload))


def test_finite_leaves_good_values_alone():
    assert finite({"a": 1, "b": "x", "c": None, "d": [2.5, True]}) == {
        "a": 1,
        "b": "x",
        "c": None,
        "d": [2.5, True],
    }
    assert finite(0.0) == 0.0, "zero is a number, not a missing one"


def test_unserializable_values_fall_back_to_str_rather_than_raising():
    from datetime import date

    out = loads(dumps_compact({"d": date(2026, 9, 21), "s": {1, 2}}))
    assert out["d"] == "2026-09-21"
    assert isinstance(out["s"], str)


def test_loads_takes_str_and_bytes():
    assert loads('{"a":1}') == {"a": 1}
    assert loads(b'{"a":1}') == {"a": 1}


def test_prompt_json_has_no_indentation_or_float_noise():
    """Whitespace and 15-digit floats are tokens paid for on every call."""
    payload = {"pe": 31.456789012345, "margin": 0.123456789, "cap": 4.2e12, "n": [1, 2]}
    assert (
        dumps_prompt(payload) == '{"pe":31.4568,"margin":0.123457,"cap":4200000000000.0,"n":[1,2]}'
    )


def test_prompt_json_is_valid_even_with_missing_numbers():
    """The pretty fallback wrote bare NaN into the prompt; this writes null."""
    out = dumps_prompt({"a": NAN, "b": (INF, 1.5), "c": "text"})
    assert json.loads(out) == {"a": None, "b": [None, 1.5], "c": "text"}
