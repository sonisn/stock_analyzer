"""_flatten_score_breakdown: fixes a pre-existing bug where
similar_past_setups() was called with the raw nested score_breakdown dict
instead of the flat {"total.<group>": v, "<group>.<leaf>": v} shape that
score_validation.py::_flatten_components produces for stored past
candidates — so the nearest-neighbor distance compared a flat dict
against a nested one and silently returned [] every run."""

from __future__ import annotations

from stock_analyzer.cli.discover_steps.helpers import _flatten_score_breakdown


def test_flattens_components_and_breakdown_into_matching_key_shape():
    components = {"fundamentals": 30.0, "trend": 20.0, "conviction": 5.0}
    breakdown = {
        "fundamentals": {"revenue_growth": 17.0, "fcf_yield": 5.0},
        "trend": {"rs_6mo": 10.0},
    }
    flat = _flatten_score_breakdown(components, breakdown)
    assert flat == {
        "total.fundamentals": 30.0,
        "total.trend": 20.0,
        "total.conviction": 5.0,
        "fundamentals.revenue_growth": 17.0,
        "fundamentals.fcf_yield": 5.0,
        "trend.rs_6mo": 10.0,
    }


def test_non_numeric_and_non_dict_leaves_skipped():
    components = {"fundamentals": 30.0}
    breakdown = {
        "fundamentals": {"revenue_growth": 17.0},
        "theme": {"matched_theme": "AI capex", "strength": 8},
        "junk": "not a dict",
    }
    flat = _flatten_score_breakdown(components, breakdown)
    assert flat == {
        "total.fundamentals": 30.0,
        "fundamentals.revenue_growth": 17.0,
        "theme.strength": 8.0,
    }


def test_none_inputs_return_empty_dict():
    assert _flatten_score_breakdown(None, None) == {}
