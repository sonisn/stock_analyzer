"""Filed income drops (data/income_drop) — synthetic XBRL facts, no network."""

from __future__ import annotations

from stock_analyzer.data import income_drop as idr


def _f(start, end, val, accn, filed):
    return {"start": start, "end": end, "val": val, "accn": accn, "filed": filed}


FACTS = [
    # Year-ago quarter, first filed and then restated in a later 10-Q.
    _f("2025-04-01", "2025-06-30", 1_000e6, "old", "2025-08-01"),
    _f("2025-04-01", "2025-06-30", 1_100e6, "new", "2026-08-01"),
    # The new 10-Q: its quarter, its half-year, and the comparison column.
    _f("2026-04-01", "2026-06-30", 550e6, "new", "2026-08-01"),
    _f("2026-01-01", "2026-06-30", 1_500e6, "new", "2026-08-01"),
    # A quarter from another filing that ends later must not be "this" period.
    _f("2026-07-01", "2026-09-30", 900e6, "later", "2026-11-01"),
]


def test_drop_uses_the_filings_own_quarter_and_the_restated_year_ago():
    assert idr.drop_in(FACTS, "new") == (550e6, 1_100e6)
    assert idr.drop_in(FACTS, "missing") is None


def test_a_10k_compares_twelve_months():
    facts = [
        _f("2024-08-01", "2025-07-31", 1_243e6, "k25", "2025-09-01"),
        _f("2025-08-01", "2026-07-31", 695e6, "k26", "2026-09-01"),
    ]
    assert idr.drop_in(facts, "k26") == (695e6, 1_243e6)


def test_describe_flags_big_drops_and_losses_only():
    assert idr.describe("operating income", 695e6, 1_243e6).startswith("operating income -44%")
    assert "swung to a $-20M loss" in idr.describe("net income", -20e6, 300e6)
    assert idr.describe("net income", 800e6, 1_000e6) is None  # -20% is not enough
    assert idr.describe("net income", -50e6, -10e6) is None  # no base to fall from
    assert idr.describe("net income", 1e6, 3e6) is None  # too small to measure


def test_foreign_filers_and_missing_cik_are_not_checked(monkeypatch):
    monkeypatch.setattr(idr._HTTP, "get_json", lambda url: 1 / 0)
    assert idr.income_drops({"cik": 1, "accession": "a", "form": "20-F"}) == []
    assert idr.income_drops({"accession": "a", "form": "10-Q"}) == []
    # An unanswered concept is no check, not a failure.
    assert idr.income_drops({"cik": 1, "accession": "a", "form": "10-Q"}) == []


def test_income_drops_reads_both_concepts(monkeypatch):
    monkeypatch.setattr(idr._HTTP, "get_json", lambda url: {"units": {"USD": FACTS}})
    got = idr.income_drops({"cik": 1, "accession": "new", "form": "10-Q"})
    assert len(got) == 2 and got[0].startswith("filed operating income -50%")


def test_risk_factor_change_measures_and_describes():
    from stock_analyzer.data.text_change import compare, describe

    word = [chr(97 + i) * 3 for i in range(20)]  # "aaa", "bbb", ...: digits aren't words
    base = [f"Our business faces the {w} risk from competition and regulation today." for w in word]
    same = compare(" ".join(base), " ".join(base))
    assert same["kept"] == 1.0 and same["cosine"] == 1.0
    new = [f"A new {w} cyber threat could disrupt our operations badly." for w in word[:12]]
    got = compare(" ".join(base[:8] + new), " ".join(base))
    assert got["kept"] == 0.4
    assert "heavily rewritten" in describe({**got, "prior_filed_on": "2025-02-20"})
    assert "about as usual" in describe({"kept": 0.72, "prior_filed_on": "2025-02-20"})
    assert compare("too short.", " ".join(base)) is None
