"""SEC filing reader: section cutting, quote checks, flags, the daily cap,
storage — no network, no model."""

from __future__ import annotations

import json
from datetime import date

import pytest
from sqlmodel import select

from stock_analyzer import openrouter
from stock_analyzer.agents import filing_reader as fr
from stock_analyzer.cli import filings
from stock_analyzer.data.sec_edgar import filing_sections, filing_text
from stock_analyzer.db.session import get_session
from stock_analyzer.db.tables import FilingFacts
from stock_analyzer.openrouter import OpenRouter, parse_json_object, spent_today
from stock_analyzer.usage import BudgetExceededError

MDA = "Revenue grew 20% on strong AI demand. " * 20
FILING_HTML = f"""<html><head><style>x{{}}</style></head><body>
<ix:header>hidden xbrl junk</ix:header>
<p>Item 2. Management's Discussion and Analysis ... 12</p>
<p>ITEM 2 &#8212; MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS</p><p>{MDA}</p>
<table><tr><td>Backlog</td><td>$</td><td>1,234</td></tr></table>
<p>ITEM 3 &#8212; QUANTITATIVE AND QUALITATIVE DISCLOSURES</p><p>rates</p>
<p>ITEM 1A. RIS K FACTORS</p><p>{"Our largest customer may leave. " * 20}</p>
<p>Item 2. Unregistered Sales</p></body></html>"""


def test_filing_text_and_sections_skip_toc_and_keep_table_rows():
    text = filing_text(FILING_HTML)
    assert "hidden xbrl" not in text and "x{}" not in text
    assert "Backlog | $ | 1,234" in text
    s = filing_sections(text, "10-Q", max_chars={"mda": 100_000, "risks": 100_000})
    assert "Revenue grew" in s["mda"][:40]  # the real header, not the TOC
    assert "Backlog" in s["mda"] and "rates" not in s["mda"]
    assert "Our largest customer" in s["risks"][:40]
    assert "Unregistered" not in s["risks"]
    capped = filing_sections(text, "10-Q", max_chars={"mda": 50, "risks": 50})
    assert len(capped["mda"]) == 50


def test_quote_check_tolerates_formatting_but_not_invention():
    src = fr.normalise("Total backlog was $1,234 million, up 12% from “last year”.\nNext line")
    assert fr.quote_found('total backlog was $1,234 million, up 12% from "last year".', src)
    assert fr.quote_found("", src)
    assert not fr.quote_found("Backlog doubled to a record $9 billion on hyperscaler orders", src)
    reply = {
        "guidance": {"quote": "up 12% from last year"},
        "key_risks": [{"quote": "a sentence the filing never said at all"}, {"quote": ""}],
    }
    assert fr.check_quotes(reply, src) == (2, 1, ["a sentence the filing never said at all"])


def test_flag_reasons():
    assert fr.flag_reasons(None, 0, 0) == ["reader reply was not usable JSON"]
    clean = {"guidance": {"direction": "raised"}, "liquidity": {"concern": False}, "caveats": []}
    assert fr.flag_reasons(clean, 10, 10) == []
    bad = {
        "guidance": {"direction": "withdrawn"},
        "liquidity": {"concern": True},
        "caveats": [
            {"issue": "material weakness", "category": "material_weakness", "severity": "high"},
            {"issue": "tariffs", "category": "other", "severity": "high"},
            {"severity": "medium"},
        ],
    }
    reasons = fr.flag_reasons(bad, 10, 7)
    assert reasons == [
        "high-severity caveat: material weakness",
        "guidance withdrawn",
        "liquidity concern",
        "only 7/10 quotes found in the filing",
    ]


def test_parse_json_object_handles_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here it is: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("nope") is None


class _Http:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def post_json(self, url, json):  # noqa: A002 — mirrors HttpClient
        self.bodies.append(json)
        text, cost = self.replies.pop(0)
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": cost},
        }


def _client(tmp_path, replies, cap=2.0) -> OpenRouter:
    c = OpenRouter("k", str(tmp_path / "t.db"), daily_cap_usd=cap)
    c._http = _Http(replies)  # type: ignore[assignment]
    return c


def test_daily_cap_counts_billed_spend_across_clients(tmp_path):
    a = _client(tmp_path, [("{}", 0.009)] * 3, cap=0.02)
    a.complete("S", "z-ai/glm-5.3-flash", "sys", "hi", max_tokens=100)
    a.complete("S", "z-ai/glm-5.3-flash", "sys", "hi", max_tokens=100)
    assert spent_today(a.db_path) == pytest.approx(0.018)
    # A second process on the same day sees the same ledger.
    b = _client(tmp_path, [("{}", 0.009)], cap=0.02)
    with pytest.raises(BudgetExceededError):
        b.complete("S", "z-ai/glm-5.3-flash", "sys", "x" * 10_000, max_tokens=3000)
    body = a._http.bodies[0]  # type: ignore[attr-defined]
    assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0.0


def test_unknown_model_is_priced_high_for_the_cap(tmp_path):
    assert openrouter.worst_case_cost("x/unknown", 350_000, 1000) == pytest.approx(0.3 + 0.015)


FACTS_OK = {
    "guidance": {"direction": "raised", "quote": "Revenue grew 20% on strong AI demand."},
    "liquidity": {"concern": False, "quote": ""},
    "caveats": [],
    "tone": "positive",
}
FACTS_FLAGGED = {
    **FACTS_OK,
    "caveats": [
        {
            "issue": "customer loss",
            "category": "customer_loss",
            "severity": "high",
            "quote": "Our largest customer may leave.",
        }
    ],
}
FILING = {
    "ticker": "ABC",
    "form": "10-Q",
    "filed_on": "2026-08-01",
    "period_end": "2026-06-30",
    "accession": "0001-26-000001",
    "url": "u",
}
SECTIONS = {"mda": MDA, "risks": "Our largest customer may leave. " * 5}


def test_read_sends_fp8_routing_and_retries_an_empty_answer(tmp_path):
    c = _client(tmp_path, [("", 0.01), (json.dumps(FACTS_OK), 0.002)])
    read = fr.read_filing(c, FILING, SECTIONS, model="z-ai/glm-5.3")
    assert read.facts == FACTS_OK and read.cost_usd == pytest.approx(0.012)
    first, second = c._http.bodies  # type: ignore[attr-defined]
    assert first["provider"]["quantizations"] == ["fp8", "bf16", "fp16"]
    assert first["reasoning"] == {"effort": "low"} and second["reasoning"] == {"enabled": False}
    assert not read.flagged and (read.quotes_checked, read.quotes_found) == (1, 1)


def test_retry_keeps_thinking_on_where_the_host_requires_it(tmp_path):
    from stock_analyzer.http_client import ClientError

    c = _client(tmp_path, [("", 0.001), (json.dumps(FACTS_OK), 0.002)])
    http = c._http  # type: ignore[attr-defined]
    real = http.post_json
    calls = []

    def post_json(url, json):  # noqa: A002
        calls.append(json)
        if json.get("reasoning") == {"enabled": False}:
            raise ClientError("400: Reasoning is mandatory for this endpoint", status=400)
        return real(url, json)

    http.post_json = post_json
    read = fr.read_filing(c, FILING, SECTIONS, model="z-ai/glm-5.3-flash")
    assert read.facts == FACTS_OK
    assert [b["max_tokens"] for b in calls] == [8000, 8000, 16000]
    assert calls[-1]["reasoning"] == {"effort": "low"}


def test_tier_b_escalates_a_flagged_bulk_read_to_the_reader(tmp_path):
    c = _client(tmp_path, [(json.dumps(FACTS_OK), 0.001)])
    item = (FILING, SECTIONS)
    clean = filings.read_tiered(c, item, tier="B", reader_model="big", bulk_model="flash")
    assert clean.reader_model == "flash" and clean.escalated_from is None

    c = _client(tmp_path, [(json.dumps(FACTS_FLAGGED), 0.001), (json.dumps(FACTS_OK), 0.01)])
    better = filings.read_tiered(c, item, tier="B", reader_model="big", bulk_model="flash")
    assert better.reader_model == "big" and better.escalated_from == "flash"
    assert better.cost_usd == pytest.approx(0.011)
    models = [b["model"] for b in c._http.bodies]  # type: ignore[attr-defined]
    assert models == ["flash", "big"]

    c = _client(tmp_path, [(json.dumps(FACTS_FLAGGED), 0.01)])
    a = filings.read_tiered(c, item, tier="A", reader_model="big", bulk_model="flash")
    assert a.reader_model == "big" and a.flagged  # tier A is read once, on the reader


def test_tier_b_sends_an_empty_or_failed_bulk_read_straight_to_the_reader(tmp_path):
    item = (FILING, SECTIONS)
    c = _client(tmp_path, [("", 0.001), (json.dumps(FACTS_OK), 0.01)])
    r = filings.read_tiered(c, item, tier="B", reader_model="big", bulk_model="flash")
    assert [b["model"] for b in c._http.bodies] == ["flash", "big"]  # type: ignore[attr-defined]
    assert r.facts == FACTS_OK and r.escalated_from == "flash"
    assert r.cost_usd == pytest.approx(0.011)

    c = _client(tmp_path, [(json.dumps(FACTS_OK), 0.01)])
    http = c._http  # type: ignore[attr-defined]
    real = http.post_json

    def post_json(url, json):  # noqa: A002
        if json["model"] == "flash":
            raise RuntimeError("host down")
        return real(url, json)

    http.post_json = post_json
    r = filings.read_tiered(c, item, tier="B", reader_model="big", bulk_model="flash")
    assert r.reader_model == "big" and r.escalated_from == "flash"


def test_a_filed_income_drop_escalates_a_clean_bulk_read(tmp_path):
    drop = "filed operating income -44% vs a year earlier ($695M from $1,243M)"
    item = ({**FILING, "income_drops": [drop]}, SECTIONS)
    c = _client(tmp_path, [(json.dumps(FACTS_OK), 0.001), (json.dumps(FACTS_OK), 0.01)])
    r = filings.read_tiered(c, item, tier="B", reader_model="big", bulk_model="flash")
    assert [b["model"] for b in c._http.bodies] == ["flash", "big"]  # type: ignore[attr-defined]
    assert r.escalated_from == "flash" and r.flag_reasons == [drop]
    # The reader is told what the figures say.
    prompt = c._http.bodies[1]["messages"][-1]["content"]  # type: ignore[attr-defined]
    assert "Income as filed (XBRL): operating income -44%" in prompt


def test_recheck_drops_rereads_only_stored_bulk_reads_that_dropped(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    for t in ("AAA", "BBB"):
        f = {**FILING, "ticker": t, "accession": f"{t}-1"}
        read = fr.FilingRead(filing=f, reader_model="flash", facts=FACTS_OK)
        filings.store(db, read, tier="B", today=date(2026, 9, 26))
    monkeypatch.setattr(
        filings, "_prepare", lambda t: ({**FILING, "ticker": t, "accession": f"{t}-1"}, SECTIONS)
    )
    monkeypatch.setattr(
        filings, "income_drops", lambda f: ["filed net income -73%"] if f["ticker"] == "BBB" else []
    )
    c = _client(tmp_path, [(json.dumps(FACTS_OK), 0.01)])
    monkeypatch.setattr(filings, "client_from_settings", lambda s: c)
    monkeypatch.setattr(filings, "log_usage_summary", lambda: None)
    settings = filings.Settings(
        discover_db_path=db, openrouter_reader_model="big", openrouter_bulk_model="flash"
    )
    tiers = {"AAA": "B", "BBB": "B"}
    assert filings.run(settings, tiers, today=date(2026, 9, 27), recheck_drops=True) == 0
    assert [b["model"] for b in c._http.bodies] == ["big"]  # type: ignore[attr-defined]
    with get_session(db) as s:
        row = s.exec(select(FilingFacts).where(FilingFacts.ticker == "BBB")).one()
        got = (row.reader_model, row.escalated_from, row.flag_reasons)
    assert got == (
        "big",
        "flash",
        "filed net income -73%",
    )


def test_store_keeps_two_filings_for_tier_a_and_one_for_tier_b(tmp_path):
    db = str(tmp_path / "t.db")
    for tier, ticker in (("A", "AAA"), ("B", "BBB")):
        for n, filed in enumerate(["2026-02-01", "2026-05-01", "2026-08-01"]):
            f = {**FILING, "ticker": ticker, "accession": f"{ticker}-{n}", "filed_on": filed}
            read = fr.FilingRead(filing=f, reader_model="r", facts=FACTS_OK)
            filings.store(db, read, tier=tier, today=date(2026, 9, 27))
    with get_session(db) as s:
        kept = sorted(r.accession for r in s.exec(select(FilingFacts)).all())
    assert kept == ["AAA-1", "AAA-2", "BBB-2"]


def test_run_promotes_tier_a_bulk_reads_and_skips_what_is_current(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    filing = {**FILING, "accession": "acc-1"}
    for ticker, model in (("AAA", "flash"), ("BBB", "flash")):
        f = {**filing, "ticker": ticker, "accession": f"{ticker}-1"}
        filings.store(
            db,
            fr.FilingRead(filing=f, reader_model=model, facts=FACTS_OK),
            tier="B",
            today=date(2026, 9, 26),
        )
    monkeypatch.setattr(
        filings, "_prepare", lambda t: ({**filing, "ticker": t, "accession": f"{t}-1"}, SECTIONS)
    )
    seen = []
    monkeypatch.setattr(
        filings,
        "read_tiered",
        lambda client, item, *, tier, reader_model, bulk_model: (
            seen.append((item[0]["ticker"], tier))
            or fr.FilingRead(filing=item[0], reader_model=reader_model, facts=FACTS_OK)
        ),
    )
    monkeypatch.setattr(filings, "log_usage_summary", lambda: None)
    from stock_analyzer.reporting import filing_alert

    monkeypatch.setattr(filing_alert, "earnings_releases", lambda *a, **k: [])
    settings = filings.Settings(
        discover_db_path=db,
        openrouter_api_key="k",
        openrouter_reader_model="big",
        openrouter_bulk_model="flash",
    )
    tiers = {"AAA": "A", "BBB": "B", "CCC": "B"}
    assert filings.run(settings, tiers, today=date(2026, 9, 27)) == 0
    # AAA only had a bulk read and is now tier A; BBB is current; CCC is new.
    assert sorted(seen) == [("AAA", "A"), ("CCC", "B")]


def test_compare_fields():
    other = {**FACTS_OK, "tone": "neutral"}
    agree = filings.compare(FACTS_OK, other)
    assert agree["tone"] is False and agree["guidance.direction"] is True
    assert filings.compare(FACTS_OK, FACTS_FLAGGED)["high_caveat"] is False


def test_evidence_pack_is_compact_and_shows_what_changed(tmp_path):
    from stock_analyzer.data.filing_evidence import evidence_packs, prefer_pack

    db = str(tmp_path / "t.db")
    older = {**FACTS_OK, "demand": {"direction": "steady", "detail": "flat"}, "tone": "neutral"}
    newer = {
        **FACTS_FLAGGED,
        "demand": {"direction": "accelerating", "detail": "AWS +37%", "quote": "x"},
        "backlog": {"value_usd_millions": 21200, "change": "growing", "detail": "RPO"},
        "key_risks": [{"risk": "channel partners", "quote": "q"}],
    }
    for n, (filed, facts) in enumerate((("2026-05-01", older), ("2026-08-01", newer))):
        f = {**FILING, "accession": f"acc-{n}", "filed_on": filed, "period_end": filed}
        read = fr.FilingRead(
            filing=f, reader_model="r", facts=facts, quotes_checked=4, quotes_found=4
        )
        filings.store(db, read, tier="A", today=date(2026, 9, 27))
    packs = evidence_packs(db, ["abc", "NONE"])
    p = packs["ABC"]
    assert list(packs) == ["ABC"] and p["filed_on"] == "2026-08-01"
    assert p["demand"] == "accelerating: AWS +37%"
    assert p["backlog"] == "growing: RPO ($21,200M)"
    assert p["key_risks"] == ["channel partners"] and p["quotes_verified"] == "4/4"
    assert p["events"][0]["category"] == "customer_loss" and "quote" in p["events"][0]
    assert p["vs_prior_filing"]["demand"] == "steady -> accelerating"
    assert "margins" not in p  # nothing to say is left out, not sent as null
    assert prefer_pack(p, None) and prefer_pack(p, "2026-08-01")
    assert not prefer_pack(p, "2026-09-01")  # the run fetched a newer 10-Q
    assert not prefer_pack(None, None)
    assert evidence_packs(str(tmp_path / "empty.db"), ["ABC"]) == {}


def test_tier_a_includes_the_shortlist_of_rebalance_runs(tmp_path):
    from stock_analyzer.db.tables import Run, Scorecard

    db = str(tmp_path / "t.db")
    with get_session(db) as s:
        for i, kind in enumerate(("discover", "rebalance", "rebalance", "rebalance"), start=1):
            s.add(
                Run(id=i, run_at=f"2026-09-2{i}", kind=kind, universe_size=1, survivors=1, picks=0)
            )
            s.add(Scorecard(run_id=i, ticker=f"T{i}"))
    assert sorted(filings.analyzed_recently(db)) == ["T2", "T3", "T4"]


def test_filing_features_count_every_category_and_red_flags_only_high(tmp_path):
    from stock_analyzer.data.filing_evidence import filing_features, red_flags

    db = str(tmp_path / "t.db")
    caveats = [
        {"issue": "mw", "category": "material_weakness", "severity": "high", "quote": "q"},
        {"issue": "rs", "category": "restatement", "severity": "medium", "quote": "q"},
        {"issue": "im", "category": "impairment", "severity": "medium", "quote": "q"},
        {"issue": "im2", "category": "impairment", "severity": "high", "quote": "q"},
    ]
    rows = {
        "AAA": ({**FACTS_OK, "caveats": caveats}, "2026-08-01", ["filed net income -40%"]),
        "BBB": (FACTS_OK, "2026-08-01", []),
        "OLD": ({**FACTS_OK, "caveats": caveats}, "2025-01-01", []),  # superseded
    }
    for t, (facts, filed, reasons) in rows.items():
        f = {**FILING, "ticker": t, "accession": f"{t}-1", "filed_on": filed}
        read = fr.FilingRead(filing=f, reader_model="r", facts=facts, flag_reasons=reasons)
        filings.store(db, read, tier="B", today=date(2026, 9, 27))
    today = date(2026, 9, 27)
    feats = filing_features(db, ["AAA", "BBB", "OLD", "NONE"], today=today)
    assert set(feats) == {"AAA", "BBB"}
    assert feats["AAA"] == {
        "filing_material_weakness": 1,
        "filing_material_weakness_high": 1,
        "filing_restatement": 1,
        "filing_impairment": 2,
        "filing_impairment_high": 1,
        "filing_income_drop": 1.0,
    }
    assert feats["BBB"] == {"filing_income_drop": 0.0}
    # Medium restatements and impairments are recorded but not scored.
    assert red_flags(db, ["AAA", "BBB", "OLD"], today=today) == {"AAA": ["material_weakness"]}


def test_candidate_snapshot_keeps_the_filing_facts():
    import json as _json
    from types import SimpleNamespace

    from stock_analyzer.db.repository import insert_candidate_snapshot

    added: list = []
    session = SimpleNamespace(add=added.append)
    insert_candidate_snapshot(
        session,  # type: ignore[arg-type]
        1,
        "AAA",
        {"market_cap": 5e9},
        None,
        {"filing_going_concern_high": 1},
    )
    assert _json.loads(added[0].data) == {"market_cap": 5e9, "filing_going_concern_high": 1}


def test_helper_agent_on_openrouter_falls_back_but_never_past_the_cap(tmp_path):
    from types import SimpleNamespace

    from stock_analyzer.openrouter import HELPER_EXTRA, OpenRouterAgent

    used = []

    def fallback():
        used.append(1)
        return SimpleNamespace(run=lambda p: SimpleNamespace(content="from claude"))

    c = _client(tmp_path, [('{"AVGO": [2]}', 0.001)])
    agent = OpenRouterAgent("Rerank", "z-ai/glm-5.3", "sys", client=c, json_mode=True)
    assert agent.run("news").content == '{"AVGO": [2]}'
    body = c._http.bodies[0]  # type: ignore[attr-defined]
    routing = {k: v for k, v in body["provider"].items() if k != "only"}
    assert routing == HELPER_EXTRA["provider"] and body["response_format"]

    # An empty answer or an outage goes to the fallback...
    c = _client(tmp_path, [("", 0.001)])
    agent = OpenRouterAgent("Insider", "z-ai/glm-5.3", "sys", client=c, fallback=fallback)
    assert agent.run("x").content == "from claude" and used == [1]
    no_key = OpenRouterAgent("Insider", "z-ai/glm-5.3", "sys", client=None, fallback=fallback)
    assert no_key.run("x").content == "from claude"

    # ...but a call refused by the daily cap is never re-spent elsewhere.
    c = _client(tmp_path, [], cap=0.0)
    agent = OpenRouterAgent("Insider", "z-ai/glm-5.3", "sys", client=c, fallback=fallback)
    with pytest.raises(BudgetExceededError):
        agent.run("x" * 1000)
    assert used == [1, 1]


def test_reader_calls_go_only_to_approved_hosts(tmp_path):
    c = _client(tmp_path, [(json.dumps(FACTS_OK), 0.005), (json.dumps(FACTS_OK), 0.001)])
    fr.read_filing(c, FILING, SECTIONS, model="z-ai/glm-5.3")
    fr.read_filing(c, FILING, SECTIONS, model="z-ai/glm-5.3-flash")
    big, flash = c._http.bodies  # type: ignore[attr-defined]
    assert big["provider"]["only"] == ["io-net", "morph", "novita", "baidu"]
    assert big["provider"]["quantizations"] == ["fp8", "bf16", "fp16"]
    assert "akashml" not in flash["provider"]["only"]
