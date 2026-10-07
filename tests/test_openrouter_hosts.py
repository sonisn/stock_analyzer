"""OpenRouter host guardrails: allowlist, known-answer check, per-host
quality, the insider grounding check and the Claude spot-check — no network."""

from __future__ import annotations

import json
from datetime import date

import pytest

from stock_analyzer import openrouter_hosts as oh
from stock_analyzer.agents import filing_reader as fr
from stock_analyzer.agents.insider import ungrounded_tickers
from stock_analyzer.cli import filings
from stock_analyzer.db.session import get_session
from stock_analyzer.db.tables import OpenRouterHostCheck
from stock_analyzer.openrouter import APPROVED_HOSTS, NoApprovedHostError, OpenRouterAgent

from .test_filing_reader import FACTS_OK, FILING, SECTIONS, HTTPError, _client

TODAY = date(2026, 9, 28)
CANARY_OK = {
    "guidance": {"direction": "raised", "quote": "We are raising our full-year revenue outlook"},
    "demand": {"direction": "accelerating", "quote": ""},
    "caveats": [
        {
            "issue": "material weakness",
            "category": "material_weakness",
            "severity": "high",
            "quote": "identified a material weakness in internal control over financial reporting",
        }
    ],
}
# What a host that masked "Toledo, Ohio" as "[ADDRESS]" would quote back.
CANARY_MASKED = {
    **CANARY_OK,
    "demand": {
        "direction": "accelerating",
        "quote": "driven by utility demand for grid transformers built in [ADDRESS].",
    },
}


def test_calls_are_limited_to_the_approved_hosts_left(tmp_path):
    c = _client(tmp_path, [(json.dumps(FACTS_OK), 0.01)])
    c.excluded = {"z-ai/glm-5.3": {"morph"}}
    fr.read_filing(c, FILING, SECTIONS, model="z-ai/glm-5.3")
    only = c._http.bodies[0]["provider"]["only"]  # type: ignore[attr-defined]
    assert only == [h for h in APPROVED_HOSTS["z-ai/glm-5.3"] if h != "morph"]

    c.excluded = {"z-ai/glm-5.3": set(APPROVED_HOSTS["z-ai/glm-5.3"])}
    with pytest.raises(NoApprovedHostError):
        fr.read_filing(c, FILING, SECTIONS, model="z-ai/glm-5.3")


def test_known_answer_check_fails_masked_input_and_excludes_the_host(tmp_path):
    db = str(tmp_path / "t.db")
    hosts = APPROVED_HOSTS["z-ai/glm-5.3"]
    replies = [(json.dumps(CANARY_OK), 0.003)] * len(hosts)
    replies[1] = (json.dumps(CANARY_MASKED), 0.003)
    c = _client(tmp_path, replies)
    c.db_path = db
    checks = oh.run_canaries(c, db, ["z-ai/glm-5.3"], today=TODAY)
    assert [ch.passed for ch in checks] == [i != 1 for i in range(len(hosts))]
    assert checks[1].detail == "quotes 2/3 match the text sent; redaction placeholder in the reply"
    # Each check was pinned to its host, with no fallback to another.
    first = c._http.bodies[0]["provider"]  # type: ignore[attr-defined]
    assert first["only"] == [hosts[0]] and first["allow_fallbacks"] is False
    assert c.excluded == {"z-ai/glm-5.3": {hosts[1]}}
    # Stored, so the next process skips it too — until a later check passes.
    assert oh.excluded_hosts(db, today=TODAY) == {"z-ai/glm-5.3": {hosts[1]}}
    with get_session(db) as s:
        s.merge(
            OpenRouterHostCheck(day="2026-09-29", model="z-ai/glm-5.3", host=hosts[1], passed=True)
        )
    assert oh.excluded_hosts(db, today=date(2026, 9, 29)) == {}


def test_a_host_with_no_endpoint_is_skipped_but_not_a_doctor_failure(tmp_path):
    """io-net dropped glm-5.3 and two flash hosts went fp4-only (2026-10-03):
    OpenRouter's 404 kept the host out, and the doctor reported it as a
    failed known-answer check."""
    from stock_analyzer.cli.ops import openrouter_host_problems

    db = str(tmp_path / "t.db")
    hosts = APPROVED_HOSTS["z-ai/glm-5.3"]
    c = _client(tmp_path, [(json.dumps(CANARY_OK), 0.003)] * (len(hosts) - 1))
    c.db_path = db
    post = c._http.post_json  # type: ignore[attr-defined]

    def post_json(url, json):  # noqa: A002
        if json["provider"]["only"] == [hosts[0]]:
            raise HTTPError(404, "No endpoints found for the request with quantization")
        return post(url, json)

    c._http.post_json = post_json  # type: ignore[attr-defined]
    checks = oh.run_canaries(c, db, ["z-ai/glm-5.3"], today=TODAY)
    assert checks[0].detail.startswith("unavailable:") and not checks[0].passed
    assert all(ch.passed for ch in checks[1:])
    assert oh.excluded_hosts(db, today=TODAY) == {"z-ai/glm-5.3": {hosts[0]}}
    problems, summary = openrouter_host_problems(db, today=TODAY)
    assert problems == []
    assert summary.endswith(f"no endpoint at the last check: {hosts[0]} (z-ai/glm-5.3)")


def test_canary_problems_name_each_miss():
    read = fr.FilingRead(
        filing=oh.CANARY_FILING,
        reader_model="m",
        facts={**CANARY_OK, "guidance": {"direction": "maintained"}, "caveats": []},
        quotes_checked=2,
        quotes_found=2,
        provider="Baidu",
    )
    got = oh.canary_problems(read, "io-net")
    assert got == [
        "quotes 0/0 match the text sent",
        "guidance 'maintained', not raised",
        "missed the material weakness",
        "served by Baidu, not io-net",
    ]


def test_a_host_whose_quotes_slip_is_excluded(tmp_path):
    db = str(tmp_path / "t.db")
    for n in range(10):
        read = fr.FilingRead(
            filing={**FILING, "ticker": f"T{n}", "accession": f"a{n}"},
            reader_model="z-ai/glm-5.3",
            facts=FACTS_OK,
            quotes_checked=10,
            quotes_found=9,  # 90%
            provider="Novita",
        )
        filings.store(db, read, tier="A", today=TODAY)
    q = oh.host_quality(db, today=TODAY)
    assert q[0]["host"] == "novita" and q[0]["problem"] == "quote match 90.0% < 97%"
    assert oh.excluded_hosts(db, today=TODAY) == {"z-ai/glm-5.3": {"novita"}}
    from stock_analyzer.cli.ops import openrouter_host_problems

    # Already skipped by every run: a note, not a doctor failure.
    problems, summary = openrouter_host_problems(db, today=TODAY)
    assert problems == []
    assert "skipped until they pass: novita (z-ai/glm-5.3): quote match 90.0% < 97%" in summary


def test_the_doctor_fails_only_when_a_model_has_no_host_left(tmp_path):
    """2026-10-05: three hosts failed the known-answer check and were
    already skipped, yet the doctor emailed FAILED."""
    from stock_analyzer.cli.ops import openrouter_host_problems

    db = str(tmp_path / "t.db")
    hosts = APPROVED_HOSTS["z-ai/glm-5.3"]
    with get_session(db) as session:
        for h in hosts[:-1]:
            session.merge(
                OpenRouterHostCheck(
                    day=TODAY.isoformat(), model="z-ai/glm-5.3", host=h, passed=False, detail="x"
                )
            )
    problems, summary = openrouter_host_problems(db, today=TODAY)
    assert problems == []
    assert f"{hosts[0]} (z-ai/glm-5.3): failed its last known-answer check" in summary
    with get_session(db) as session:
        session.merge(
            OpenRouterHostCheck(
                day=TODAY.isoformat(),
                model="z-ai/glm-5.3",
                host=hosts[-1],
                passed=False,
                detail="x",
            )
        )
    problems, _ = openrouter_host_problems(db, today=TODAY)
    assert problems == [
        f"z-ai/glm-5.3: no usable host left — all {len(hosts)} approved hosts are skipped"
    ]


def test_a_placeholder_in_any_read_flags_it_for_the_better_reader():
    masked = {**FACTS_OK, "summary": "Revenue from the plant in [ADDRESS] rose."}
    assert "redaction placeholder" in fr.flag_reasons(masked, 1, 1)[0]
    assert fr.flag_reasons(FACTS_OK, 1, 1) == []


def test_insider_report_naming_an_unsourced_ticker_goes_to_the_fallback(tmp_path, monkeypatch):
    from stock_analyzer.data import sec_edgar

    monkeypatch.setattr(
        sec_edgar, "load_ticker_titles", lambda: {"DKS": "DICK'S SPORTING GOODS, INC."}
    )
    source = "Dick’s Sporting Goods CFO buys shares; PANW insiders sell."
    good = "- DKS: CFO (CFO) bought.\n- PANW: selling."
    assert ungrounded_tickers(source, good) == []
    assert ungrounded_tickers(source, good + "\n- ZZZZ: invented") == [
        "ZZZZ is not in the source items"
    ]

    used = []

    def fallback():
        used.append(1)
        return type("A", (), {"run": lambda self, p: type("R", (), {"content": "claude"})()})()

    c = _client(tmp_path, [("- ZZZZ: invented", 0.001)])
    agent = OpenRouterAgent(
        "Insider", "z-ai/glm-5.3", "sys", client=c, fallback=fallback, validate=ungrounded_tickers
    )
    assert agent.run(source).content == "claude" and used == [1]


def test_spot_check_stores_claude_agreement_per_host(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    for n, provider in enumerate(["Io Net", "Baidu"]):
        read = fr.FilingRead(
            filing={**FILING, "ticker": f"T{n}", "accession": f"a{n}"},
            reader_model="z-ai/glm-5.3",
            facts=FACTS_OK,
            provider=provider,
        )
        filings.store(db, read, tier="A", today=TODAY)
    page = f"ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS\n{'Revenue grew. ' * 40}\nITEM 3. x"
    monkeypatch.setattr(filings, "fetch_filing_text", lambda url: page)
    monkeypatch.setattr(filings, "claude_read", lambda s, f, sec: {**FACTS_OK, "tone": "neutral"})
    monkeypatch.setattr(filings, "log_usage_summary", lambda: None)
    settings = filings.Settings(discover_db_path=db)
    assert filings.spot_check(settings, 5, today=TODAY, seed=1) == 0
    rows = {r["provider"]: r for r in oh.spot_check_summary(db, today=TODAY)}
    assert set(rows) == {"Io Net", "Baidu"}
    assert rows["Baidu"]["compared"] == 7 and rows["Baidu"]["agreed"] == 6  # tone differs
    # Already checked: a second run finds nothing new.
    assert filings.spot_check(settings, 5, today=TODAY) == 0
    assert sum(r["filings"] for r in oh.spot_check_summary(db, today=TODAY)) == 2
