"""Company news events for watched stocks, in the evening alert."""

from __future__ import annotations

import json
from datetime import date

from stock_analyzer.data import event_news as en
from stock_analyzer.reporting import event_watch as ew
from stock_analyzer.reporting import filing_alert as fa
from stock_analyzer.reporting.drop_alert import build_alert

from .test_filing_alert import EIGHTK, _filing
from .test_filing_reader import _client

TODAY = date(2026, 10, 12)

FEED = """<?xml version="1.0"?><rss><channel>
<item><title>FERC rejects Oklo complaint on PJM study cycle - Utility Dive</title>
 <link>https://news.google.com/a</link><pubDate>Fri, 10 Oct 2026 12:00:00 GMT</pubDate>
 <source url="https://www.utilitydive.com">Utility Dive</source></item>
<item><title>Is Oklo the Next Trillion-Dollar Stock? - The Motley Fool</title>
 <link>https://news.google.com/b</link><pubDate>Sat, 11 Oct 2026 12:00:00 GMT</pubDate>
 <source url="https://www.fool.com">The Motley Fool</source></item>
<item><title>Oklo signs power agreement with data center operator - Business Wire</title>
 <link>https://news.google.com/c</link><pubDate>Sat, 11 Oct 2026 13:00:00 GMT</pubDate>
 <source url="https://www.businesswire.com">Business Wire</source></item>
<item><title>Oklo inks data center power deal - Reuters</title>
 <link>https://news.google.com/d</link><pubDate>Sat, 11 Oct 2026 14:00:00 GMT</pubDate>
 <source url="https://www.reuters.com">Reuters</source></item>
<item><title>Oklo signs supply agreement in 2025 - Old News</title>
 <link>https://news.google.com/e</link><pubDate>Mon, 01 Sep 2026 12:00:00 GMT</pubDate>
 <source url="https://old.example">Old News</source></item>
<item><title>NuScale signs agreement with TVA - Reuters</title>
 <link>https://news.google.com/f</link><pubDate>Sat, 11 Oct 2026 12:00:00 GMT</pubDate>
 <source url="https://www.reuters.com">Reuters</source></item>
</channel></rss>"""


def test_event_headlines_stay_and_commentary_goes():
    for title in (
        "FERC rejects Oklo complaint seeking to reinstate project",
        "Oklo Gains NRC Approval for Aurora license application",
        "Oklo signs 500 MW power agreement with Switch",
        "Oklo Posts Its First Quarterly Revenue",
    ):
        assert en.looks_like_event(title), title
    for title in (
        "Is Oklo the Next Trillion-Dollar Stock?",
        "NuScale vs. Oklo: Which Nuclear Stock is the Better Buy Now",
        "Oklo Jumps 13%, NuScale Power Climbs 10%",
        "Canaccord reiterates Oklo stock Buy rating",
        "OKLO Hits a 52-Week Low",
        "Everything You Need to Know About Oklo's Strategy",
    ):
        assert not en.looks_like_event(title), title


def test_query_and_feed():
    assert en.company_query("Oklo Inc.") == '"Oklo"'
    assert en.company_query("REPUBLIC SERVICES, INC.") == '"REPUBLIC SERVICES"'
    items = en.parse_feed(FEED)
    assert items[0]["title"] == "FERC rejects Oklo complaint on PJM study cycle"
    assert items[0]["source"] == "Utility Dive" and items[0]["published"] == date(2026, 10, 10)
    assert en.parse_feed("not xml") == []


def _feed(monkeypatch, text=FEED):
    class R:
        def __init__(self, t):
            self.text = t

    urls = []
    monkeypatch.setattr(en._HTTP, "get", lambda url: urls.append(url) or R(text))
    return urls


def test_fetch_keeps_recent_events_naming_the_company(monkeypatch):
    urls = _feed(monkeypatch)
    got = en.fetch_events("Oklo Inc.", today=TODAY)
    assert "%22Oklo%22+when%3A3d" in urls[0]
    # Commentary, a month-old item and another company's news are dropped.
    assert [i["source"] for i in got] == ["Business Wire", "Reuters", "Utility Dive"]


READ = {
    "events": [
        {
            "i": 1,
            "same_as": [2],
            "kind": "customer",
            "direction": "positive",
            "what": "Oklo signed a power agreement with a data center operator.",
            "why": "A signed customer is a step toward revenue.",
        },
        {"i": 3, "same_as": [], "kind": "regulatory", "direction": "negative",
         "what": "FERC rejected Oklo's PJM complaint.", "why": "Grid access may be delayed."},
        {"i": 9, "kind": "other"},  # out of range: ignored
    ]
}  # fmt: skip


def test_new_events_are_read_once_and_repeats_grouped(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    _feed(monkeypatch)
    c = _client(tmp_path, [(json.dumps(READ), 0.0005)])
    c.db_path = db
    got = ew.new_watch_events(c, db, {"OKLO": "Oklo Inc."}, today=TODAY, model="z-ai/glm-5.3-flash")
    assert [(e["kind"], e["source"]) for e in got] == [
        ("customer", "Business Wire"),
        ("regulatory", "Utility Dive"),
    ]
    assert [a["source"] for a in got[0]["also"]] == ["Reuters"]
    sent = c._http.bodies[0]["messages"][1]["content"]  # type: ignore[attr-defined]
    assert "Company: Oklo Inc. (OKLO)" in sent and "3. [2026-10-10] Utility Dive" in sent

    # The next evening the same headlines are neither read nor sent again.
    again = _client(tmp_path, [])
    assert ew.new_watch_events(again, db, {"OKLO": "Oklo"}, today=TODAY, model="z-ai/m") == []
    assert again._http.bodies == []  # type: ignore[attr-defined]

    subject, body = build_alert([], [], [], got)
    assert subject == "Holding alert: OKLO customer, OKLO regulatory"
    assert "<h3>Company news</h3>" in body and "▲ customer" in body
    assert "Also reported by" in body and "open the link before acting" in body


def test_watched_companies_get_their_press_release_8ks(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(
        fa, "filings_since", lambda t, since, forms=(): [_filing("8-K", "k9", ["8.01"])]
    )
    monkeypatch.setattr(fa, "fetch_filing_text", lambda url: "Oklo announces a customer.")
    monkeypatch.setattr(fa, "exhibit_99_text", lambda f: "Press release")
    # Not watched: an 8.01 alone is routine.
    c = _client(tmp_path, [])
    assert fa.new_holding_filings(c, db, ["ABC"], today=TODAY, model="z-ai/m") == []
    c = _client(tmp_path, [(json.dumps(EIGHTK), 0.004)])
    c.db_path = db
    got = fa.new_holding_filings(
        c, db, ["ABC"], today=TODAY, model="z-ai/m", watched=frozenset({"ABC"})
    )
    assert [i["filing"]["accession"] for i in got] == ["k9"]
    assert fa.subject_part(got[0]) == "ABC 8-K (other event)"
