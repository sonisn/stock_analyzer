"""A held stock's new SEC filings, read and emailed the same evening."""

from __future__ import annotations

import json
from datetime import date

from stock_analyzer.data import sec_edgar
from stock_analyzer.reporting import filing_alert as fa
from stock_analyzer.reporting.drop_alert import build_alert

from .test_filing_reader import FACTS_OK, MDA, _client

TODAY = date(2026, 9, 25)


def test_filings_since_keeps_dates_items_and_accessions(monkeypatch):
    monkeypatch.setattr(sec_edgar, "_load_ticker_map", lambda: {"ABC": 42})
    recent = {
        "form": ["8-K", "10-Q", "8-K", "4"],
        "accessionNumber": ["0001-26-3", "0001-26-2", "0001-26-1", "0001-26-0"],
        "primaryDocument": ["e.htm", "q.htm", "old.htm", "f4.xml"],
        "filingDate": ["2026-09-25", "2026-09-24", "2026-08-01", "2026-09-25"],
        "reportDate": ["", "2026-06-30", "", ""],
        "items": ["2.02,9.01", "", "5.07", ""],
    }
    monkeypatch.setattr(sec_edgar._HTTP, "get_json", lambda url: {"filings": {"recent": recent}})
    got = sec_edgar.filings_since("abc", date(2026, 9, 21))
    assert [(f["form"], f["accession"]) for f in got] == [
        ("10-Q", "0001-26-2"),
        ("8-K", "0001-26-3"),
    ]
    assert got[1]["items"] == ["2.02", "9.01"]
    assert got[1]["url"].endswith("/42/0001263/e.htm")


def test_exhibit_99_is_the_press_release(monkeypatch):
    listing = {
        "directory": {
            "item": [{"name": "abc-20260925.htm"}, {"name": "abc-x8kxex99.htm"}, {"name": "R1.htm"}]
        }
    }
    monkeypatch.setattr(sec_edgar._HTTP, "get_json", lambda url: listing)
    fetched = []
    monkeypatch.setattr(sec_edgar, "fetch_filing_text", lambda url: fetched.append(url) or "PR")
    f = {"url": "https://www.sec.gov/Archives/edgar/data/42/0001/abc.htm", "accession": "a"}
    assert sec_edgar.exhibit_99_text(f) == "PR"
    assert fetched == ["https://www.sec.gov/Archives/edgar/data/42/0001/abc-x8kxex99.htm"]


EIGHTK = {
    "headline": "Record revenue, guidance raised",
    "what_happened": "Q3 revenue rose 22%.",
    "guidance": {"direction": "raised", "detail": "FY revenue to $70B", "quote": ""},
    "numbers": [{"metric": "revenue", "value": "$18.1B", "vs_prior": "+22% y/y", "quote": ""}],
    "events": [],
    "tone": "positive",
}


def _filing(form, acc, items=()):
    return {
        "ticker": "ABC",
        "cik": 42,
        "accession": acc,
        "form": form,
        "filed_on": "2026-09-25",
        "period_end": "2026-06-30" if form != "8-K" else None,
        "items": list(items),
        "url": f"https://www.sec.gov/Archives/edgar/data/42/{acc}/doc.htm",
    }


def test_new_filings_are_read_once_and_routine_8ks_skipped(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    filings = [
        _filing("10-Q", "q1"),
        _filing("8-K", "k1", ["2.02", "9.01"]),
        _filing("8-K", "k2", ["5.07"]),  # shareholder vote: routine
    ]
    monkeypatch.setattr(fa, "filings_since", lambda t, since, forms=(): filings)
    page = f"ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS\n{MDA}\nITEM 3. QUANTITATIVE AND x"
    monkeypatch.setattr(fa, "fetch_filing_text", lambda url: page)
    monkeypatch.setattr(fa, "exhibit_99_text", lambda f: "Press release")
    c = _client(tmp_path, [(json.dumps(FACTS_OK), 0.01), (json.dumps(EIGHTK), 0.004)])
    c.db_path = db

    got = fa.new_holding_filings(c, db, ["ABC"], today=TODAY, model="z-ai/glm-5.3")
    assert [(i["kind"], i["filing"]["accession"]) for i in got] == [
        ("periodic", "q1"),
        ("8-K", "k1"),
    ]
    sent = c._http.bodies[1]["messages"][1]["content"]  # type: ignore[attr-defined]
    assert "2.02 (results)" in sent and "EXHIBIT 99" in sent and "Press release" in sent

    # The next evening: both are stored, nothing is read or sent again.
    again = _client(tmp_path, [])
    assert fa.new_holding_filings(again, db, ["ABC"], today=TODAY, model="m") == []
    assert again._http.bodies == []  # type: ignore[attr-defined]


def test_the_cap_stops_the_check_without_losing_what_was_read(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(
        fa, "filings_since", lambda t, since, forms=(): [_filing("8-K", "k1", ["5.02"])]
    )
    monkeypatch.setattr(fa, "fetch_filing_text", lambda url: "The CFO resigned.")
    monkeypatch.setattr(fa, "exhibit_99_text", lambda f: None)
    c = _client(tmp_path, [], cap=0.0000001)
    assert fa.new_holding_filings(c, db, ["ABC"], today=TODAY, model="z-ai/glm-5.3") == []


def test_email_shows_the_filing_with_its_link_and_what_changed():
    periodic = {
        "kind": "periodic",
        "filing": _filing("10-Q", "q1"),
        "pack": {
            "period_end": "2026-06-30",
            "summary": "Revenue +24%.",
            "demand": "accelerating: AI networking",
            "vs_prior_filing": {"period_end": "2026-03-31", "demand": "steady -> accelerating"},
            "events": [
                {
                    "issue": "Material weakness",
                    "category": "material_weakness",
                    "severity": "high",
                    "quote": "we identified a material weakness",
                }
            ],
            "quotes_verified": "11/11",
        },
    }
    eightk = {"kind": "8-K", "filing": _filing("8-K", "k1", ["2.02"]), "summary": EIGHTK}
    subject, body = build_alert([], [], [periodic, eightk])
    assert subject == "Holding alert: ABC 10-Q, ABC 8-K (results)"
    for text in (
        "New SEC filings",
        "doc.htm",
        "Revenue +24%.",
        "demand steady -&gt; accelerating",
        "⚠ Material weakness",
        "Guidance: <b>raised</b>",
        "revenue: $18.1B (+22% y/y)",
    ):
        assert text in body, text


def test_press_release_without_an_ex99_name_is_the_largest_other_document(monkeypatch):
    listing = {
        "directory": {
            "item": [
                {"name": "nvda-20260826.htm", "size": "26457"},
                {"name": "q2fy27cfocommentary.htm", "size": "256881"},
                {"name": "q2fy27pr.htm", "size": "341113"},
                {"name": "R1.htm", "size": "938220"},
            ]
        }
    }
    monkeypatch.setattr(sec_edgar._HTTP, "get_json", lambda url: listing)
    monkeypatch.setattr(sec_edgar, "fetch_filing_text", lambda url: url.rsplit("/", 1)[1])
    f = {"url": "https://www.sec.gov/Archives/edgar/data/1/2/nvda-20260826.htm", "accession": "a"}
    assert sec_edgar.exhibit_99_text(f) == "q2fy27pr.htm"


def test_tesla_names_its_press_release_exhibit991(monkeypatch):
    listing = {
        "directory": {
            "item": [
                {"name": "0001628280-26-049213-index.html", "size": ""},
                {"name": "exhibit991.htm", "size": "52312"},
                {"name": "tsla-20260722.htm", "size": "30000"},
            ]
        }
    }
    monkeypatch.setattr(sec_edgar._HTTP, "get_json", lambda url: listing)
    monkeypatch.setattr(sec_edgar, "fetch_filing_text", lambda url: url.rsplit("/", 1)[1])
    f = {"url": "https://www.sec.gov/Archives/edgar/data/1/2/tsla-20260722.htm", "accession": "a"}
    assert sec_edgar.exhibit_99_text(f) == "exhibit991.htm"


def test_the_latest_earnings_release_is_read_once_and_reaches_the_deciders(tmp_path, monkeypatch):
    from stock_analyzer.data.filing_evidence import earnings_releases

    db = str(tmp_path / "t.db")
    older = {**_filing("8-K", "k0", ["2.02"]), "filed_on": "2026-07-25"}
    newer = _filing("8-K", "k1", ["2.02", "9.01"])
    asked = []

    def since(t, day, forms=()):
        asked.append(forms)
        return [older, _filing("8-K", "k2", ["5.02"]), newer]

    monkeypatch.setattr(fa, "filings_since", since)
    monkeypatch.setattr(fa, "fetch_filing_text", lambda url: "Results.")
    monkeypatch.setattr(fa, "exhibit_99_text", lambda f: "Press release")
    c = _client(tmp_path, [(json.dumps(EIGHTK), 0.004)])
    c.db_path = db
    got = fa.earnings_releases(c, db, ["ABC"], today=TODAY, model="z-ai/glm-5.3")
    assert [i["filing"]["accession"] for i in got] == ["k1"] and asked == [("8-K",)]
    # Stored, so the next week reads nothing.
    again = _client(tmp_path, [])
    assert fa.earnings_releases(again, db, ["ABC"], today=TODAY, model="m") == []

    assert earnings_releases(db, ["abc", "NONE"]) == {
        "ABC": {
            "filed_on": "2026-09-25",
            "headline": "Record revenue, guidance raised",
            "guidance": "raised: FY revenue to $70B",
            "numbers": ["revenue $18.1B +22% y/y"],
        }
    }


def test_a_huge_filing_is_read_only_up_to_the_cap(monkeypatch):
    class Resp:
        url = "u"
        encoding = "utf-8"
        content = b"<p>Item 7. MD&amp;A text</p>" + b"x" * 200

    monkeypatch.setattr(sec_edgar, "MAX_FILING_BYTES", 40)
    assert sec_edgar.capped_text(Resp()) == "<p>Item 7. MD&amp;A text</p>" + "x" * 12


def test_a_planned_sale_logs_who_and_how_much():
    item = {
        "kind": "event",
        "event": "planned_sale",
        "filing": {"ticker": "BE", "form": "144"},
        "facts": {
            "seller": "A. OFFICER",
            "relationship": "Officer",
            "value_usd": 2336880.26,
            "sale_date": "10/01/2026",
        },
    }
    assert fa.summary_line(item) == (
        "BE planned insider sale: A. OFFICER (Officer), $2.3M around 10/01/2026"
    )
