"""SEC event filings (data/sec_events, reporting/filing_alert): parsing,
the 13D index scan, alert thresholds and the email — no network."""

from __future__ import annotations

import json
from datetime import date

from stock_analyzer.data import sec_events as se
from stock_analyzer.reporting import filing_alert as fa

from .test_filing_reader import _client

TODAY = date(2026, 9, 28)

FORM_144 = """<edgarSubmission><formData><issuerInfo>
<nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>H&amp;S INVESTMENTS I LP</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
<relationshipsToIssuer><relationshipToIssuer>Chairman</relationshipToIssuer></relationshipsToIssuer>
</issuerInfo><securitiesInformation><noOfUnitsSold>70218</noOfUnitsSold>
<aggregateMarketValue>25000000.00</aggregateMarketValue><noOfUnitsOutstanding>4773629865</noOfUnitsOutstanding>
<approxSaleDate>09/23/2026</approxSaleDate></securitiesInformation>
<securitiesToBeSold><natureOfAcquisitionTransaction>Restricted Stock Units</natureOfAcquisitionTransaction></securitiesToBeSold>
</formData></edgarSubmission>"""

FORM_13D = """<edgarSubmission><formData><coverPageHeader><issuerInfo><issuerCIK>0000111111</issuerCIK>
<issuerName>Target Co</issuerName></issuerInfo></coverPageHeader>
<reportingPersons><reportingPersonInfo><reportingPersonName>Elliott Investment Management L.P.</reportingPersonName>
<percentOfClass>5.8</percentOfClass></reportingPersonInfo></reportingPersons>
<items1To7><item4><transactionPurpose>The Reporting Person intends to seek board representation.</transactionPurpose></item4></items1To7>
</formData></edgarSubmission>"""


def test_form_144_fields_are_parsed_and_unescaped():
    got = se.parse_planned_sale(FORM_144)
    assert got["seller"] == "H&S INVESTMENTS I LP" and got["relationship"] == "Chairman"
    assert got["value_usd"] == 25_000_000 and got["shares"] == 70218
    assert got["pct_of_outstanding"] == 0.001 and got["acquired_as"] == "Restricted Stock Units"


def test_13d_fields_and_issuer():
    got = se.parse_13d(FORM_13D)
    assert got["holders"] == ["Elliott Investment Management L.P."] and got["percent"] == 5.8
    assert got["issuer_cik"] == 111111 and "board representation" in got["purpose"]


def test_daily_index_returns_each_listed_party(monkeypatch):
    index = "\n".join(
        [
            "Form Type   Company Name   CIK   Date Filed  File Name",
            "SCHEDULE 13D/A   Elliott Investment Management L.P.   1791786   20260921   edgar/data/1791786/0000950142-26-000001.txt",
            "SCHEDULE 13D/A   Target Co   111111   20260921   edgar/data/111111/0000950142-26-000001.txt",
            "SCHEDULE 13D/A   Unlisted Holding   999   20260921   edgar/data/999/0000950142-26-000002.txt",
            "10-K   Target Co   111111   20260921   edgar/data/111111/0000950142-26-000003.txt",
        ]
    )
    monkeypatch.setattr(se._HTTP, "get", lambda url: type("R", (), {"text": index})())
    monkeypatch.setattr(se, "load_ticker_cik_map", lambda: {"TGT": 111111, "OTHER": 222})
    got = se.thirteen_d_targets(date(2026, 9, 21), {"TGT", "OTHER"})
    assert [(g["ticker"], g["accession"], g["holders"]) for g in got] == [
        ("TGT", "0000950142-26-000001", ["Elliott Investment Management L.P."])
    ]


def _filing(ticker, form, acc, cik=111111):
    return {
        "ticker": ticker,
        "cik": cik,
        "accession": acc,
        "form": form,
        "filed_on": "2026-09-25",
        "items": [],
        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/xsl/primary_doc.xml",
    }


def test_small_planned_sales_are_recorded_not_alerted(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    small = FORM_144.replace("25000000.00", "400000.00")
    monkeypatch.setattr(fa, "primary_xml", lambda f: small if f["accession"] == "s" else FORM_144)
    c = _client(tmp_path, [])
    assert fa.read_event(c, db, _filing("AVGO", "144", "s"), model="x/m", today=TODAY) is None
    big = fa.read_event(c, db, _filing("AVGO", "144", "b"), model="x/m", today=TODAY)
    assert big and "plans to sell 70,218 shares, $25.0M" in fa.filing_block(big)
    assert "s" in fa._seen(db)  # recorded, so not fetched again


def test_a_13d_is_kept_for_the_issuer_only_and_classified(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(fa, "primary_xml", lambda f: FORM_13D)
    read = {
        "stance": "activist",
        "headline": "Elliott seeks board seats",
        "demands": "board",
        "quote": "",
    }
    c = _client(tmp_path, [(json.dumps(read), 0.001)])
    # The index row for the holder's own listed company is not the target.
    assert (
        fa.read_event(
            c, db, _filing("EHLD", "SCHEDULE 13D", "x", cik=222), model="x/m", today=TODAY
        )
        is None
    )
    item = fa.read_event(c, db, _filing("TGT", "SCHEDULE 13D", "y"), model="x/m", today=TODAY)
    assert item and "Elliott seeks board seats" in fa.filing_block(item)
    assert fa.subject_part(item) == "TGT activist stake"
    assert fa.activist_targets(db, today=TODAY) == ["TGT"]


def test_an_offering_says_what_is_sold_and_the_dilution(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(
        fa, "fetch_filing_text", lambda url: "We may offer up to $1,000,000,000 of common stock."
    )
    monkeypatch.setattr(
        fa,
        "shares_change",
        lambda cik, today: {"latest": 186e6, "year_ago": 147.6e6, "change_pct": 26.0},
    )
    offering = {
        "security": "common",
        "at_the_market": True,
        "amount_usd_millions": 1000,
        "headline": "$1B at-the-market program",
        "quote": "We may offer up to $1,000,000,000 of common stock.",
    }
    c = _client(tmp_path, [(json.dumps(offering), 0.002)])
    item = fa.read_event(c, db, _filing("OKLO", "424B5", "o"), model="x/m", today=TODAY)
    block = fa.filing_block(item)
    assert "common (at-the-market program), $1.00B" in block
    assert "Shares outstanding +26.0% in a year" in block


def test_late_filing_needs_no_model(tmp_path):
    db = str(tmp_path / "t.db")
    c = _client(tmp_path, [])
    item = fa.read_event(c, db, _filing("XYZ", "NT 10-Q", "n"), model="x/m", today=TODAY)
    assert "cannot file its report on time" in fa.filing_block(item)
    assert c._http.bodies == []  # type: ignore[attr-defined]


def test_dashboard_sec_highlights_carry_pack_release_and_events(tmp_path):
    from stock_analyzer.agents import filing_reader as fr
    from stock_analyzer.cli import filings
    from stock_analyzer.cli.dashboard import _sec_highlights
    from stock_analyzer.dashboard_page import render_page

    from .test_filing_reader import FACTS_OK, FILING

    db = str(tmp_path / "t.db")
    filings.store(
        db, fr.FilingRead(filing=FILING, reader_model="x/m", facts=FACTS_OK), tier="A", today=TODAY
    )
    c = _client(tmp_path, [])
    fa.read_event(c, db, _filing("ABC", "NT 10-Q", "n"), model="x/m", today=TODAY)
    sec = _sec_highlights(db, ["ABC", "NONE"], today=TODAY)
    assert set(sec) == {"ABC"}
    assert sec["ABC"]["pack"]["tone"] == "positive"
    assert sec["ABC"]["events"][0]["kind"] == "late-filing notice"
    assert "cannot file its report on time" in sec["ABC"]["events"][0]["lines"][0]
    page = render_page(
        {
            "holdings": [],
            "record": {"rows": 0, "tickers": 0, "first": "", "last": ""},
            "latest_run": 1,
            "generated": "2026-09-28",
            "holdings_ok": True,
            "ibd": {},
            "sec": sec,
        }
    )
    assert "SEC events on your holdings" in page and "late-filing notice" in page
