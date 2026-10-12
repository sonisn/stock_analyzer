"""The moonshot basket: entry screen, spread, sell rules, stored members."""

from __future__ import annotations

import math
from datetime import date

from stock_analyzer.discover import moonshot as ms

TODAY = date(2026, 3, 1)


def _year(fy: int, val: float) -> dict:
    return {
        "start": f"{fy}-01-01",
        "end": f"{fy}-12-31",
        "val": val,
        "filed": f"{fy + 1}-02-15",
        "accn": f"a{fy}",
    }


def _instant(fy: int, val: float) -> dict:
    return {"end": f"{fy}-12-31", "val": val, "filed": f"{fy + 1}-02-15", "accn": f"a{fy}"}


def _body(
    *,
    revenue=(100.0, 130.0, 170.0),  # FY2023-2025
    gross=(40.0, 55.0, 75.0),
    op=(-10.0, -8.0, -5.0),
    cfo=(-10.0, -10.0, -12.0),
    capex=(2.0, 2.0, 3.0),
    cash=60.0,
    shares=(100.0, 104.0, 108.0),
) -> dict:
    years = (2023, 2024, 2025)

    def flow(vals):
        return {"units": {"USD": [_year(y, v) for y, v in zip(years, vals, strict=True)]}}

    return {
        "facts": {
            "us-gaap": {
                "Revenues": flow(revenue),
                "GrossProfit": flow(gross),
                "OperatingIncomeLoss": flow(op),
                "NetCashProvidedByUsedInOperatingActivities": flow(cfo),
                "PaymentsToAcquirePropertyPlantAndEquipment": flow(capex),
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_instant(2025, cash)]}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": f"{y + 1}-02-01", "val": v, "filed": f"{y + 1}-02-15"}
                            for y, v in zip(years, shares, strict=True)
                        ]
                    }
                }
            },
        }
    }


def test_snapshot_reads_growth_margins_dilution_and_runway_as_filed():
    s = ms.snapshot(_body(), today=TODAY)
    assert (s.revenue, s.revenue_1y, s.revenue_2y) == (170.0, 130.0, 100.0)
    assert round(s.revenue_growth, 3) == round(170 / 130 - 1, 3)
    assert round(s.operating_margin, 3) == round(-5 / 170, 3)
    assert round(s.gross_margin_change, 3) == round(75 / 170 - 55 / 130, 3)
    assert s.free_cash_flow == -15.0 and s.cash == 60.0 and s.runway_years == 4.0
    assert round(s.dilution, 3) == round(108 / 104 - 1, 3)


def test_snapshot_does_not_read_a_split_as_a_share_issue():
    # 104 shares a year ago, a 2-for-1 in June 2025, 216 on today's cover page.
    body = _body(shares=(100.0, 104.0, 216.0))
    assert ms.snapshot(body, today=TODAY).dilution > 1  # without the split: "108%"
    s = ms.snapshot(body, today=TODAY, splits=[(date(2025, 6, 1), 2.0)])
    assert (s.shares, s.shares_1y, s.shares_2y) == (216.0, 208.0, 200.0)
    assert round(s.dilution, 3) == round(216 / 208 - 1, 3)


def test_entry_rules():
    good = ms.snapshot(_body(), today=TODAY)
    assert ms.entry_reasons(good, market_cap=500e6, dollar_volume=5e6) == []
    assert ms.entry_reasons(good, market_cap=3e9, dollar_volume=5e6) == [
        "market cap outside $150M-$2B"
    ]
    slow = ms.snapshot(_body(revenue=(100.0, 130.0, 140.0)), today=TODAY)
    assert "revenue growth under 25%" in ms.entry_reasons(slow, market_cap=5e8, dollar_volume=5e6)
    deep_loss = ms.snapshot(_body(op=(-50.0, -50.0, -60.0)), today=TODAY)
    assert "operating margin outside -20% to +5%" in ms.entry_reasons(
        deep_loss, market_cap=5e8, dollar_volume=5e6
    )
    # Low and falling gross margin fails; low but rising passes.
    falling = ms.snapshot(_body(gross=(30.0, 40.0, 40.0)), today=TODAY)
    assert "gross margin under 40% and not rising" in ms.entry_reasons(
        falling, market_cap=5e8, dollar_volume=5e6
    )
    rising = ms.snapshot(_body(gross=(20.0, 30.0, 50.0)), today=TODAY)
    assert ms.entry_reasons(rising, market_cap=5e8, dollar_volume=5e6) == []
    diluting = ms.snapshot(_body(shares=(100.0, 100.0, 125.0)), today=TODAY)
    assert any(
        "share count" in r for r in ms.entry_reasons(diluting, market_cap=5e8, dollar_volume=5e6)
    )
    short_cash = ms.snapshot(_body(cash=20.0), today=TODAY)
    assert "under two years of cash" in ms.entry_reasons(
        short_cash, market_cap=5e8, dollar_volume=5e6
    )
    pre_revenue = ms.Snapshot(day=TODAY)
    assert "no revenue" in ms.entry_reasons(pre_revenue, market_cap=5e8, dollar_volume=5e6)


def test_sell_rules_come_from_filings_only():
    intact = ms.snapshot(_body(), today=TODAY)
    assert ms.thesis_breaks(intact) == []
    assert ms.thesis_breaks(intact, red_flags=["going_concern"]) == [
        "going-concern doubt in the latest filing"
    ]
    broke = ms.snapshot(
        _body(revenue=(170.0, 130.0, 100.0), cash=10.0, shares=(100.0, 120.0, 150.0)),
        today=TODAY,
    )
    why = " | ".join(ms.thesis_breaks(broke))
    assert "under a year of cash" in why
    assert "share count up 25% and 20%" in why
    assert "revenue down two years running" in why
    # Cash-generative: infinite runway, never a cash break.
    rich = ms.snapshot(_body(cfo=(10.0, 20.0, 30.0)), today=TODAY)
    assert rich.runway_years == math.inf and ms.thesis_breaks(rich) == []


def _cand(t, cap, industry="Software", sic=7372):
    return ms.Candidate(t, t, cap, ms.Snapshot(day=TODAY), industry, sic)


def test_fill_slots_smallest_first_within_the_spread_rules():
    cands = [
        _cand("S1", 900e6),
        _cand("S2", 300e6),
        _cand("S3", 200e6),
        _cand("S4", 160e6),
        _cand("B1", 170e6, "Pharma", 2834),
        _cand("B2", 180e6, "Biotech", 2836),
    ]
    # 4 slots: biotech capped at 1, software at 3.
    got = [c.ticker for c in ms.fill_slots(cands, open_slots=6, slots=4)]
    assert got == ["S4", "B1", "S3", "S2"]
    # A software member already held leaves room for two more.
    held = [_cand("H1", 0.0)]
    got = [c.ticker for c in ms.fill_slots(cands, open_slots=6, slots=20, held=held)]
    assert got.count("S4") == 1 and len([t for t in got if t.startswith("S")]) == 2


def test_members_round_trip(tmp_path):
    db = str(tmp_path / "d.db")
    ms.add_member(db, "acme", today=TODAY, entry_price=12.5)
    ms.add_member(db, "ZETA", today=TODAY, entry_price=None)
    assert {m.ticker for m in ms.load_members(db)} == {"ACME", "ZETA"}
    assert ms.remove_member(db, "acme", today=TODAY, reason="revenue down two years")
    assert not ms.remove_member(db, "acme", today=TODAY, reason="again")
    assert [m.ticker for m in ms.load_members(db)] == ["ZETA"]
    # Re-adding a removed stock starts a new holding.
    ms.add_member(db, "ACME", today=date(2026, 12, 2), entry_price=9.0)
    assert {m.ticker for m in ms.load_members(db)} == {"ACME", "ZETA"}
