"""Tracked hedge funds' quarter-over-quarter moves, from 13F filings."""

from __future__ import annotations

from stock_analyzer.data import hedge_funds_13f as hf
from stock_analyzer.reporting.health import build_portfolio_health, render_health_html

FUND = next(iter(hf.FUNDS))  # Berkshire Hathaway's CIK
NAME = hf.FUNDS[FUND]


def _fake(periods: dict[str, dict[str, float]]):
    """filings() and holdings() for one fund with the given {period: {cusip: shares}}."""
    order = sorted(periods, reverse=True)

    def filings(cik):
        if cik != FUND:
            return []
        return [{"accession": f"acc-{p}", "period": p, "filed": p} for p in order]

    def holdings(cik, accession):
        period = accession.removeprefix("acc-")
        return {c: {"shares": n, "value_usd": n * 10} for c, n in periods[period].items()}

    return filings, holdings


LOOKUP = {"C_AAPL": "AAPL", "C_NEW": "NEWCO", "C_GONE": "GONE", "C_FLAT": "FLAT", "C_BOND": None}


def test_sync_stores_new_quarters_once_and_changes_compare_the_latest_two(tmp_path):
    db = str(tmp_path / "f.db")
    filings, holdings = _fake(
        {
            "2026-03-31": {"C_AAPL": 100, "C_GONE": 50, "C_FLAT": 100, "C_BOND": 5},
            "2026-06-30": {"C_AAPL": 160, "C_NEW": 20, "C_FLAT": 104, "C_BOND": 5},
        }
    )
    asked: list[list[str]] = []

    def lookup(cusips):
        asked.append(sorted(cusips))
        return {c: LOOKUP[c] for c in cusips}

    first = hf.sync(db, filings=filings, holdings=holdings, lookup=lookup)
    assert first["new_filings"] == 2
    again = hf.sync(db, filings=filings, holdings=holdings, lookup=lookup)
    assert again["new_filings"] == 0
    # Each CUSIP is looked up once, ever.
    assert sum(len(a) for a in asked) == len(LOOKUP)

    moves = hf.changes(db)
    assert set(moves) == {"AAPL", "NEWCO", "GONE"}  # FLAT's +4% is a rebalance; BOND unmapped
    (aapl,) = moves["AAPL"]
    assert (aapl["fund"], aapl["action"], round(aapl["shares_change_pct"])) == (NAME, "added", 60)
    assert moves["NEWCO"][0]["action"] == "new" and moves["GONE"][0]["action"] == "exited"
    assert hf.changes(db, ["aapl"]).keys() == {"AAPL"}
    assert hf.summarize(moves["AAPL"] + moves["GONE"]) == (
        f"1 buying ({NAME} +60% to 55.4% of fund); 1 selling ({NAME} sold out, was 19.6%)"
    )


def test_only_the_latest_four_quarters_are_kept(tmp_path):
    db = str(tmp_path / "k.db")
    quarters = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]
    for q in quarters:  # a new filing each quarter, as the nightly job would see it
        filings, holdings = _fake({q: {"C_AAPL": 1}})
        hf.sync(db, filings=filings, holdings=holdings, lookup=lambda cs: {c: "AAPL" for c in cs})
    from sqlalchemy import text

    from stock_analyzer.db.session import get_session

    with get_session(db) as session:
        kept = sorted(
            p for (p,) in session.execute(text("SELECT DISTINCT period FROM fund_positions")).all()
        )
    assert kept == quarters[1:]


def test_holdings_in_the_email_show_the_quarter_and_the_moves():
    moves = {
        "AVGO": [
            {
                "fund": "Coatue",
                "action": "added",
                "shares_change_pct": 40.0,
                "weight_pct": 6.0,
                "weight_before_pct": 4.3,
                "value_usd": 9e8,
                "period": "2026-06-30",
                "filed": "2026-08-14",
            },
            {
                "fund": "Tiger Global",
                "action": "exited",
                "shares_change_pct": -100.0,
                "weight_pct": 0.0,
                "weight_before_pct": 3.2,
                "value_usd": 1e8,
                "period": "2026-06-30",
                "filed": "2026-08-14",
            },
        ]
    }
    held = {"IRA": [{"ticker": "AVGO", "units": 10, "price": 350.0}]}
    health = build_portfolio_health(held, prices={"AVGO": 350.0}, fund_moves=lambda t: moves)
    body = render_health_html(health)
    assert "Hedge funds in your holdings (13F)" in body and "quarter ending 2026-06-30" in body
    assert (
        "1 buying (Coatue +40% to 6.0% of fund); 1 selling (Tiger Global sold out, was 3.2%)"
        in body
    )
    assert "Hedge funds in your holdings" not in render_health_html(build_portfolio_health(held))


def test_only_conviction_moves_count_and_consensus_is_flagged(tmp_path):
    db = str(tmp_path / "w.db")
    # Big positions (1,000 shares = $10k) and one tiny one (10 = $100, 0.5%).
    filings, holdings = _fake(
        {
            "2026-03-31": {"C_AAPL": 1000, "C_FLAT": 1000, "C_GONE": 10},
            "2026-06-30": {"C_AAPL": 1000, "C_FLAT": 1000, "C_NEW": 10},
        }
    )
    hf.sync(db, filings=filings, holdings=holdings, lookup=lambda cs: {c: LOOKUP[c] for c in cs})
    moves = hf.changes(db)
    assert "NEWCO" not in moves and "GONE" not in moves  # 0.5% positions: noise


def test_a_starter_position_grown_into_a_real_one_is_new(tmp_path):
    db = str(tmp_path / "s.db")
    filings, holdings = _fake(
        {
            "2026-03-31": {"C_AAPL": 1, "C_FLAT": 1000},  # AAPL 0.1%: a placeholder
            "2026-06-30": {"C_AAPL": 300, "C_FLAT": 1000},
        }
    )
    hf.sync(db, filings=filings, holdings=holdings, lookup=lambda cs: {c: LOOKUP[c] for c in cs})
    (aapl,) = hf.changes(db)["AAPL"]
    assert aapl["action"] == "new" and aapl["shares_change_pct"] is None

    def buy(fund):
        return {
            "fund": fund,
            "action": "new",
            "shares_change_pct": None,
            "weight_pct": 3.0,
            "weight_before_pct": 0.0,
        }

    assert hf.summarize([buy("A")]).startswith("1 buying")
    assert hf.summarize([buy("A"), buy("B")]).startswith("Consensus: 2 funds buying.")
    assert hf.summarize([buy("A"), buy("B"), buy("C")]).startswith(
        "Consensus: 3 funds buying. 3 buying (A new, 3.0% of fund; B new"
    )


def test_consensus_buys_need_two_funds_in_the_same_fresh_quarter(monkeypatch):
    from datetime import date

    def move(fund, period, filed, action="new"):
        return {"fund": fund, "action": action, "period": period, "filed": filed}

    monkeypatch.setattr(
        hf,
        "changes",
        lambda db: {
            "TWO": [move("A", "2026-06-30", "2026-08-14"), move("B", "2026-06-30", "2026-08-10")],
            "ONE": [move("A", "2026-06-30", "2026-08-14")],
            "SPLIT": [move("A", "2026-06-30", "2026-08-14"), move("B", "2026-03-31", "2026-05-15")],
            "STALE": [move("A", "2025-12-31", "2026-02-14"), move("B", "2025-12-31", "2026-02-10")],
            "SOLD": [
                move("A", "2026-06-30", "2026-08-14", "exited"),
                move("B", "2026-06-30", "2026-08-10"),
            ],
        },
    )
    assert hf.consensus_buys("db", today=date(2026, 9, 28)) == ["TWO"]
