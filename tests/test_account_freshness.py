"""A broker connection that has stopped syncing.

On 2026-09-20 the Schwab HSA had last synced holdings on 2026-07-06: its
share counts, cash and prices were 75 days old while the other two
accounts updated nightly, and nothing in the reports said so.
"""

from __future__ import annotations

from datetime import UTC, datetime

from stock_analyzer.data.brokerage import (
    account_sync_status,
    stale_account_notes,
)

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _account(name, holdings_sync, txn_sync=None, account_id=None):
    return {
        "id": account_id or name,
        "name": name,
        "sync_status": {
            "holdings": {"last_successful_sync": holdings_sync},
            "transactions": {"last_successful_sync": txn_sync},
        },
    }


def test_days_stale_measures_the_last_holdings_sync():
    status = account_sync_status(
        [_account("HSA", "2026-07-06T14:57:27.295621+00:00", "2026-07-05")], now=NOW
    )
    assert round(status["HSA"]["days_stale"]) == 76
    assert status["HSA"]["holdings_synced_at"].date().isoformat() == "2026-07-06"
    # Transaction syncs are plain dates, not timestamps.
    assert status["HSA"]["transactions_synced_at"].date().isoformat() == "2026-07-05"


def test_a_fresh_account_is_not_flagged():
    accounts = [_account("Robinhood Individual", "2026-09-19T23:34:45.694203+00:00")]
    assert stale_account_notes(account_sync_status(accounts, now=NOW)) == []


def test_a_weekend_without_a_sync_is_not_stale():
    # Friday evening's sync, read on Monday morning: three days, no flag.
    accounts = [_account("Traditional IRA", "2026-09-18T20:00:00+00:00")]
    assert stale_account_notes(account_sync_status(accounts, now=NOW)) == []


def test_a_dead_connection_is_named_with_the_date_and_the_fix():
    accounts = [
        _account("HSA Brokerage ...263", "2026-07-06T14:57:27.295621+00:00"),
        _account("Traditional IRA", "2026-09-19T15:43:26.436101+00:00"),
    ]
    notes = stale_account_notes(account_sync_status(accounts, now=NOW))
    assert len(notes) == 1
    assert "HSA Brokerage ...263" in notes[0]
    assert "2026-07-06" in notes[0] and "76 days ago" in notes[0]
    assert "reconnect" in notes[0].lower()


def test_an_unknown_sync_time_is_not_treated_as_stale():
    # No sync_status at all, or a value SnapTrade left null: silence beats
    # telling someone to reconnect a connection that is probably fine.
    assert account_sync_status([{"id": "1", "name": "A"}], now=NOW)["A"]["days_stale"] is None
    assert stale_account_notes(account_sync_status([_account("B", None)], now=NOW)) == []
    assert stale_account_notes(account_sync_status([_account("C", "not a date")], now=NOW)) == []


def test_notes_key_accounts_by_the_same_label_everything_else_uses():
    # Two accounts named "Individual" get the institution appended by
    # account_labels; the freshness note must use that same label.
    accounts = [
        {
            "id": "1",
            "name": "Individual",
            "institution_name": "Robinhood",
            "sync_status": {"holdings": {"last_successful_sync": "2026-05-01T00:00:00+00:00"}},
        },
        {
            "id": "2",
            "name": "Individual",
            "institution_name": "Fidelity",
            "sync_status": {"holdings": {"last_successful_sync": "2026-09-20T00:00:00+00:00"}},
        },
    ]
    notes = stale_account_notes(account_sync_status(accounts, now=NOW))
    assert len(notes) == 1
    assert notes[0].startswith("Individual (Robinhood) last synced 2026-05-01")


# --- the reports have to say it ---------------------------------------------------


STALE_NOTE = (
    "HSA Brokerage ...263 last synced 2026-07-06, 76 days ago — its holdings, "
    "cash and prices are frozen; reconnect it in SnapTrade"
)


def test_the_daily_email_leads_with_a_frozen_account():
    from stock_analyzer.reporting.health import (
        build_portfolio_health,
        decision_items,
        render_decisions_html,
        render_health_html,
    )

    health = build_portfolio_health(
        {"HSA Brokerage ...263": [{"ticker": "BE", "units": 73, "price": 298.61}]},
        stale_accounts=[STALE_NOTE],
    )
    top = decision_items(health)[0]
    assert top["priority"] == 1 and top["label"] == "STALE DATA"
    assert "reconnect" in top["text"].lower()
    assert "2026-07-06" in render_decisions_html(health)
    assert "Stale account data" in render_health_html(health)


def test_a_frozen_account_is_not_graded_as_a_stock_suggestion():
    # The quarterly review grades SELL/REVIEW advice; "reconnect your
    # broker" is not advice about a stock and has no ticker.
    from stock_analyzer.reporting.health import build_portfolio_health, suggestion_rows

    health = build_portfolio_health({}, stale_accounts=[STALE_NOTE])
    assert suggestion_rows(health, today="2026-09-20") == []


def test_the_rebalance_glance_flags_it_before_the_plan():
    from stock_analyzer.discover.rebalance_sections import append_rebalance_glance

    sections = []
    append_rebalance_glance(
        sections,
        rebalance_plan=None,
        thesis_checks=[{"ticker": "OK", "status": "BROKEN", "return_pct": -30.0}],
        harvest_candidates=None,
        stop_loss_warnings=None,
        stale_accounts=[STALE_NOTE],
    )
    lines = [s.text for s in sections if s.kind == "para" and s.text.startswith("•")]
    assert lines[0] == f"• Stale account data — {STALE_NOTE}."
    assert "OK" in lines[1]


# --- symbols no market data exists for ---------------------------------------------


def test_cusips_and_commingled_funds_are_not_analyzable():
    from stock_analyzer.data.brokerage import is_listed_symbol

    # Taronis Technologies (SEC registration revoked) and Taronis Fuels
    # (bankrupt) arrive as CUSIPs; the 401(k) pool as kind "other".
    assert not is_listed_symbol("876214206", "stock")
    assert not is_listed_symbol("87621P209", "stock")
    assert not is_listed_symbol("FGCCPS", "other")
    assert not is_listed_symbol("", None) and not is_listed_symbol(None, None)

    assert is_listed_symbol("BE", "stock")
    assert is_listed_symbol("GOOGL", "stock")
    assert is_listed_symbol("BRK.B", "stock")  # class suffixes are real tickers
    assert is_listed_symbol("SPAXX", None)  # a money-market fund still has a quote
    assert is_listed_symbol("NVDA", "unrecognized_kind")  # a new kind is not a drop


def test_unlisted_holdings_are_split_out_not_dropped():
    from stock_analyzer.data.brokerage import listed_tickers
    from stock_analyzer.reporting.health import aggregate_positions

    holdings = {
        "HSA Brokerage ...263": [{"ticker": "BE", "kind": "stock", "units": 73, "price": 265.63}],
        "Broadcom U.S. 401(k) Plan": [
            {"ticker": "FGCCPS", "kind": "other", "units": 40.244, "price": 111.71}
        ],
        "Individual ...004": [
            {
                "ticker": "876214206",
                "kind": "stock",
                "units": 144,
                "price": 0,
                "average_purchase_price": 12.294171,
            }
        ],
    }
    analyze, skipped = listed_tickers(holdings)
    assert analyze == ["BE"]
    assert skipped == ["876214206", "FGCCPS"]
    # ...but the 401(k) money is still part of the portfolio's value.
    positions = aggregate_positions(holdings)
    assert round(positions["FGCCPS"]["value"], 2) == 4495.66
    assert round(positions["876214206"]["cost"], 2) == 1770.36
