"""Universe construction — the sampling frame vs the conviction overlay.

The frame decides which names can be picked at all, so these tests pin the
distinction: index/watchlist/holdings grant eligibility without granting
score, news coverage adds conviction without being a precondition, and a
bad SEC map can never empty the frame.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from stock_analyzer.data.universe_base import load_base_universe
from stock_analyzer.discover import universe as uni

_FRAME = ("AAPL", "MSFT", "NVDA")
# The SEC map the real loader returns, keyed by ticker.
_SEC_MAP = {"AAPL": 1, "MSFT": 2, "NVDA": 3, "TSLA": 4, "BRK-B": 5}


def _build(
    *,
    insider: list[dict] | None = None,
    hedge: list[dict] | None = None,
    watchlist: tuple[str, ...] = (),
    holdings: tuple[str, ...] = (),
    sec_map: dict[str, int] | None = None,
    frame: tuple[str, ...] = _FRAME,
):
    with (
        patch.object(uni, "fetch_insider_trades", return_value=insider or []),
        patch.object(uni, "fetch_hedge_fund_trades", return_value=hedge or []),
        patch.object(
            uni,
            "load_ticker_cik_map",
            return_value=_SEC_MAP if sec_map is None else sec_map,
        ),
    ):
        return uni.build_universe(watchlist=watchlist, holdings=holdings, base_universe=frame)


def _item(text: str) -> dict:
    return {"title": text, "snippet": ""}


# --- the sampling frame ---------------------------------------------------


def test_base_universe_is_the_frame_even_with_no_news():
    """The pipeline must have a real candidate pool when the news feeds are
    empty. Previously an empty feed meant an empty universe."""
    out = _build()
    assert set(out) == set(_FRAME)
    assert all(data["in_base_universe"] for data in out.values())
    assert all("index" in data["sources"] for data in out.values())


def test_watchlist_and_holdings_join_the_frame():
    out = _build(watchlist=("TSLA",), holdings=("BRK-B",))
    assert out["TSLA"]["in_base_universe"] is True
    assert "watchlist" in out["TSLA"]["sources"]
    assert out["BRK-B"]["in_base_universe"] is True
    assert "holding" in out["BRK-B"]["sources"]


def test_watchlist_membership_adds_no_conviction():
    """Eligibility, not score. The +5 conviction bonus it used to carry gave
    every watchlist name a structural head start in the ranking."""
    out = _build(watchlist=("TSLA",))
    assert out["TSLA"]["conviction"] == 0


def test_frame_names_are_kept_even_when_the_sec_map_is_empty():
    """A failed SEC fetch must not empty the sampling frame — those names are
    there because an index or the user put them there."""
    out = _build(watchlist=("TSLA",), sec_map={})
    assert set(out) >= set(_FRAME) | {"TSLA"}


# --- the conviction overlay ----------------------------------------------


def test_news_mentions_add_conviction_to_a_frame_name():
    out = _build(insider=[_item("NVDA insider buying accelerates")])
    assert out["NVDA"]["in_base_universe"] is True
    assert "insider" in out["NVDA"]["sources"]
    assert out["NVDA"]["conviction"] >= 1


def test_hedge_fund_mentions_are_weighted_double():
    insider_only = _build(insider=[_item("NVDA bought by insiders")])
    hedge_only = _build(hedge=[_item("NVDA bought by a big fund")])
    assert hedge_only["NVDA"]["conviction"] == 2 * insider_only["NVDA"]["conviction"]


def test_offframe_news_name_is_admitted_but_flagged():
    """A name only the press surfaced can still be analyzed — it just isn't
    part of the frame, which callers can see."""
    out = _build(insider=[_item("TSLA insider buying")])
    assert "TSLA" in out
    assert out["TSLA"]["in_base_universe"] is False
    assert out["TSLA"]["sources"] == ["insider"]


def test_offframe_garbage_is_dropped_by_sec_validation():
    """Regex noise that isn't a real listing must not reach yfinance."""
    out = _build(insider=[_item("THE CEO SAID GROWTH WAS STRONG")])
    assert set(out) == set(_FRAME)


def test_repeat_mentions_do_not_duplicate_the_source_label():
    out = _build(
        insider=[_item("NVDA buying"), _item("NVDA more buying")],
    )
    assert out["NVDA"]["sources"].count("insider") == 1
    assert out["NVDA"]["conviction"] == 2


# --- base universe loader -------------------------------------------------


def test_bundled_snapshot_loads_and_excludes_comments():
    tickers = load_base_universe()
    assert len(tickers) > 400
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    assert not any(t.startswith("#") for t in tickers)
    assert all(t == t.upper() for t in tickers)


def test_override_file_replaces_the_frame():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mine.txt")
        with open(path, "w") as fh:
            fh.write("# my own frame\nFOO\nbar\n\nBAZ # trailing comment\n")
        tickers = load_base_universe(path)
    assert tickers == ("FOO", "BAR", "BAZ")


def test_unreadable_override_falls_back_to_the_bundle():
    """An empty frame would silently reduce the pipeline to news-only names,
    which is the behavior the frame exists to prevent."""
    tickers = load_base_universe("/nonexistent/path/to/universe.txt")
    assert "AAPL" in tickers


def test_override_dedups_while_preserving_order():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dupes.txt")
        with open(path, "w") as fh:
            fh.write("AAA\nBBB\nAAA\nCCC\n")
        assert load_base_universe(path) == ("AAA", "BBB", "CCC")


@pytest.mark.parametrize("text", ["", "# only comments\n", "   \n\n"])
def test_empty_override_falls_back_to_the_bundle(text):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "empty.txt")
        with open(path, "w") as fh:
            fh.write(text)
        assert "AAPL" in load_base_universe(path)
