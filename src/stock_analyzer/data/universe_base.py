"""The base universe — the sampling frame the screen filters DOWN from.

This distinction is the whole point of the module. A filter can only remove
names the universe already contained, so the sampling frame sets the ceiling
on how good a pick can be. The pipeline used to build its universe purely
from tickers regex-matched out of the last 30 days of insider-buying and
hedge-fund coverage, which meant:

  - a name was only eligible if the press had recently written about it,
    making recency and popularity a prerequisite for every pick, and
  - any capitalized 2-5 letter token that escaped a ~70-word blacklist
    entered the universe as a "ticker", burning a Sonnet analyst call on
    symbols that were never companies.

So the news feeds are now a CONVICTION OVERLAY on top of a real frame
(S&P 500 constituents plus the user's watchlist and holdings), not the
frame itself.

The default frame is a bundled snapshot rather than a live scrape: a
deterministic, offline, testable list beats making every run depend on a
third-party page's markup. Override it with `DISCOVER_UNIVERSE_FILE`
pointing at a newline-delimited ticker list (comments with `#` allowed).

To refresh the bundled snapshot:

    uv run --with lxml python -c "
    import httpx, io, pandas as pd
    r = httpx.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                  timeout=30, headers={'User-Agent': 'stock-analyzer/0.1'})
    df = pd.read_html(io.StringIO(r.text), attrs={'id': 'constituents'})[0]
    print('\\n'.join(sorted({str(s).strip().upper().replace('.', '-')
                             for s in df['Symbol']})))"
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..logging import get_logger

logger = get_logger(__name__)

_STATIC = Path(__file__).parent / "static"
_BUNDLED = _STATIC / "sp500_constituents.txt"
# Every US-listed stock worth $2B+ that passes the tradability filters
# (data/universe_scan.py). The discover frame since 2026-09-27; the S&P 500
# list stays for the model, which was trained and validated on it.
_US_2B = _STATIC / "us_2b_universe.txt"
# Where the nightly rescan (and `ops universe`) writes: outside the repo, so
# it never leaves the working tree dirty (which would stop the 08:30
# auto-update). Preferred over the bundled copy, which predates the
# quality rules (~1,900 names), whenever present.
LOCAL_US_2B = Path(os.path.expanduser("~/.stock_analyzer/us_2b_universe.txt"))
_LISTS = {"sp500": _BUNDLED, "us_2b": _US_2B}


def _us_2b_file() -> Path:
    return LOCAL_US_2B if LOCAL_US_2B.exists() else _US_2B


# Env var that swaps the frame for a user-supplied list.
_OVERRIDE_ENV = "DISCOVER_UNIVERSE_FILE"
# Which bundled list the frame uses when no file overrides it.
_KIND_ENV = "DISCOVER_UNIVERSE"
DEFAULT_KIND = "us_2b"


def _parse_ticker_file(path: Path) -> tuple[str, ...]:
    """One ticker per line; `#` comments and blank lines ignored."""
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip().upper()
        if line:
            out.append(line)
    return tuple(dict.fromkeys(out))  # de-dup, preserve order


def sp500() -> tuple[str, ...]:
    """The bundled S&P 500 list: the model's training universe."""
    return load_base_universe(kind="sp500")


@lru_cache(maxsize=4)
def load_base_universe(path: str | None = None, *, kind: str | None = None) -> tuple[str, ...]:
    """The sampling frame, as an ordered, de-duplicated ticker tuple.

    Resolution order: explicit `path` argument, then `DISCOVER_UNIVERSE_FILE`,
    then the bundled list named by `kind` (or `DISCOVER_UNIVERSE`, default
    "us_2b": every US-listed stock >= $2B that is tradable; "sp500" for the
    S&P 500 snapshot). A missing or unreadable file falls back to the S&P
    500 bundle with a warning rather than failing the run — an empty frame
    would silently reduce the pipeline to its news-derived names, which is
    the behavior this module exists to prevent.
    """
    if kind is None and not path:
        chosen = os.getenv(_KIND_ENV, DEFAULT_KIND).strip().lower() or DEFAULT_KIND
        bundled = _us_2b_file() if chosen == "us_2b" else _LISTS.get(chosen)
        if bundled is not None and bundled != _BUNDLED and not os.getenv(_OVERRIDE_ENV):
            try:
                tickers = _parse_ticker_file(bundled)
                if tickers:
                    logger.info("Base universe: %d tickers (%s, %s)", len(tickers), chosen, bundled)
                    return tickers
            except OSError as e:
                logger.warning("Bundled %s universe unreadable (%s) — using the S&P 500", chosen, e)
    elif kind is not None and kind != "sp500":
        return _parse_ticker_file(_us_2b_file() if kind == "us_2b" else _LISTS[kind])
    candidate = path or os.getenv(_OVERRIDE_ENV) or ""
    if kind == "sp500":
        candidate = ""
    if candidate:
        expanded = Path(os.path.expanduser(candidate))
        try:
            tickers = _parse_ticker_file(expanded)
            if tickers:
                logger.info("Base universe: %d tickers from %s", len(tickers), expanded)
                return tickers
            logger.warning(
                "Base universe file %s is empty — falling back to the bundled S&P 500 snapshot.",
                expanded,
            )
        except OSError as e:
            logger.warning(
                "Could not read base universe file %s (%s) — falling back to "
                "the bundled S&P 500 snapshot.",
                expanded,
                e,
            )
    try:
        tickers = _parse_ticker_file(_BUNDLED)
    except OSError as e:
        logger.error(
            "Bundled base universe missing (%s). The pipeline will run on "
            "watchlist + holdings + news names only, which is a much weaker "
            "sampling frame.",
            e,
        )
        return ()
    logger.info("Base universe: %d tickers (bundled S&P 500 snapshot)", len(tickers))
    return tickers


def refresh_us_2b(path: Path | None = None, *, today=None, rows=None) -> int:
    """Rescan the market and rewrite the >= $2B snapshot (by default the
    local copy, LOCAL_US_2B). Returns the number of tickers written.
    Network: ~2 screener requests."""
    from datetime import date

    from . import universe_scan

    today = today or date.today()
    rows = universe_scan.scan() if rows is None else rows
    kept = universe_scan.investable(rows, today=today)
    tickers = universe_scan.symbols(kept)
    header = [
        "# Every US-listed stock worth $2B+ that passes the quality rules and the",
        "# tradability filters (data/universe_scan.py): quarterly and 12-month revenue",
        "# growth >= 8%, debt/equity <= 2, positive operating and free cash flow,",
        "# return on equity >= 10%; NYSE / Nasdaq / NYSE American common stock,",
        f"# price >= ${universe_scan.MIN_PRICE:.0f} (under $10B), average daily dollar volume >= "
        f"${universe_scan.MIN_DOLLAR_VOLUME / 1e6:.0f}M, a year of trading, no funds,",
        "# shells, SPACs, preferreds, warrants or units. Largest first.",
        f"# Snapshot {today.isoformat()}: {len(tickers)} of {rows.height} scanned. "
        "Refreshed nightly by earnings-watch; by hand: uv run ops universe",
    ]
    path = path or LOCAL_US_2B
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join([*header, *tickers]) + "\n")
    tmp.replace(path)
    load_base_universe.cache_clear()
    return len(tickers)


__all__ = ["load_base_universe", "refresh_us_2b", "sp500"]
