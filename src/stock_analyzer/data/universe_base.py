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

_BUNDLED = Path(__file__).parent / "static" / "sp500_constituents.txt"
# Env var that swaps the frame for a user-supplied list.
_OVERRIDE_ENV = "DISCOVER_UNIVERSE_FILE"


def _parse_ticker_file(path: Path) -> tuple[str, ...]:
    """One ticker per line; `#` comments and blank lines ignored."""
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip().upper()
        if line:
            out.append(line)
    return tuple(dict.fromkeys(out))  # de-dup, preserve order


@lru_cache(maxsize=2)
def load_base_universe(path: str | None = None) -> tuple[str, ...]:
    """The sampling frame, as an ordered, de-duplicated ticker tuple.

    Resolution order: explicit `path` argument, then `DISCOVER_UNIVERSE_FILE`,
    then the bundled S&P 500 snapshot. A missing or unreadable override falls
    back to the bundle with a warning rather than failing the run — an empty
    frame would silently reduce the pipeline to its news-derived names, which
    is the behavior this module exists to prevent.
    """
    candidate = path or os.getenv(_OVERRIDE_ENV) or ""
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


__all__ = ["load_base_universe"]
