"""Which industry each US stock is in, for the IBD-style group ranking.

IBD ranks 197 industry groups; Yahoo's screener knows 145 industries and
can list every $2B+ US stock in one of them, so a week's map costs about
one screener request per industry (~150, paced) instead of one profile
request per stock (~1,900). Kept for a week in one JSON file beside the
fetch cache; industries barely change.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..logging import get_logger
from . import fetch_cache

logger = get_logger(__name__)

MAX_AGE_DAYS = 7
PAGE = 250
MIN_MARKET_CAP = 2e9


def _path() -> Path | None:
    raw = os.getenv("FETCH_CACHE_DIR", fetch_cache.DEFAULT_DIR).strip()
    if not raw or raw.lower() in {"off", "0", "false", "none"}:
        return None
    return Path(os.path.expanduser(raw)) / "industry_groups.json"


def _yahoo_industries() -> dict[str, list[str]]:
    import yfinance.const as const

    mapping = getattr(const, "SECTOR_INDUSTY_MAPPING", None) or {}
    return {sector: sorted(inds) for sector, inds in mapping.items()}


def _screen_industry(industry: str, offset: int) -> dict[str, Any]:
    import yfinance as yf
    from yfinance import EquityQuery as Q

    region: list[Any] = ["region", "us"]
    cap: list[Any] = ["intradaymarketcap", MIN_MARKET_CAP]
    which: list[Any] = ["industry", industry]
    query = Q("and", [Q("eq", region), Q("gte", cap), Q("eq", which)])
    return yf.screen(query, offset=offset, size=PAGE, sortField="intradaymarketcap", sortAsc=False)


def scan(
    screen: Callable[[str, int], dict[str, Any]] = _screen_industry,
    industries: dict[str, list[str]] | None = None,
    *,
    pause: float = 0.5,
) -> dict[str, dict[str, str]]:
    """{ticker: {"sector", "industry"}} for every $2B+ US stock Yahoo lists."""
    out: dict[str, dict[str, str]] = {}
    failed = 0
    for sector, names in (industries if industries is not None else _yahoo_industries()).items():
        for industry in names:
            offset = 0
            while True:
                try:
                    page = screen(industry, offset) or {}
                except Exception as e:  # noqa: BLE001 — one industry, not the map
                    failed += 1
                    logger.debug("Industry screen failed for %s (%s)", industry, e)
                    break
                quotes = page.get("quotes") or []
                for q in quotes:
                    symbol = str(q.get("symbol") or "").upper().replace(".", "-")
                    if symbol:
                        out.setdefault(symbol, {"sector": sector, "industry": industry})
                offset += PAGE
                if len(quotes) < PAGE or offset >= int(page.get("total") or 0):
                    break
                time.sleep(pause)
            time.sleep(pause)
    logger.info(
        "Industry map: %d stocks in %d industries (%d failed)",
        len(out),
        len({v["industry"] for v in out.values()}),
        failed,
    )
    return out


def industry_map(*, now: float | None = None, rescan: Callable[[], dict] = scan) -> dict[str, str]:
    """{ticker: industry}, rescanned when the stored map is a week old."""
    now = time.time() if now is None else now
    path = _path()
    stored: dict[str, Any] = {}
    if path is not None and path.exists():
        try:
            stored = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            logger.warning("Industry map unreadable (%s) — rescanning", e)
    fresh = float(stored.get("at") or 0) >= now - MAX_AGE_DAYS * 86400
    if not fresh or not stored.get("map"):
        found = rescan()
        # A failed crawl keeps last week's map rather than an empty one.
        if found and len(found) >= len(stored.get("map") or {}) // 2:
            stored = {"at": now, "map": found}
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(stored, separators=(",", ":")))
                os.replace(tmp, path)
        else:
            logger.warning(
                "Industry rescan found only %d stocks — keeping the stored map", len(found)
            )
    return {t: v["industry"] for t, v in (stored.get("map") or {}).items()}


def sector_map() -> dict[str, str]:
    """{ticker: sector} from the stored map (no rescan: industry_map keeps
    it current). Yahoo's sector names, the same ones holdings carry."""
    path = _path()
    if path is None or not path.exists():
        return {}
    try:
        stored = json.loads(path.read_text())
    except OSError, ValueError:
        return {}
    return {t: v["sector"] for t, v in (stored.get("map") or {}).items() if v.get("sector")}
