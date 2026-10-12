"""`moonshot` — the moonshot basket (discover/moonshot.py): MOONSHOT_TARGET_PCT
of the portfolio over MOONSHOT_SLOTS equal small-company bets, held ten years.

    uv run moonshot screen            # candidates for the open slots (December)
    uv run moonshot add ACME          # you bought it: it joins the basket
    uv run moonshot remove ACME --reason "revenue down two years"
    uv run moonshot list              # members, value, open slots
    uv run moonshot check             # members' filings against the sell rules

Free data only: Yahoo's screener and bar store, and the SEC's company facts
and submissions. No LLM calls. The screen proposes; a stock joins the
basket only when you add it.
"""

from __future__ import annotations

import argparse
import math
from datetime import date, timedelta

from dotenv import load_dotenv

from ..config import Settings
from ..logging import get_logger

logger = get_logger(__name__)


def _pct(x: float | None) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:+.0%}"


def _book_value(db_path: str) -> float | None:
    """The latest recorded portfolio total (the daily snapshot), for slot sizes."""
    from ..db.repository import fetch_snapshots
    from ..db.session import get_session

    start = (date.today() - timedelta(days=14)).isoformat()
    with get_session(db_path) as session:
        rows = fetch_snapshots(session, start=start)
        return rows[-1].total if rows else None


def _universe(today: date) -> list[tuple[str, str, float]]:
    """(ticker, name, market cap) of every listed common stock in the
    basket's size band that trades $2M+ a day and has a year of history."""
    import polars as pl

    from ..data import universe_scan as us
    from ..discover import moonshot as ms

    rows = us.scan(ms.MIN_MARKET_CAP, quality=False)
    rows = rows.filter(
        pl.col("exchange").is_in(sorted(us.LISTED_EXCHANGES))
        & (pl.col("quote_type") == "EQUITY")
        & (pl.col("market_cap") <= ms.MAX_MARKET_CAP)
        & (pl.col("avg_volume") * pl.col("price") >= ms.MIN_DOLLAR_VOLUME)
        & (pl.col("first_trade") <= today - timedelta(days=us.MIN_HISTORY_DAYS))
    )
    out = []
    for sym, name, cap in zip(
        rows["symbol"].to_list(), rows["name"].to_list(), rows["market_cap"].to_list(), strict=True
    ):
        if us._NOT_COMMON.search(name or "") or us._SUFFIX.search(sym or ""):
            continue
        out.append((sym.upper().replace(".", "-"), name or "", float(cap)))
    return out


def screen(settings: Settings, today: date) -> str:
    from ..data.sec_edgar import company_sic, load_ticker_cik_map
    from ..discover import moonshot as ms

    members = ms.load_members(settings.discover_db_path)
    held = {m.ticker for m in members}
    universe = [u for u in _universe(today) if u[0] not in held]
    logger.info("Moonshot screen: %d listed stocks in the $150M-$2B band", len(universe))
    caps = {t: cap for t, _, cap in universe}
    names = {t: n for t, n, _ in universe}
    snaps = ms.live_snapshots(list(caps), cache_dir=settings.model_cache_dir, today=today)
    passed = [
        t
        for t, snap in snaps.items()
        # The screener's own volume filter already ran; dollar volume passes.
        if not ms.entry_reasons(snap, market_cap=caps[t], dollar_volume=ms.MIN_DOLLAR_VOLUME)
    ]
    ciks = load_ticker_cik_map()

    def candidate(t: str, cap: float, snap: ms.Snapshot | None) -> ms.Candidate | None:
        cik = ciks.get(t) or ciks.get(t.replace("-", "."))
        sic, industry = company_sic(cik) if cik else (None, "")
        if snap is None:
            return None
        return ms.Candidate(t, names.get(t, ""), cap, snap, industry, sic)

    candidates = [c for t in passed if (c := candidate(t, caps[t], snaps[t]))]
    held_snaps = ms.live_snapshots(sorted(held), cache_dir=settings.model_cache_dir, today=today)
    held_c = [c for t in held if (c := candidate(t, 0.0, held_snaps.get(t)))]
    open_slots = max(0, settings.moonshot_slots - len(members))
    chosen = ms.fill_slots(
        candidates, open_slots=open_slots, slots=settings.moonshot_slots, held=held_c
    )
    book = _book_value(settings.discover_db_path)
    slot = (
        settings.moonshot_target_pct / 100 * book / settings.moonshot_slots
        if book and settings.moonshot_slots
        else None
    )
    lines = [
        f"Moonshot screen {today}: {len(universe)} listed stocks in the band, "
        f"{len(snaps)} with SEC facts, {len(passed)} pass every entry rule.",
        f"Basket: {len(members)} of {settings.moonshot_slots} slots filled; "
        f"{open_slots} open" + (f" at ~${slot:,.0f} each (new cash only)." if slot else "."),
        "",
        "Proposed for the open slots (smallest first, within 3 per industry and a "
        f"{ms.MAX_BIOTECH_SHARE:.0%} biotech cap):",
    ]
    header = (
        f"  {'ticker':7} {'cap':>7} {'rev g':>6} {'op m':>6} {'GM':>5} {'GM chg':>7} "
        f"{'shares':>7} {'runway':>7}  industry / name"
    )

    def row(c: ms.Candidate) -> str:
        s = c.snap
        runway = s.runway_years
        rw = "cash+" if runway == math.inf else (f"{runway:5.1f}y" if runway is not None else "—")
        return (
            f"  {c.ticker:7} {c.market_cap / 1e6:6.0f}M {_pct(s.revenue_growth):>6} "
            f"{_pct(s.operating_margin):>6} {_pct(s.gross_margin):>5} "
            f"{_pct(s.gross_margin_change):>7} {_pct(s.dilution):>7} {rw:>7}  "
            f"{c.industry or '?'} / {c.name}"
        )

    lines += [header, *(row(c) for c in chosen)] if chosen else ["  (none)"]
    rest = [c for c in candidates if c not in chosen]
    if rest:
        lines += ["", "Also passed (spread rules or no open slot):", header]
        lines += [row(c) for c in sorted(rest, key=lambda c: c.market_cap)]
    lines += [
        "",
        "These are the rules' output, not a forecast: in the 2012-2020 study no rule "
        "picked 20x winners. The point is twenty equal bets held ten years.",
        "Buy with new cash, then: uv run moonshot add TICKER",
    ]
    return "\n".join(lines)


def check(settings: Settings, today: date) -> str:
    from ..discover import moonshot as ms

    members = ms.load_members(settings.discover_db_path)
    if not members:
        return "The basket is empty."
    breaks = ms.member_breaks(
        members,
        db_path=settings.discover_db_path,
        cache_dir=settings.model_cache_dir,
        today=today,
    )
    return "\n".join(
        f"  {t:7} " + ("SELL — " + "; ".join(why) if why else "intact: hold")
        for t, why in sorted(breaks.items())
    )


def list_members(settings: Settings) -> str:
    from ..discover import moonshot as ms

    members = ms.load_members(settings.discover_db_path)
    book = _book_value(settings.discover_db_path)
    head = (
        f"Moonshot basket: {len(members)} of {settings.moonshot_slots} slots, target "
        f"{settings.moonshot_target_pct:g}% of the portfolio"
        + (f" (~${book * settings.moonshot_target_pct / 100:,.0f})" if book else "")
    )
    rows = [
        f"  {m.ticker:7} added {m.added_on}"
        + (f" at ${m.entry_price:,.2f}" if m.entry_price else "")
        for m in sorted(members, key=lambda m: m.added_on)
    ]
    return "\n".join([head, *rows])


def _last_close(ticker: str, today: date) -> float | None:
    from ..data import yf_gateway

    bars = yf_gateway.daily_bars_many([ticker], start=today - timedelta(days=10), what="moonshot")
    frame = bars.get(ticker)
    if frame is None or frame.is_empty():
        return None
    return float(frame["Close"].drop_nans().drop_nulls()[-1])


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="moonshot", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("screen", help="candidates for the open slots")
    sub.add_parser("list", help="the members")
    sub.add_parser("check", help="members' filings against the sell rules")
    add = sub.add_parser("add", help="a stock you bought joins the basket")
    add.add_argument("ticker")
    add.add_argument("--price", type=float, help="your entry price (default: last close)")
    rm = sub.add_parser("remove", help="a member leaves the basket")
    rm.add_argument("ticker")
    rm.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    settings = Settings()
    today = date.today()

    from ..discover import moonshot as ms

    if args.cmd == "screen":
        print(screen(settings, today))
    elif args.cmd == "list":
        print(list_members(settings))
    elif args.cmd == "check":
        print(check(settings, today))
    elif args.cmd == "add":
        ticker = args.ticker.upper()
        members = ms.load_members(settings.discover_db_path)
        if ticker not in {m.ticker for m in members} and len(members) >= settings.moonshot_slots:
            parser.error(f"all {settings.moonshot_slots} slots are filled")
        price = args.price if args.price is not None else _last_close(ticker, today)
        ms.add_member(settings.discover_db_path, ticker, today=today, entry_price=price)
        print(f"{ticker} joined the basket" + (f" at ${price:,.2f}." if price else "."))
    elif args.cmd == "remove":
        if ms.remove_member(
            settings.discover_db_path, args.ticker, today=today, reason=args.reason
        ):
            print(f"{args.ticker.upper()} left the basket: {args.reason}")
        else:
            parser.error(f"{args.ticker.upper()} is not in the basket")


if __name__ == "__main__":
    main()
