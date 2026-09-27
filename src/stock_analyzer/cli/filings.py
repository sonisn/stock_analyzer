"""`read-filings` — structured facts from each stock's latest 10-Q/10-K/20-F.

Weekly, Saturday night (scripts/run_filings.sh), over the whole $2B+
universe in two tiers:

  A. the stocks acted on — holdings, picks, the discover shortlist and the
     market leaders — read by the reader model (GLM-5.3);
  B. everything else, read by the bulk model (GLM-5.3-Flash). A bulk read
     that reports a serious event, or fails, is read again on the reader
     model, and that read replaces it.

For each stock: the newest filing from EDGAR (free), skipped when it is
already stored — unless a tier-A stock only has a bulk read, which is
re-read on the reader model (a stock becoming a pick gets the better read
on the next run). Results land in `filing_facts` as each read finishes.

Every OpenRouter call counts against OPENROUTER_DAILY_CAP_USD. When the
cap is reached the rest waits for the next run. The weekly run sets a $5
cap; a peak earnings week is ~300-400 filings (~$1-2).

`--compare-claude N` also has Claude read N of the filings with the same
instructions and prints a field-by-field agreement table (spends Claude
credit). `--dry-run` fetches and cuts the filings without calling a model.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlmodel import col, select

from ..agents.filing_reader import (
    READER_INSTRUCTIONS,
    READER_MAX_TOKENS,
    SECTION_CHARS,
    FilingRead,
    filing_prompt,
    read_filing,
)
from ..config import Settings
from ..data.forecast_snapshots import tracked_tickers
from ..data.sec_edgar import fetch_filing_text, filing_sections, latest_filing
from ..data.universe_base import all_us_2b
from ..db.session import get_session
from ..db.tables import FilingFacts
from ..logging import get_logger
from ..openrouter import OpenRouter, client_from_settings, parse_json_object, spent_today
from ..serialization import dumps_compact
from ..usage import BudgetExceededError, log_usage_summary

logger = get_logger(__name__)

KEEP = {"A": 2, "B": 1}
_SEC_WORKERS = 4  # the SEC client rate-limits itself to 8 requests a second
_READ_WORKERS = 8
# Fields compared between a reader and Claude.
COMPARE_FIELDS = (
    "guidance.direction",
    "demand.direction",
    "margins.direction",
    "backlog.change",
    "liquidity.concern",
    "tone",
)

Prepared = tuple[dict[str, Any], dict[str, str]]


def tier_a(settings: Settings, *, today: date) -> list[str]:
    """Holdings, recent picks, the latest discover survivors, earnings
    standouts and the market leaders fed to discover."""
    from .ibd import top_leaders

    db = settings.discover_db_path
    leaders = top_leaders(db, settings.discover_ibd_leaders, today=today)
    return list(dict.fromkeys([*tracked_tickers(db, today=today), *leaders]))


def _stored(db: str) -> dict[str, str]:
    """{accession: model that read it}."""
    with get_session(db) as session:
        rows = session.exec(select(FilingFacts.accession, FilingFacts.reader_model)).all()
    return {acc: model for acc, model in rows}


def _prepare(ticker: str) -> Prepared | str:
    """(filing, sections) ready to read, or why the ticker was skipped."""
    filing = latest_filing(ticker)
    if filing is None:
        return "no 10-Q/10-K/20-F on EDGAR"
    text = fetch_filing_text(filing["url"])
    if not text:
        return "filing text unavailable"
    sections = filing_sections(text, filing["form"], max_chars=SECTION_CHARS)
    if "mda" not in sections:
        return f"MD&A not found in {filing['form']}"
    return filing, sections


def store(db: str, read: FilingRead, *, tier: str, today: date) -> None:
    f = read.filing
    with get_session(db) as session:
        session.merge(
            FilingFacts(
                accession=f["accession"],
                ticker=f["ticker"],
                tier=tier,
                form=f["form"],
                filed_on=f["filed_on"],
                period_end=f.get("period_end"),
                url=f["url"],
                read_on=today.isoformat(),
                reader_model=read.reader_model,
                provider=read.provider,
                facts=dumps_compact(read.facts) if read.facts is not None else "",
                quotes_checked=read.quotes_checked,
                quotes_found=read.quotes_found,
                flagged=read.flagged,
                flag_reasons="; ".join(read.flag_reasons),
                escalated_from=read.escalated_from,
                cost_usd=round(read.cost_usd, 6),
            )
        )
        session.flush()
        rows = session.exec(
            select(FilingFacts)
            .where(FilingFacts.ticker == f["ticker"])
            .order_by(col(FilingFacts.filed_on).desc())
        ).all()
        for old in rows[KEEP.get(tier, 1) :]:
            session.delete(old)


def read_tiered(
    client: OpenRouter,
    item: Prepared,
    *,
    tier: str,
    reader_model: str,
    bulk_model: str,
) -> FilingRead:
    """Tier A on the reader. Tier B on the bulk model, read again on the
    reader when the bulk read is flagged, comes back empty, or fails —
    about 1 bulk read in 120 thinks past its output room, and a second
    try on the reader costs a cent where a retry on the bulk model often
    can't turn its thinking off (2026-09-27 sweep)."""
    filing, sections = item
    if tier == "A" or bulk_model == reader_model:
        return read_filing(client, filing, sections, model=reader_model)
    bulk: FilingRead | None
    try:
        bulk = read_filing(client, filing, sections, model=bulk_model, retry=False)
    except BudgetExceededError:
        raise
    except Exception as e:  # noqa: BLE001 — the reader gets its turn
        logger.info("%s: bulk read failed (%s), reading on %s", filing["ticker"], e, reader_model)
        bulk = None
    if bulk is not None and bulk.facts is not None and not bulk.flagged:
        return bulk
    try:
        better = read_filing(client, filing, sections, model=reader_model)
    except BudgetExceededError:
        if bulk is None:
            raise
        return bulk  # keep the bulk read; the cap stops the batch next
    better.escalated_from = bulk_model
    if bulk is not None:
        better.cost_usd += bulk.cost_usd
    return better


def _get(d: dict[str, Any] | None, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def compare(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, bool]:
    """Field-by-field agreement of two readings of the same filing."""
    out = {f: _get(a, f) == _get(b, f) for f in COMPARE_FIELDS}

    def high(x: dict[str, Any] | None) -> bool:
        return any(
            isinstance(c, dict) and c.get("severity") == "high"
            for c in (x or {}).get("caveats") or []
        )

    out["high_caveat"] = high(a) == high(b)
    return out


def claude_read(settings: Settings, filing: dict[str, Any], sections: dict[str, str]) -> Any:
    from ..llm import AgnoAgent, deterministic_model_kwargs

    agent = AgnoAgent(
        "FilingReader (Claude)",
        "claude",
        settings.discover_sonnet_model,
        instructions=READER_INSTRUCTIONS,
        model_kwargs={**deterministic_model_kwargs("claude"), "max_tokens": READER_MAX_TOKENS},
    )
    return parse_json_object(agent.run(filing_prompt(filing, sections)).content or "")


def _line(read: FilingRead, tier: str) -> str:
    f, facts = read.filing, read.facts or {}
    esc = f"  (escalated from {read.escalated_from})" if read.escalated_from else ""
    return (
        f"{tier} {f['ticker']:<6} {f['form']:<5} {f.get('period_end') or '':<10} "
        f"guidance={_get(facts, 'guidance.direction') or '?':<10} "
        f"tone={facts.get('tone') or '?':<8} caveats={len(facts.get('caveats') or [])} "
        f"quotes={read.quotes_found}/{read.quotes_checked} "
        f"{'FLAGGED' if read.flagged else 'ok':<7} ${read.cost_usd:.4f}{esc}"
    )


def run(
    settings: Settings,
    tiers: dict[str, str],
    *,
    today: date,
    force: bool = False,
    dry_run: bool = False,
    compare_claude: int = 0,
) -> int:
    """Read the latest filing of every ticker in `tiers` ({ticker: "A"|"B"})."""
    db = settings.discover_db_path
    client = None if dry_run else client_from_settings(settings)
    if not dry_run and client is None:
        print("OPENROUTER_API_KEY is not set.")
        return 1
    reader, bulk = settings.openrouter_reader_model, settings.openrouter_bulk_model

    tickers = sorted(tiers, key=lambda t: tiers[t])  # tier A first
    with ThreadPoolExecutor(_SEC_WORKERS) as ex:
        prepared = dict(zip(tickers, ex.map(_prepare, tickers), strict=True))
    stored = {} if force else _stored(db)
    todo: list[tuple[str, Prepared]] = []
    skipped = up_to_date = 0
    for t in tickers:
        p = prepared[t]
        if isinstance(p, str):
            skipped += 1
            logger.info("%s skipped: %s", t, p)
            continue
        acc = p[0]["accession"]
        have = stored.get(acc)
        # A tier-A stock with only a bulk read is promoted to the reader.
        if have and (tiers[t] == "B" or have == reader or have == "queued"):
            up_to_date += 1
            continue
        stored[acc] = "queued"  # share classes (GOOG/GOOGL) file one document
        todo.append((tiers[t], p))
    n_a = sum(1 for tier, _ in todo if tier == "A")
    print(
        f"{len(tiers)} stocks: {up_to_date} up to date, {skipped} without a readable filing, "
        f"{len(todo)} to read ({n_a} on {reader}, {len(todo) - n_a} on {bulk})"
    )
    if dry_run:
        for tier, (filing, sections) in todo:
            sizes = ", ".join(f"{k} {len(v):,}" for k, v in sections.items())
            print(
                f"{tier} {filing['ticker']:<6} {filing['form']} filed {filing['filed_on']}: {sizes}"
            )
        return 0
    assert client is not None
    print(f"Spent today ${spent_today(db):.3f} of ${settings.openrouter_daily_cap_usd:.2f}\n")

    reads: list[tuple[str, FilingRead, Prepared]] = []
    stopped: str | None = None
    queue = list(todo)
    with ThreadPoolExecutor(_READ_WORKERS) as ex:
        running: dict[Future[FilingRead], tuple[str, Prepared]] = {}
        while queue or running:
            while queue and not stopped and len(running) < _READ_WORKERS:
                tier, item = queue.pop(0)
                fut = ex.submit(
                    read_tiered, client, item, tier=tier, reader_model=reader, bulk_model=bulk
                )
                running[fut] = (tier, item)
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                tier, item = running.pop(fut)
                try:
                    r = fut.result()
                except BudgetExceededError as e:
                    stopped = stopped or str(e)
                    queue.insert(0, (tier, item))
                    continue
                except Exception as e:  # noqa: BLE001 — one bad filing never sinks the batch
                    logger.warning("Filing read failed for %s (%s)", item[0]["ticker"], e)
                    print(f"{tier} {item[0]['ticker']:<6} read failed: {e}")
                    continue
                store(db, r, tier=tier, today=today)
                reads.append((tier, r, item))
                if tier == "A" or r.flagged or r.escalated_from:
                    print(_line(r, tier))
                    for reason in r.flag_reasons:
                        print(f"           flag: {reason}")

    if compare_claude and reads:
        _compare_with_claude(settings, reads[:compare_claude], today=today)

    total = sum(r.cost_usd for _, r, _ in reads)
    escalated = sum(1 for _, r, _ in reads if r.escalated_from)
    print(
        f"\n{len(reads)} read ({escalated} escalated to {reader}), ${total:.3f}; "
        f"today ${spent_today(db):.3f} of ${settings.openrouter_daily_cap_usd:.2f}"
    )
    if stopped:
        print(f"Stopped at the daily cap with {len(queue)} left for the next run: {stopped}")
    log_usage_summary()
    return 0


def _compare_with_claude(
    settings: Settings, reads: list[tuple[str, FilingRead, Prepared]], *, today: date
) -> None:
    rows: list[dict[str, Any]] = []
    agreements: list[dict[str, bool]] = []
    for _, r, (filing, sections) in reads:
        try:
            theirs = claude_read(settings, filing, sections)
        except Exception as e:  # noqa: BLE001
            print(f"{filing['ticker']}: Claude read failed ({e})")
            continue
        agreements.append(compare(r.facts, theirs))
        rows.append(
            {
                "ticker": filing["ticker"],
                "reader_model": r.reader_model,
                "agree": agreements[-1],
                "reader": r.facts,
                "claude": theirs,
            }
        )
    if not rows:
        return
    print(f"\nReader vs Claude ({settings.discover_sonnet_model}) on {len(rows)} filings:")
    for fld in (*COMPARE_FIELDS, "high_caveat"):
        hits = sum(a[fld] for a in agreements)
        print(f"  {fld:<20} {hits}/{len(rows)}")
    out = Path(settings.dashboard_path).expanduser().parent / f"filing_compare_{today}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"  details: {out}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="read-filings", description=__doc__.split("\n")[0])
    parser.add_argument(
        "tickers", nargs="*", help="read just these, on the reader model (default: the universe)"
    )
    parser.add_argument("--limit", type=int, default=0, help="read at most this many stocks")
    parser.add_argument("--force", action="store_true", help="re-read filings already stored")
    parser.add_argument("--dry-run", action="store_true", help="fetch and cut, no model calls")
    parser.add_argument(
        "--compare-claude",
        type=int,
        default=0,
        metavar="N",
        help="also read N filings on Claude and compare (spends Claude credit)",
    )
    args = parser.parse_args()
    settings = Settings()
    today = date.today()
    if args.tickers:
        tiers = {t.upper(): "A" for t in args.tickers}
    else:
        tiers = {t: "B" for t in all_us_2b()}
        tiers.update({t: "A" for t in tier_a(settings, today=today)})
    if args.limit:
        tiers = dict(sorted(tiers.items(), key=lambda kv: kv[1])[: args.limit])
    return run(
        settings,
        tiers,
        today=today,
        force=args.force,
        dry_run=args.dry_run,
        compare_claude=args.compare_claude,
    )


if __name__ == "__main__":
    raise SystemExit(main())
