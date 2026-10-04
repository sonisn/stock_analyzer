"""`read-filings` — structured facts from each stock's latest 10-Q/10-K/20-F.

Weekly, Saturday night (scripts/run_filings.sh), over the stocks whose facts
can reach the deciding models — ~500 of the ~1,900 in the $2B+ universe
(`--all` reads every one) — in two tiers:

  A. the stocks acted on — holdings, picks, the discover shortlist and the
     market leaders — read by the reader model (GLM-5.3);
     Their latest earnings 8-K (the press release, where guidance is — a
     10-Q seldom states it) is read too, once the filings are done.
  B. stocks that passed the screen's hard filter in the last ELIGIBLE_DAYS
     (90), read by the bulk model (GLM-5.3-Flash). A bulk read
     that reports a serious event, or fails, is read again on the reader
     model, and that read replaces it — as is one whose filing shows
     operating or net income down 25%+ in its own XBRL figures
     (data/income_drop), which the bulk model can read past.

For each stock: the newest filing from EDGAR (free) — and, for a stock
read for the first time, the one before it too, so its facts can say what
changed — skipped when it is already stored — unless a tier-A stock only has a bulk read, which is
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
from sqlalchemy import text
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
from ..data.income_drop import income_drops
from ..data.sec_edgar import (
    exhibit_sections,
    fetch_filing_text,
    filing_sections,
    latest_filing,
    latest_filings,
)
from ..data.text_change import ANNUAL_FORMS, risk_change, risk_sections
from ..data.universe_base import all_us_2b
from ..db.session import exec_sql, get_session
from ..db.tables import FilingFacts
from ..logging import get_logger
from ..openrouter import OpenRouter, client_from_settings, parse_json_object, spent_today
from ..serialization import dumps_compact
from ..usage import BudgetExceededError, log_usage_summary

logger = get_logger(__name__)

KEEP = {"A": 2, "B": 2}
# A stock counts as screen-eligible when it passed the hard filter in a run
# this recent; only those (and tier A) are read each week.
ELIGIBLE_DAYS = 90
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


def analyzed_recently(db: str, runs: int = 3) -> list[str]:
    """Stocks the Analyst scored in the last `runs` runs, discover or
    rebalance — the shortlist the deciding models actually saw.
    `tracked_tickers` counts discover runs only, so a rebalance-only
    routine left its survivors on the bulk reader (2026-09-27)."""
    with get_session(db) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT DISTINCT ticker FROM scorecards WHERE run_id IN "
                "(SELECT id FROM runs ORDER BY id DESC LIMIT :n)"
            ),
            {"n": runs},
        ).all()
    return [r[0].upper() for r in rows if r[0]]


def passed_recently(db: str, *, days: int = ELIGIBLE_DAYS) -> list[str]:
    """Stocks that passed the screen's hard filter in any run of the last
    `days` — the ones whose filing facts can reach the deciding models.
    ~430 of the ~1,900 in the $2B+ universe (2026-09-28)."""
    with get_session(db) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT DISTINCT ticker FROM candidates WHERE passed_filter = 1 AND run_id IN "
                "(SELECT id FROM runs WHERE run_at >= date('now', :window))"
            ),
            {"window": f"-{days} day"},
        ).all()
    return [r[0].upper() for r in rows if r[0]]


def tier_a(settings: Settings, *, today: date) -> list[str]:
    """Holdings, recent picks, the latest discover survivors, the recent
    Analyst shortlists, earnings standouts and the market leaders."""
    from .ibd import top_leaders

    db = settings.discover_db_path
    leaders = top_leaders(db, settings.discover_ibd_leaders, today=today)
    return list(
        dict.fromkeys([*tracked_tickers(db, today=today), *analyzed_recently(db), *leaders])
    )


def _risk_change(filing: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return risk_change(filing, filing.get("_risks_full") or "")
    except Exception as e:  # noqa: BLE001 — a missing comparison never stops a read
        logger.info("%s: risk-factor comparison failed (%s)", filing.get("ticker"), e)
        return None


def _known_tickers(db: str) -> set[str]:
    with get_session(db) as session:
        return {
            t for (t,) in exec_sql(session, text("SELECT DISTINCT ticker FROM filing_facts")).all()
        }


def _prepare_previous(ticker: str) -> Prepared | None:
    """The filing before the newest, cut and ready to read — for a stock
    read for the first time, so its facts can say what changed."""
    filings = latest_filings(ticker, 2)
    if len(filings) < 2:
        return None
    filing = filings[1]
    text_ = fetch_filing_text(filing["url"])
    sections = filing_sections(text_ or "", filing["form"], max_chars=SECTION_CHARS)
    return (filing, sections) if "mda" in sections else None


def _stored(db: str) -> dict[str, str]:
    """{accession: model that read it}."""
    with get_session(db) as session:
        rows = session.exec(select(FilingFacts.accession, FilingFacts.reader_model)).all()
    return {acc: model for acc, model in rows}


def foreign_aware_sections(filing: dict[str, Any], text: str) -> dict[str, str]:
    """The filing's MD&A and risk factors. A 40-F's are exhibits, so those
    come first; a 20-F or 10-K without them in the main document tries
    its exhibits too (a 20-F filer's annual report is often one)."""
    form = filing["form"]
    if form == "40-F":
        found = exhibit_sections(filing, max_chars=SECTION_CHARS)
        return found or filing_sections(text, form, max_chars=SECTION_CHARS)
    sections = filing_sections(text, form, max_chars=SECTION_CHARS)
    if "mda" not in sections and form != "10-Q":
        sections = {**exhibit_sections(filing, max_chars=SECTION_CHARS), **sections}
    return sections


def _prepare(ticker: str) -> Prepared | str:
    """(filing, sections) ready to read, or why the ticker was skipped."""
    filing = latest_filing(ticker)
    if filing is None:
        return "no 10-Q/10-K/20-F on EDGAR"
    text = fetch_filing_text(filing["url"])
    if not text:
        return "filing text unavailable"
    sections = foreign_aware_sections(filing, text)
    if "mda" not in sections:
        return f"MD&A not found in {filing['form']}"
    if filing["form"] in ANNUAL_FORMS:
        # The whole risk-factor section, for `risk_change` if this one is read.
        filing["_risks_full"] = risk_sections(text, filing["form"])
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
                risk_kept=(f.get("risk_change") or {}).get("kept"),
                risk_cosine=(f.get("risk_change") or {}).get("cosine"),
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
    drops: list[str] = filing.get("income_drops") or []
    if tier == "A" or bulk_model == reader_model:
        return _with_drops(read_filing(client, filing, sections, model=reader_model), drops)
    if filing.get("bulk_stored"):  # --recheck-drops: the bulk read is already in the table
        better = read_filing(client, filing, sections, model=reader_model)
        better.escalated_from = bulk_model
        return _with_drops(better, drops)
    bulk: FilingRead | None
    try:
        bulk = read_filing(client, filing, sections, model=bulk_model, retry=False)
    except BudgetExceededError:
        raise
    except Exception as e:  # noqa: BLE001 — the reader gets its turn
        logger.info("%s: bulk read failed (%s), reading on %s", filing["ticker"], e, reader_model)
        bulk = None
    if bulk is not None and bulk.facts is not None and not bulk.flagged and not drops:
        return bulk
    try:
        better = read_filing(client, filing, sections, model=reader_model)
    except BudgetExceededError:
        if bulk is None:
            raise
        return _with_drops(bulk, drops)  # keep the bulk read; the cap stops the batch next
    better.escalated_from = bulk_model
    if bulk is not None:
        better.cost_usd += bulk.cost_usd
    return _with_drops(better, drops)


def _with_drops(read: FilingRead, drops: list[str]) -> FilingRead:
    """The filed income drops join the read's flags, so the stored row
    says why it was escalated and the screen can score it."""
    read.flag_reasons += [d for d in drops if d not in read.flag_reasons]
    return read


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
    from ..llm import ModelAgent, deterministic_settings

    agent = ModelAgent(
        "FilingReader (Claude)",
        "claude",
        settings.discover_sonnet_model,
        instructions=READER_INSTRUCTIONS,
        settings=deterministic_settings(max_tokens=READER_MAX_TOKENS),
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
    recheck_drops: bool = False,
    host_check: bool = False,
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
    bulk_reads: list[tuple[str, Prepared]] = []  # stored bulk reads, for --recheck-drops
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
            if recheck_drops and have == bulk:
                p[0]["bulk_stored"] = True
                bulk_reads.append((tiers[t], p))
            continue
        stored[acc] = "queued"  # share classes (GOOG/GOOGL) file one document
        todo.append((tiers[t], p))
    # A stock read for the first time also gets the filing before, so its
    # facts can say what changed from one quarter to the next.
    known = _known_tickers(db) if not dry_run else set()
    new = [] if dry_run else [(tier, p) for tier, p in todo if p[0]["ticker"] not in known]
    with ThreadPoolExecutor(_SEC_WORKERS) as ex:
        previous = list(ex.map(lambda tp: _prepare_previous(tp[1][0]["ticker"]), new))
    for (tier, _), prev in zip(new, previous, strict=True):
        if prev is not None and prev[0]["accession"] not in stored:
            stored[prev[0]["accession"]] = "queued"
            todo.append((tier, prev))
    if new:
        print(f"{len(new)} stocks new to the table: their previous filing is read too")
    # A 10-K about to be read is compared with last year's risk factors
    # (two free SEC requests); the full sections are then dropped.
    annual = [p for _, p in todo if p[0].get("_risks_full")]
    with ThreadPoolExecutor(_SEC_WORKERS) as ex:
        changes = list(ex.map(lambda p: _risk_change(p[0]), annual))
    for p, change in zip(annual, changes, strict=True):
        p[0]["risk_change"] = change
    for p in prepared.values():
        if not isinstance(p, str):
            p[0].pop("_risks_full", None)
    # Only the filings about to be read (and, with --recheck-drops, the
    # stored bulk reads): two free SEC requests each.
    with ThreadPoolExecutor(_SEC_WORKERS) as ex:
        for (_, (filing, _)), drops in zip(
            [*todo, *bulk_reads],
            ex.map(lambda tp: income_drops(tp[1][0]), [*todo, *bulk_reads]),
            strict=True,
        ):
            filing["income_drops"] = drops
    rechecked = [tp for tp in bulk_reads if tp[1][0]["income_drops"]]
    if recheck_drops:
        print(
            f"--recheck-drops: {len(rechecked)} of {len(bulk_reads)} stored bulk reads show "
            f"a filed income drop; re-reading them on {reader}"
        )
        todo += rechecked
    n_a = sum(1 for tier, (f, _) in todo if tier == "A" or f.get("bulk_stored"))
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
    print(f"Spent today ${spent_today(db):.3f} of ${settings.openrouter_daily_cap_usd:.2f}")
    if host_check:
        from ..openrouter_hosts import run_canaries

        checks = run_canaries(client, db, [reader, bulk], today=today)
        bad = [f"{c.model} on {c.host}: {c.detail}" for c in checks if not c.passed]
        print(
            f"Host check: {len(checks) - len(bad)}/{len(checks)} passed "
            f"(${sum(c.cost_usd for c in checks):.3f})"
        )
        for line in bad:
            print(f"  skipped this run — {line}")
    print()

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

    releases = 0
    if not stopped and not recheck_drops:
        from ..reporting.filing_alert import earnings_releases

        wanted = [t for t in tickers if tiers[t] == "A"]
        try:
            releases = len(earnings_releases(client, db, wanted, today=today, model=reader))
        except BudgetExceededError as e:
            stopped = str(e)
        print(f"{releases} new earnings releases read (8-K item 2.02, tier A)")

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
    try:
        from ..openrouter_hosts import report_lines

        print("\nHosts, last 30 days:")
        print("\n".join(report_lines(db, today=today)) or "  no reads yet")
    except Exception as e:  # noqa: BLE001 — a report line never fails the run
        logger.warning("Host report failed (%s)", e)
    log_usage_summary()
    return 0


SPOT_CHECK_DAYS = 35


def spot_check(settings: Settings, n: int, *, today: date, seed: int | None = None) -> int:
    """Claude re-reads `n` random open-model reads from the last
    SPOT_CHECK_DAYS and the agreement is stored per field (table
    filing_spot_checks): the running measure of the open readers against
    Claude, by model and host. ~$0.07 a filing on Claude Sonnet."""
    import random

    from ..db.tables import FilingSpotCheck

    db = settings.discover_db_path
    since = date.fromordinal(today.toordinal() - SPOT_CHECK_DAYS).isoformat()
    with get_session(db) as session:
        done = set(session.exec(select(FilingSpotCheck.accession)).all())
        rows = [
            r
            for r in session.exec(
                select(FilingFacts).where(FilingFacts.read_on >= since, FilingFacts.facts != "")
            ).all()
            if r.accession not in done
        ]
        rows = [
            {k: getattr(r, k) for k in FilingFacts.model_fields}  # detached copies
            for r in rows
        ]
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    agreed_total = compared_total = 0
    for row in picked:
        filing = {
            k: row[k] for k in ("ticker", "form", "filed_on", "period_end", "accession", "url")
        }
        text_ = fetch_filing_text(row["url"])
        sections = filing_sections(text_ or "", row["form"], max_chars=SECTION_CHARS)
        if "mda" not in sections:
            print(f"{row['ticker']}: filing no longer cuts cleanly, skipped")
            continue
        try:
            theirs = claude_read(settings, filing, sections)
        except Exception as e:  # noqa: BLE001
            print(f"{row['ticker']}: Claude read failed ({e})")
            continue
        agree = compare(json.loads(row["facts"]), theirs)
        agreed, compared = sum(agree.values()), len(agree)
        agreed_total += agreed
        compared_total += compared
        with get_session(db) as session:
            session.merge(
                FilingSpotCheck(
                    accession=row["accession"],
                    checked_on=today.isoformat(),
                    ticker=row["ticker"],
                    reader_model=row["reader_model"],
                    provider=row["provider"],
                    claude_model=settings.discover_sonnet_model,
                    agreed=agreed,
                    compared=compared,
                    fields=dumps_compact(agree),
                )
            )
        misses = ", ".join(f for f, ok in agree.items() if not ok) or "none"
        print(
            f"{row['ticker']:<6} {row['reader_model']:<20} {row['provider'] or '?':<14} "
            f"{agreed}/{compared} agree (differ: {misses})"
        )
    if compared_total:
        print(
            f"\nSpot-check: {agreed_total}/{compared_total} fields agree "
            f"({agreed_total / compared_total:.0%}) on {len(picked)} filings"
        )
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="the whole $2B+ universe (default: tier A + stocks that passed the screen lately)",
    )
    parser.add_argument("--force", action="store_true", help="re-read filings already stored")
    parser.add_argument("--dry-run", action="store_true", help="fetch and cut, no model calls")
    parser.add_argument(
        "--spot-check",
        type=int,
        default=0,
        metavar="N",
        help="only: have Claude re-read N random open-model reads and store the agreement",
    )
    parser.add_argument(
        "--recheck-drops",
        action="store_true",
        help="re-read stored bulk reads whose filing shows a filed income drop (free SEC check)",
    )
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
    if args.spot_check:
        return spot_check(settings, args.spot_check, today=today)
    if args.tickers:
        tiers = {t.upper(): "A" for t in args.tickers}
    else:
        base = all_us_2b() if args.all else passed_recently(settings.discover_db_path)
        tiers = {t: "B" for t in base}
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
        recheck_drops=args.recheck_drops,
        # The known-answer host check runs before the weekly sweep, not
        # before a hand-picked read of a few tickers.
        host_check=not args.tickers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
