"""What well-known hedge funds bought and sold last quarter, from their 13F filings.

A 13F lists a fund's US stock positions at each quarter end; it is filed up
to 45 days later, so what it shows can be four and a half months old by the
time it is read. The evidence that copying 13Fs beats the market after that
delay is weak, so this is shown for context — which famous funds added to
or cut a stock you hold or are looking at — not scored and not a signal.

FUNDS is a curated list of concentrated, long-term investors (a fund running
thousands of quant positions says nothing about any one of them). Each
night `sync` checks every fund's SEC submissions (one request each) and
downloads a filing's holdings only when it is new, so a quarter costs a few
dozen requests. CUSIPs become tickers through OpenFIGI's free mapping
(cached; the AdGuard rule `@@||openfigi.com^$important` must allow it).
`changes` compares each fund's latest quarter with the one before and keeps
only conviction moves: positions of MIN_FUND_WEIGHT (2%) or more of the
fund's reported portfolio, with each move's weight shown, and `summarize`
flags CONSENSUS_FUNDS (3) or more funds buying the same stock. No LLM.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import date
from typing import Any

from sqlalchemy import text

from ..db.session import exec_sql, get_session
from ..db.tables import CusipTicker, FundPosition
from ..http_client import HttpClient
from ..logging import get_logger

logger = get_logger(__name__)

# CIK -> display name. Resolved and checked against SEC submissions
# 2026-09-27 (each had filed a 13F in 2026). Scion (last filed 2025-11) and
# Greenlight (2024-02) were left out as no longer filing.
FUNDS: dict[str, str] = {
    "0001067983": "Berkshire Hathaway",
    "0001336528": "Pershing Square",
    "0001656456": "Appaloosa",
    "0001061768": "Baupost",
    "0001040273": "Third Point",
    "0001103804": "Viking Global",
    "0001135730": "Coatue",
    "0001061165": "Lone Pine",
    "0001536411": "Duquesne (Druckenmiller)",
    "0001709323": "Himalaya (Li Lu)",
    "0001167483": "Tiger Global",
    "0001747057": "D1 Capital",
    "0001541617": "Altimeter",
    "0000934639": "Maverick",
    "0001581811": "Egerton",
    "0001112520": "Akre",
    "0001868537": "Fundsmith",
    "0001166559": "Gates Foundation Trust",
}
FUND_KEEP_PERIODS = 4
# A change smaller than this either way is a rebalance, not a view.
MIN_CHANGE = 0.10
# Only moves in positions this big a share of the fund's reported portfolio
# are shown: a new position of 0.2% says little, one of 8% is a best idea.
# (The research that holds up after the filing delay is on conviction
# positions; copying every 13F trade does not.) Buys are judged on the new
# weight, sells on the weight before.
MIN_FUND_WEIGHT = 0.02
# A position below this share of the fund last quarter was a placeholder:
# growing it into a real one is a new position, not a "+14,323%" add
# (Fundsmith in TSM, 2026-06-30).
STARTER_WEIGHT = 0.002
# This many tracked funds buying the same stock in one quarter is flagged.
CONSENSUS_FUNDS = 3

_UA = {"User-Agent": "stock-analyzer research-bot (soni.snehal@gmail.com)"}
_SEC = HttpClient(default_headers=_UA, timeout=30.0, rate_limit_per_min=480, name="sec-13f")
_FIGI = HttpClient(timeout=30.0, rate_limit_per_min=20, name="openfigi")  # keyless limit 25/min
_FIGI_BATCH = 10


def latest_filings(cik: str) -> list[dict[str, str]]:
    """Original 13F-HR filings, newest first: {accession, period, filed}."""
    sub = _SEC.get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    r = sub["filings"]["recent"]
    out = []
    for acc, form, period, filed in zip(
        r["accessionNumber"], r["form"], r["reportDate"], r["filingDate"], strict=True
    ):
        if form == "13F-HR" and period:
            out.append({"accession": acc, "period": period, "filed": filed})
    return out


def filing_holdings(cik: str, accession: str) -> dict[str, dict[str, float]]:
    """{cusip: {shares, value_usd}} for the filing's stock positions (options
    and principal-amount lines left out), summed across its lines."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    files = [i["name"] for i in _SEC.get_json(base + "/index.json")["directory"]["item"]]
    table = next(
        (n for n in files if n.lower().endswith(".xml") and "primary_doc" not in n.lower()), None
    )
    if table is None:
        return {}
    root = ET.fromstring(_SEC.get_bytes(f"{base}/{table}"))
    ns = {"n": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda t: f"n:{t}") if ns else (lambda t: t)
    out: dict[str, dict[str, float]] = {}
    for row in root.findall(q("infoTable"), ns):

        def val(path: str, row: ET.Element = row) -> str | None:
            node = row.find(path, ns)
            return node.text.strip() if node is not None and node.text else None

        if val(q("putCall")):
            continue
        if (val(f"{q('shrsOrPrnAmt')}/{q('sshPrnamtType')}") or "SH") != "SH":
            continue
        cusip = (val(q("cusip")) or "").upper()
        shares = float(val(f"{q('shrsOrPrnAmt')}/{q('sshPrnamt')}") or 0)
        value = float(val(q("value")) or 0)
        if not cusip:
            continue
        slot = out.setdefault(cusip, {"shares": 0.0, "value_usd": 0.0})
        slot["shares"] += shares
        slot["value_usd"] += value
    return out


def _figi(cusips: list[str]) -> dict[str, str | None]:
    """OpenFIGI: each CUSIP's US common-stock ticker, or None."""
    out: dict[str, str | None] = {}
    for i in range(0, len(cusips), _FIGI_BATCH):
        chunk = cusips[i : i + _FIGI_BATCH]
        body = [{"idType": "ID_CUSIP", "idValue": c} for c in chunk]
        answers = _FIGI.post_json("https://api.openfigi.com/v3/mapping", json=body)
        for cusip, answer in zip(chunk, answers, strict=True):
            rows = [
                d
                for d in (answer.get("data") or [])
                if d.get("exchCode") == "US" and d.get("ticker")
            ]
            out[cusip] = rows[0]["ticker"].replace("/", "-").upper() if rows else None
    return out


def map_cusips(
    db_path: str, cusips: list[str], *, lookup: Callable[[list[str]], dict[str, str | None]] = _figi
) -> dict[str, str | None]:
    """{cusip: ticker} from the cache, asking OpenFIGI only for new ones."""
    wanted = sorted(set(cusips))
    cached: dict[str, str | None] = {}
    with get_session(db_path) as session:
        for c in wanted:
            row = session.get(CusipTicker, c)
            if row is not None:
                cached[c] = row.ticker
    todo = [c for c in wanted if c not in cached]
    if todo:
        found = lookup(todo)
        today = date.today().isoformat()
        with get_session(db_path) as session:
            for c in todo:
                session.merge(CusipTicker(cusip=c, ticker=found.get(c), checked_on=today))
        cached.update({c: found.get(c) for c in todo})
    return cached


def sync(
    db_path: str,
    *,
    filings: Callable[[str], list[dict[str, str]]] = latest_filings,
    holdings: Callable[[str, str], dict[str, dict[str, float]]] = filing_holdings,
    lookup: Callable[[list[str]], dict[str, str | None]] = _figi,
) -> dict[str, Any]:
    """Store every tracked fund's two latest quarters that aren't stored yet;
    prune older ones. Returns {"funds", "new_filings", "positions"}."""
    new_filings = positions = 0
    for cik, name in FUNDS.items():
        try:
            recent = filings(cik)[:2]
        except Exception as e:  # noqa: BLE001 — one fund's outage is not the night's
            logger.warning("13F: submissions for %s failed (%s)", name, e)
            continue
        with get_session(db_path) as session:
            have = {
                p
                for (p,) in exec_sql(
                    session,
                    text("SELECT DISTINCT period FROM fund_positions WHERE cik = :c"),
                    params={"c": cik},
                ).all()
            }
        for f in recent:
            if f["period"] in have:
                continue
            try:
                rows = holdings(cik, f["accession"])
                tickers = map_cusips(db_path, list(rows), lookup=lookup)
            except Exception as e:  # noqa: BLE001 — not stored, so retried next night
                logger.warning("13F: %s %s skipped (%s)", name, f["period"], e)
                continue
            with get_session(db_path) as session:
                for cusip, h in rows.items():
                    session.merge(
                        FundPosition(
                            cik=cik,
                            period=f["period"],
                            cusip=cusip,
                            ticker=tickers.get(cusip),
                            shares=h["shares"],
                            value_usd=h["value_usd"],
                            filed=f["filed"],
                            accession=f["accession"],
                        )
                    )
            new_filings += 1
            positions += len(rows)
            logger.info("13F: %s %s — %d positions", name, f["period"], len(rows))
            time.sleep(0.2)
        _prune(db_path, cik)
    logger.info("13F sync: %d new filing(s), %d positions", new_filings, positions)
    return {"funds": len(FUNDS), "new_filings": new_filings, "positions": positions}


def _prune(db_path: str, cik: str) -> None:
    with get_session(db_path) as session:
        periods = [
            p
            for (p,) in exec_sql(
                session,
                text(
                    "SELECT DISTINCT period FROM fund_positions WHERE cik = :c ORDER BY period DESC"
                ),
                params={"c": cik},
            ).all()
        ]
        for old in periods[FUND_KEEP_PERIODS:]:
            exec_sql(
                session,
                text("DELETE FROM fund_positions WHERE cik = :c AND period = :p"),
                params={"c": cik, "p": old},
            )


def changes(db_path: str, tickers: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """{ticker: [{fund, action, shares_change_pct, weight_pct, weight_before_pct,
    value_usd, period, filed}]} between each fund's latest two stored
    quarters, conviction moves only. action is "new", "added", "trimmed" or
    "exited"; weights are percent of the fund's reported portfolio. A move
    under MIN_CHANGE, or in a position under MIN_FUND_WEIGHT, is left out."""
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text("SELECT cik, period, ticker, shares, value_usd, filed FROM fund_positions"),
        ).all()
    by_fund: dict[str, dict[str, dict[str, tuple[float, float, str]]]] = {}
    totals: dict[tuple[str, str], float] = {}
    for cik, period, ticker, shares, value, filed in rows:
        # The fund's whole reported portfolio, including what didn't map to
        # a ticker, is the denominator.
        totals[(cik, period)] = totals.get((cik, period), 0.0) + (value or 0.0)
        if ticker is not None:
            slot = by_fund.setdefault(cik, {}).setdefault(period, {})
            s0, v0, _ = slot.get(ticker, (0.0, 0.0, filed))
            slot[ticker] = (s0 + shares, v0 + (value or 0.0), filed)
    wanted = {t.upper() for t in tickers} if tickers else None
    out: dict[str, list[dict[str, Any]]] = {}
    for cik, periods in by_fund.items():
        if len(periods) < 2:
            continue
        latest, prior = sorted(periods, reverse=True)[:2]
        now, before = periods[latest], periods[prior]
        total_now, total_before = totals.get((cik, latest)) or 0.0, totals.get((cik, prior)) or 0.0
        for ticker in set(now) | set(before):
            if wanted is not None and ticker not in wanted:
                continue
            s_now = now.get(ticker, (0.0, 0.0, ""))
            s_before = before.get(ticker, (0.0, 0.0, ""))
            w_now = s_now[1] / total_now if total_now else 0.0
            w_before = s_before[1] / total_before if total_before else 0.0
            if s_before[0] <= 0 or (w_before < STARTER_WEIGHT and s_now[0] > s_before[0]):
                action, pct = "new", None
            elif s_now[0] <= 0:
                action, pct = "exited", -100.0
            else:
                pct = (s_now[0] / s_before[0] - 1) * 100
                if abs(pct) < MIN_CHANGE * 100:
                    continue
                action = "added" if pct > 0 else "trimmed"
            judged = w_now if action in ("new", "added") else w_before
            if judged < MIN_FUND_WEIGHT:
                continue
            out.setdefault(ticker, []).append(
                {
                    "fund": FUNDS.get(cik, cik),
                    "action": action,
                    "shares_change_pct": pct,
                    "weight_pct": w_now * 100,
                    "weight_before_pct": w_before * 100,
                    "value_usd": s_now[1] or s_before[1],
                    "period": latest,
                    "filed": s_now[2] or s_before[2],
                }
            )
    for moves in out.values():
        moves.sort(key=lambda m: -max(m["weight_pct"], m["weight_before_pct"]))
    return out


def summarize(moves: list[dict[str, Any]]) -> str:
    """'Consensus: 3 funds buying. 3 buying (Pershing Square new, 12% of fund;
    ...); 1 selling (Tiger Global sold out, was 4.0%)'. Largest positions first."""
    buy = [m for m in moves if m["action"] in ("new", "added")]
    sell = [m for m in moves if m["action"] in ("trimmed", "exited")]

    def label(m: dict[str, Any]) -> str:
        if m["action"] == "new":
            return f"{m['fund']} new, {m['weight_pct']:.1f}% of fund"
        if m["action"] == "exited":
            return f"{m['fund']} sold out, was {m['weight_before_pct']:.1f}%"
        return f"{m['fund']} {m['shares_change_pct']:+.0f}% to {m['weight_pct']:.1f}% of fund"

    parts = []
    if buy:
        parts.append(f"{len(buy)} buying ({'; '.join(label(m) for m in buy[:4])})")
    if sell:
        parts.append(f"{len(sell)} selling ({'; '.join(label(m) for m in sell[:4])})")
    text = "; ".join(parts)
    if len(buy) >= CONSENSUS_FUNDS:
        text = f"Consensus: {len(buy)} funds buying. " + text
    return text
