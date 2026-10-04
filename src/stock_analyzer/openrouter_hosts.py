"""Guardrails on the OpenRouter hosts the open models run on.

The same model at the same precision is served by a dozen hosts, and they
are not interchangeable: on 2026-09-28 two of them answered "Capital of
France?" as if "France" had been masked to "[ADDRESS]". Three checks keep
a bad host away from the filing reads and the helper roles:

  - an allowlist (openrouter.APPROVED_HOSTS): a host gets traffic only
    once vetted;
  - a known-answer check before each weekly read (`run_canaries`): every
    approved host reads the same short made-up filing, whose facts are
    known, pinned to that host. A host that misses them — or whose quotes
    don't match the text it was sent, the sign of altered input — is
    skipped until it passes again;
  - per-host read quality over the last 30 days (`host_quality`): quote
    match rate and unusable replies from `filing_facts`. A host that
    slips below QUOTE_RATE_FLOOR is skipped too.

`excluded_hosts` combines the last two; `client_from_settings` applies it
to every call. The Claude spot-check (cli/filings.py --spot-check, run by hand) is a
fourth, slower measure: agreement with Claude, by model and host.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy import select, text

from .db.session import exec_sql, get_session
from .db.tables import FilingSpotCheck, OpenRouterHostCheck
from .logging import get_logger
from .openrouter import APPROVED_HOSTS, HOST_NAMES, OpenRouter

logger = get_logger(__name__)

# A failed check keeps a host out until a later one passes; a check older
# than this no longer counts either way (the weekly run renews it).
CHECK_VALID_DAYS = 8
QUALITY_DAYS = 30
QUOTE_RATE_FLOOR = 0.97
MIN_QUOTES = 50  # below this a host's rate is noise
UNUSABLE_CEILING = 0.10
MIN_READS = 20

# A made-up 10-Q with known answers. Names and places sit inside the
# sentences a reader must quote, so masked input shows up as quotes that
# don't match the text.
CANARY_FILING: dict[str, Any] = {
    "ticker": "CANARY",
    "form": "10-Q",
    "filed_on": "2026-08-01",
    "period_end": "2026-06-30",
    "accession": "canary",
    "url": "",
}
CANARY_SECTIONS: dict[str, str] = {
    "mda": (
        "Net revenue for the quarter was $412.6 million, up 31% from $315.0 million a "
        "year ago, driven by utility demand for grid transformers built in Toledo, Ohio. "
        "Revenue growth accelerated from 18% in the prior quarter. "
        "We are raising our full-year revenue outlook to $1.70 billion to $1.75 billion, "
        "from $1.55 billion to $1.60 billion. "
        "Gross margin was 38.2%, compared with 35.9% a year ago. "
        "Backlog was $2.1 billion at quarter end, up from $1.4 billion a year ago. "
        "Chief Financial Officer Maria Delgado and our auditor, Ernst & Young LLP, "
        "identified a material weakness in internal control over financial reporting "
        "related to revenue recognition at our Toledo, Ohio plant. "
        "We repurchased $25.0 million of common stock during the quarter."
    ),
    "risks": "Our revenue depends on the capital spending of a small number of utilities.",
}


def canary_problems(read: Any, host: str) -> list[str]:
    """What a host got wrong on the known-answer filing ([] = passed)."""
    facts = read.facts
    if not isinstance(facts, dict):
        return ["no usable JSON"]
    from .agents.filing_reader import _quotes, filing_prompt, has_placeholder, normalise

    problems = []
    # Strict here, unlike a filing read: every quote exactly as sent, since
    # one masked name inside a long quote would pass the loose match.
    source = normalise(filing_prompt(CANARY_FILING, CANARY_SECTIONS))
    quotes = [q for q in _quotes(facts) if q.strip()]
    exact = sum(normalise(q) in source for q in quotes)
    if not quotes or exact < len(quotes):
        problems.append(f"quotes {exact}/{len(quotes)} match the text sent")
    if has_placeholder(facts):
        problems.append("redaction placeholder in the reply")
    if (facts.get("guidance") or {}).get("direction") != "raised":
        problems.append(f"guidance {(facts.get('guidance') or {}).get('direction')!r}, not raised")
    if (facts.get("demand") or {}).get("direction") != "accelerating":
        problems.append(f"demand {(facts.get('demand') or {}).get('direction')!r}")
    if not any(
        isinstance(c, dict) and c.get("category") == "material_weakness"
        for c in facts.get("caveats") or []
    ):
        problems.append("missed the material weakness")
    served = read.provider
    if served and served != HOST_NAMES.get(host, served):
        problems.append(f"served by {served}, not {host}")
    return problems


def canary(client: OpenRouter, model: str, host: str, *, today: date) -> OpenRouterHostCheck:
    from .agents.filing_reader import read_filing

    try:
        read = read_filing(client, CANARY_FILING, CANARY_SECTIONS, model=model, host=host)
        problems, cost = canary_problems(read, host), read.cost_usd
    except Exception as e:  # noqa: BLE001 — an unreachable host fails its check
        from .usage import BudgetExceededError

        if isinstance(e, BudgetExceededError):
            raise
        problems, cost = [f"call failed: {str(e)[:120]}"], 0.0
    return OpenRouterHostCheck(
        day=today.isoformat(),
        model=model,
        host=host,
        passed=not problems,
        detail="; ".join(problems) or "ok",
        cost_usd=round(cost, 6),
    )


def run_canaries(
    client: OpenRouter, db: str, models: list[str], *, today: date
) -> list[OpenRouterHostCheck]:
    """Check every approved host of `models`, store the results, and drop
    the failures from `client` for the rest of the run."""
    checks = []
    for model in dict.fromkeys(models):
        for host in APPROVED_HOSTS.get(model, []):
            checks.append(canary(client, model, host, today=today))
    with get_session(db) as session:
        for c in checks:
            session.merge(c)
    for c in checks:
        if not c.passed:
            client.excluded.setdefault(c.model, set()).add(c.host)
            logger.warning("Host check failed: %s on %s (%s)", c.model, c.host, c.detail)
    return checks


def failed_checks(db: str, *, today: date) -> dict[str, set[str]]:
    """{model: hosts whose latest check (within CHECK_VALID_DAYS) failed}."""
    since = (today - timedelta(days=CHECK_VALID_DAYS)).isoformat()
    latest: dict[tuple[str, str], tuple[str, bool]] = {}  # (model, host) -> (day, passed)
    with get_session(db) as session:
        for c in session.scalars(
            select(OpenRouterHostCheck).where(OpenRouterHostCheck.day >= since)
        ).all():
            seen = latest.get((c.model, c.host))
            if seen is None or c.day >= seen[0]:
                latest[(c.model, c.host)] = (c.day, bool(c.passed))
    out: dict[str, set[str]] = {}
    for (model, host), (_, passed) in latest.items():
        if not passed:
            out.setdefault(model, set()).add(host)
    return out


def host_quality(db: str, *, today: date, days: int = QUALITY_DAYS) -> list[dict[str, Any]]:
    """Per (model, host) over the last `days`: reads, quote match rate, the
    share of unusable replies, and `problem` when it is below the floor."""
    slug = {name: s for s, name in HOST_NAMES.items()}
    since = (today - timedelta(days=days)).isoformat()
    with get_session(db) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT reader_model, provider, COUNT(*), SUM(quotes_found), SUM(quotes_checked), "
                "SUM(CASE WHEN facts = '' THEN 1 ELSE 0 END) FROM filing_facts "
                "WHERE read_on >= :since AND provider IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 3 DESC"
            ),
            {"since": since},
        ).all()
    out = []
    for model, provider, reads, found, checked, unusable in rows:
        found, checked, unusable = int(found or 0), int(checked or 0), int(unusable or 0)
        rate = found / checked if checked else None
        problem = None
        if rate is not None and checked >= MIN_QUOTES and rate < QUOTE_RATE_FLOOR:
            problem = f"quote match {rate:.1%} < {QUOTE_RATE_FLOOR:.0%}"
        elif reads >= MIN_READS and unusable / reads > UNUSABLE_CEILING:
            problem = f"{unusable}/{reads} unusable replies"
        out.append(
            {
                "model": model,
                "host": slug.get(provider, provider),
                "provider": provider,
                "reads": int(reads),
                "quote_rate": rate,
                "unusable": unusable,
                "problem": problem,
            }
        )
    return out


def excluded_hosts(db: str, *, today: date) -> dict[str, set[str]]:
    """{model: hosts to skip}: a failed latest check, or quality below the
    floor. Never raises — with no data, nothing is excluded."""
    try:
        out = failed_checks(db, today=today)
        for q in host_quality(db, today=today):
            if q["problem"]:
                out.setdefault(q["model"], set()).add(q["host"])
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("Host guardrails unavailable (%s) — allowlist only", e)
        return {}


def spot_check_summary(db: str, *, today: date, days: int = 120) -> list[dict[str, Any]]:
    """Claude agreement per (model, host) from the spot-checks."""
    since = (today - timedelta(days=days)).isoformat()
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    with get_session(db) as session:
        for r in session.scalars(
            select(FilingSpotCheck).where(FilingSpotCheck.checked_on >= since)
        ).all():
            groups.setdefault((r.reader_model, r.provider or "?"), []).append(
                (r.agreed, r.compared)
            )
    return [
        {
            "model": model,
            "provider": provider,
            "filings": len(rs),
            "agreed": sum(a for a, _ in rs),
            "compared": sum(c for _, c in rs),
        }
        for (model, provider), rs in sorted(groups.items())
    ]


def report_lines(db: str, *, today: date) -> list[str]:
    """The host section of the weekly run's summary and the doctor check."""
    lines = []
    for q in host_quality(db, today=today):
        rate = f"{q['quote_rate']:.1%}" if q["quote_rate"] is not None else "n/a"
        flag = f"  <-- {q['problem']}" if q["problem"] else ""
        lines.append(
            f"  {q['model']:<20} {q['provider']:<14} {q['reads']:>4} reads  quotes {rate:>6}  "
            f"unusable {q['unusable']}{flag}"
        )
    for s in spot_check_summary(db, today=today):
        pct = s["agreed"] / s["compared"] if s["compared"] else 0
        lines.append(
            f"  Claude spot-check {s['model']:<20} {s['provider']:<14} {s['filings']} filings, "
            f"{s['agreed']}/{s['compared']} fields agree ({pct:.0%})"
        )
    return lines
