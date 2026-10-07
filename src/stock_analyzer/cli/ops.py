"""`ops` — keeping the scheduled jobs honest. No LLM tokens are spent.

  ops alert JOB LOG STATUS   email the tail of a failed job's log
  ops backup                 copy the database, keeping the newest BACKUP_KEEP
                             (and upload it to BACKUP_REMOTE when set),
                             and delete logs older than LOG_KEEP_DAYS
  ops restore [NAME]         download and check an off-site backup (the newest
                             by default); --apply swaps it in, saving the current
                             database first
  ops doctor                 check every key, model id and data source for free,
                             and free space on the disks the data lives on
  ops universe               rescan US stocks >= $2B and rewrite the discover
                             universe (~/.stock_analyzer/us_2b_universe.txt,
                             preferred over the bundled copy)

`scripts/run_job.sh` calls `alert` when a cron job exits non-zero; before
it existed a failed job was silent until someone noticed an email missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from ..config import Settings
from ..logging import get_logger

logger = get_logger(__name__)

ALERT_TAIL_LINES = 80


# --- alert --------------------------------------------------------------------


def _tail(path: str, lines: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except OSError as e:
        return f"(could not read {path}: {e})"


def alert(settings: Settings, job: str, log_path: str, status: int) -> None:
    from ..reporting.smtp import SmtpServer

    body = (
        f"The scheduled job '{job}' exited with status {status} at "
        f"{datetime.now():%Y-%m-%d %H:%M %Z}.\n\n"
        f"Log: {log_path}\n\nLast {ALERT_TAIL_LINES} lines:\n\n"
        f"{_tail(log_path, ALERT_TAIL_LINES)}"
    )
    if not settings.email_to:
        print(body)
        return
    SmtpServer().send_email(settings.email_to, f"stock-analyzer: {job} FAILED", body)


# --- backup -------------------------------------------------------------------


def backup(db_path: str, backup_dir: str, keep: int, *, now: datetime | None = None) -> Path:
    """A consistent copy through SQLite's backup API (safe while the WAL is
    live, unlike a file copy), then the oldest copies past `keep` go."""
    src = Path(os.path.expanduser(db_path))
    if not src.exists():
        raise FileNotFoundError(f"database not found: {src}")
    out_dir = Path(os.path.expanduser(backup_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o700)
    dest = out_dir / f"{src.stem}-{(now or datetime.now()):%Y%m%d-%H%M%S}.db"
    with sqlite3.connect(src) as s, sqlite3.connect(dest) as d:
        s.backup(d)
    dest.chmod(0o600)
    for old in sorted(out_dir.glob(f"{src.stem}-*.db"))[:-keep] if keep > 0 else []:
        old.unlink()
    logger.info("Backup: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


Run = Callable[..., subprocess.CompletedProcess]


def upload_offsite(dest: Path, remote: str, keep_days: int, *, run: Run = subprocess.run) -> None:
    """Copy one backup to the rclone `remote` and delete copies there older
    than `keep_days`. The remote is meant to be an rclone "crypt" remote, so
    the cloud provider only ever stores an encrypted file. Raises on failure
    (the cron wrapper then alerts); the local backup is already written."""
    remote = remote.rstrip("/")
    run(["rclone", "copyto", str(dest), f"{remote}/{dest.name}"], check=True, timeout=600)
    if keep_days > 0:
        prefix = dest.name.rsplit("-", 2)[0]
        run(
            [
                "rclone",
                "delete",
                remote,
                "--min-age",
                f"{keep_days}d",
                "--include",
                f"{prefix}-*.db",
            ],
            check=True,
            timeout=600,
        )
    logger.info("Off-site copy: %s/%s", remote, dest.name)


def newest_offsite_age_days(
    remote: str, *, now: datetime, run: Run = subprocess.run
) -> float | None:
    """Age in days of the newest backup on the remote, or None when empty."""
    out = run(
        ["rclone", "lsjson", remote.rstrip("/"), "--files-only"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    times = [
        datetime.fromisoformat(f["ModTime"].replace("Z", "+00:00"))
        for f in json.loads(out or "[]")
        if f.get("Name", "").endswith(".db")
    ]
    if not times:
        return None
    return (now.astimezone() - max(times)).total_seconds() / 86400


def restore(
    remote: str, dest_dir: str, name: str | None = None, *, run: Run = subprocess.run
) -> Path:
    """Download one off-site backup (the newest, or `name`) into `dest_dir`,
    decrypted by the rclone remote, and check it with SQLite's
    integrity_check. Doesn't touch the live database (`apply_restore` does,
    on request). Returns the restored file."""
    remote = remote.rstrip("/")
    if name is None:
        listing = run(
            ["rclone", "lsjson", remote, "--files-only"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        names = sorted(f["Name"] for f in json.loads(listing or "[]") if f["Name"].endswith(".db"))
        if not names:
            raise FileNotFoundError(f"no backups on {remote}")
        name = names[-1]  # stock-YYYYmmdd-HHMMSS.db sorts by time
    out_dir = Path(os.path.expanduser(dest_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name
    run(["rclone", "copyto", f"{remote}/{name}", str(dest)], check=True, timeout=600)
    with sqlite3.connect(dest) as db:
        verdict = db.execute("PRAGMA integrity_check").fetchone()[0]
    if verdict != "ok":
        raise RuntimeError(f"{dest} failed the integrity check: {verdict}")
    return dest


def apply_restore(restored: Path, db_path: str) -> Path:
    """Replace the live database with `restored` through SQLite's backup
    API, after saving the current one beside it. A plain file copy over a
    database in WAL mode can be corrupted by the leftover -wal file; the
    backup API writes through SQLite, so it can't. Returns the saved copy."""
    live = Path(os.path.expanduser(db_path))
    saved = live.with_name(f"{live.name}.before-restore-{datetime.now():%Y%m%d-%H%M%S}")
    with sqlite3.connect(live) as cur, sqlite3.connect(saved) as keep:
        cur.backup(keep)
    with sqlite3.connect(restored) as src, sqlite3.connect(live) as dst:
        src.backup(dst)
    return saved


def prune_logs(log_dirs: list[str], keep_days: int, *, now: datetime | None = None) -> int:
    """Delete `*.log` files last written more than `keep_days` ago. Returns
    how many went. The age is the file's mtime, so a log still being
    appended to is never a candidate."""
    if keep_days <= 0:
        return 0
    cutoff = ((now or datetime.now()) - timedelta(days=keep_days)).timestamp()
    removed = 0
    for d in log_dirs:
        path = Path(os.path.expanduser(d))
        if not path.is_dir():
            continue
        for f in path.glob("*.log"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
    if removed:
        logger.info("Deleted %d log file(s) older than %d days", removed, keep_days)
    return removed


def _log_dirs() -> list[str]:
    # The per-process logs (logging.py) and the per-job cron logs (run_job.sh).
    repo_logs = Path(__file__).resolve().parents[3] / "logs"
    return [os.getenv("LOG_DIR", "~/.stock_analyzer/logs"), str(repo_logs)]


# --- doctor -------------------------------------------------------------------

Check = tuple[str, Callable[[], str]]

# Free space below this share of a disk, or below the floor, fails the
# doctor so there is time to buy a disk: a full disk fails every SQLite
# write (and anything else on it — the data disk also holds the photos).
DISK_MIN_FREE_PCT = 10.0
DISK_MIN_FREE_GB = 50.0


def disk_space_problem(total: int, free: int) -> str | None:
    """Why this much free space is too little, or None when it's fine."""
    need = max(total * DISK_MIN_FREE_PCT / 100, DISK_MIN_FREE_GB * 1e9)
    if free >= need:
        return None
    return (
        f"only {free / 1e9:,.0f} GB free of {total / 1e9:,.0f} GB "
        f"(want {need / 1e9:,.0f} GB: {DISK_MIN_FREE_PCT:.0f}% of the disk, "
        f"never under {DISK_MIN_FREE_GB:.0f} GB)"
    )


def _disk_checks(settings: Settings) -> list[Check]:
    """One check per disk the database, its backups and the bar store sit
    on (a disk holding several is checked once)."""
    from ..data.bar_store import store_dir

    places = {
        "database": Path(os.path.expanduser(settings.discover_db_path)).parent,
        "backups": Path(os.path.expanduser(settings.backup_dir)),
    }
    bars = store_dir()
    if bars is not None:
        places["price cache"] = bars
    by_disk: dict[int, tuple[Path, list[str]]] = {}
    for label, path in places.items():
        while not path.exists() and path != path.parent:
            path = path.parent  # a backup dir not created yet: check its disk
        dev = path.stat().st_dev
        by_disk.setdefault(dev, (path, []))[1].append(label)

    def check_for(path: Path) -> Callable[[], str]:
        def check() -> str:
            usage = shutil.disk_usage(path)
            problem = disk_space_problem(usage.total, usage.free)
            if problem:
                raise RuntimeError(problem)
            return f"{usage.free / 1e9:,.0f} GB free of {usage.total / 1e9:,.0f} GB ({path})"

        return check

    return [(f"Disk ({', '.join(labels)})", check_for(path)) for path, labels in by_disk.values()]


# The nightly job rescans the universe; a week without a rescan means the
# job or Yahoo's screener has been failing, and a missing file means discover
# is back on the ~1,900-name bundled list (the slow path).
UNIVERSE_MAX_AGE_DAYS = 7


def universe_file_problem(path: Path, now: float) -> str | None:
    """Why the discover universe file is a problem, or None when it's fine."""
    if not path.exists():
        return f"{path} missing — discover falls back to the ~1,900-name bundled list"
    age = (now - path.stat().st_mtime) / 86400
    if age > UNIVERSE_MAX_AGE_DAYS:
        return f"{path} last rescanned {age:.0f} days ago — is the nightly job failing?"
    return None


def _state_checks() -> list[Check]:
    """Files the nightly job keeps current: the universe and the fetch cache."""
    import time

    from ..data import fetch_cache
    from ..data.universe_base import LOCAL_US_2B, load_base_universe

    def universe_check() -> str:
        problem = universe_file_problem(LOCAL_US_2B, time.time())
        if problem:
            raise RuntimeError(problem)
        age = (time.time() - LOCAL_US_2B.stat().st_mtime) / 86400
        return f"{len(load_base_universe())} tickers, rescanned {age:.1f} days ago"

    def cache_check() -> str:
        # Informational: an empty cache only means the next run is slow.
        parts = []
        for kind in ("fundamentals", "eps_revisions", "contracted_book"):
            got = fetch_cache.entries(kind)
            oldest = min((float(e["at"]) for e in got.values()), default=None)
            age = f", oldest {(time.time() - oldest) / 86400:.1f}d" if oldest else ""
            parts.append(f"{kind} {len(got)}{age}")
        return "; ".join(parts)

    return [("Discover universe", universe_check), ("Fetch cache", cache_check)]


# The backup runs nightly; two days without a new off-site copy means it
# has been failing (or the cloud login expired).
OFFSITE_MAX_AGE_DAYS = 2


def _offsite_checks(settings: Settings) -> list[Check]:
    if not settings.backup_remote:
        return []

    def check() -> str:
        age = newest_offsite_age_days(settings.backup_remote, now=datetime.now())
        if age is None:
            raise RuntimeError(f"no backups on {settings.backup_remote}")
        if age > OFFSITE_MAX_AGE_DAYS:
            raise RuntimeError(f"newest off-site backup is {age:.1f} days old")
        return f"newest on {settings.backup_remote} is {age:.1f} days old"

    return [("Off-site backup", check)]


def _llm_checks(settings: Settings) -> list[Check]:
    """One check per configured (provider, model): a model lookup is free
    and fails on a bad key or a model id the provider does not serve."""
    pairs: set[tuple[str, str]] = {
        ("claude", settings.discover_opus_model),
        ("claude", settings.discover_sonnet_model),
        (settings.llm_provider, settings.llm_model),
        (settings.discover_redteam_provider, settings.resolve_redteam_model()),
        (settings.discover_fallback_provider, settings.resolve_fallback_model()),
        *settings.resolve_ranker_rounds(),
    }
    if settings.discover_haiku_model:
        pairs.add(("claude", settings.discover_haiku_model))

    def lookup(provider: str, model: str) -> str:
        if provider == "claude":
            import anthropic

            anthropic.Anthropic().models.retrieve(model)
        elif provider == "openai":
            import openai

            openai.OpenAI().models.retrieve(model)
        elif provider == "gemini":
            from google import genai

            # Held in a name: a temporary client is closed before its call.
            client = genai.Client(api_key=settings.google_api_key)
            client.models.get(model=model)
        return "model found"

    return [
        (f"LLM {p}/{m}", lambda p=p, m=m: lookup(p, m))
        for p, m in sorted(pairs)  # type: ignore[misc]
    ]


# Below this Claude agreement (over at least SPOT_CHECK_MIN_FIELDS fields in
# the last 120 days) an open reader has drifted from its tested 83-92%.
SPOT_CHECK_FLOOR = 0.70
SPOT_CHECK_MIN_FIELDS = 20


def openrouter_host_problems(db: str, *, today: date) -> tuple[list[str], str]:
    """(problems, summary) for the OpenRouter hosts.

    A host that failed its known-answer check, or whose reads fell below
    the quality floor, is already skipped by every run (`excluded_hosts`):
    it is named in the summary, not a problem — the doctor emailed FAILED
    on 2026-10-05 for three hosts that were already out. It is a problem
    when a model has no approved host left, or when Claude's spot checks
    stop agreeing with a host, which nothing acts on by itself."""
    from ..openrouter import APPROVED_HOSTS
    from ..openrouter_hosts import (
        excluded_hosts,
        failed_checks,
        host_quality,
        spot_check_summary,
        unavailable_hosts,
    )

    unavailable = unavailable_hosts(db, today=today)
    skipped = [
        f"{host} ({model}): failed its last known-answer check"
        for model, hosts in sorted(failed_checks(db, today=today).items())
        for host in sorted(hosts - unavailable.get(model, set()))
    ]
    skipped += [
        f"{q['host']} ({q['model']}): {q['problem']}"
        for q in host_quality(db, today=today)
        if q["problem"]
    ]
    excluded = excluded_hosts(db, today=today)
    problems = [
        f"{model}: no usable host left — all {len(hosts)} approved hosts are skipped"
        for model, hosts in sorted(APPROVED_HOSTS.items())
        if hosts and set(hosts) <= excluded.get(model, set())
    ]
    for sc in spot_check_summary(db, today=today):
        if (
            sc["compared"] >= SPOT_CHECK_MIN_FIELDS
            and sc["agreed"] / sc["compared"] < SPOT_CHECK_FLOOR
        ):
            problems.append(
                f"{sc['model']} on {sc['provider']}: Claude agrees on only "
                f"{sc['agreed']}/{sc['compared']} fields"
            )
    hosts = sum(len(h) for h in APPROVED_HOSTS.values())
    summary = f"{hosts} approved model/host pairs"
    summary += f"; skipped until they pass: {'; '.join(skipped)}" if skipped else ", none flagged"
    gone = [f"{h} ({m})" for m, hs in sorted(unavailable.items()) for h in sorted(hs)]
    if gone:
        summary += f"; no endpoint at the last check: {', '.join(gone)}"
    return problems, summary


def _openrouter_checks(settings: Settings) -> list[Check]:
    if not settings.openrouter_api_key:
        return []

    def check() -> str:
        problems, summary = openrouter_host_problems(settings.discover_db_path, today=date.today())
        if problems:
            raise RuntimeError("; ".join(problems))
        return summary

    return [("OpenRouter hosts", check)]


def _data_checks(settings: Settings) -> list[Check]:
    def finnhub_check() -> str:
        from ..data import finnhub

        client = finnhub._client()
        if client is None:
            raise RuntimeError("FINNHUB_API_KEY not set")
        quote = client.quote("SPY")
        if not quote.get("c"):
            raise RuntimeError(f"no price in {quote}")
        return f"SPY {quote['c']}"

    def fred_check() -> str:
        import httpx

        key = os.getenv("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED_API_KEY not set")
        r = httpx.get(
            "https://api.stlouisfed.org/fred/series",
            params={"series_id": "DGS10", "api_key": key, "file_type": "json"},
            timeout=15,
        )
        r.raise_for_status()
        return "ok"

    def snaptrade_check() -> str:
        from ..data.brokerage import _client, _credentials, _unwrap

        user_id, user_secret = _credentials()
        accounts = _unwrap(
            _client().account_information.list_user_accounts(
                user_id=user_id, user_secret=user_secret
            )
        )
        return f"{len(accounts or [])} account(s)"

    def yahoo_check() -> str:
        """The most-used source, unofficial and able to break without notice:
        prices (bars, quotes) and analyst estimates are separate endpoints,
        and either can fail while the other works."""
        from ..data import frames, yf_gateway

        bars = frames.closes(
            frames.bars_from_pandas(
                yf_gateway.ticker_call("SPY", "doctor", lambda t: t.history(period="5d"))
            )
        )
        if bars is None:
            raise RuntimeError("no SPY price history — daily prices and the bar store are stuck")
        trend = yf_gateway.ticker_call("AAPL", "doctor", lambda t: t.eps_trend)
        if trend is None or trend.empty:
            raise RuntimeError(
                "no AAPL EPS trend — estimates, revisions, standouts and snapshots are blind"
            )
        return f"SPY {float(bars['Close'][-1]):.2f}; AAPL next-year EPS estimate ok"

    def smtp_check() -> str:
        import smtplib

        from ..reporting.smtp import SMTP_TIMEOUT_SECONDS, SmtpServer

        s = SmtpServer()
        ctx = s._ssl_context()
        if s.use_ssl:
            server = smtplib.SMTP_SSL(s.host, s.port, context=ctx, timeout=SMTP_TIMEOUT_SECONDS)
        else:
            server = smtplib.SMTP(s.host, s.port, timeout=SMTP_TIMEOUT_SECONDS)
            server.starttls(context=ctx)
        with server:
            server.login(s.username, s.password)
        return f"login ok ({s.host})"

    def database_check() -> str:
        path = Path(os.path.expanduser(settings.discover_db_path))
        if not path.exists():
            raise RuntimeError(f"{path} does not exist")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            result = conn.execute("PRAGMA quick_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"quick_check: {result}")
        mode = path.stat().st_mode & 0o777
        warn = f"; readable by others (mode {mode:o})" if mode & 0o077 else ""
        return f"{path.stat().st_size / 1e6:.1f} MB, integrity ok{warn}"

    def env_file_check() -> str:
        path = Path(".env")
        if not path.exists():
            return "no .env in this directory"
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise RuntimeError(f".env holds secrets but is mode {mode:o}; run chmod 600 .env")
        return f"mode {mode:o}"

    def key_only(name: str) -> Callable[[], str]:
        def check() -> str:
            if not os.getenv(name):
                raise RuntimeError(f"{name} not set")
            # Any real request would spend a search or a render.
            return "key set (quota not checked: every request costs one)"

        return check

    return [
        ("Database", database_check),
        (".env permissions", env_file_check),
        ("Yahoo", yahoo_check),
        ("Finnhub", finnhub_check),
        ("FRED", fred_check),
        ("SnapTrade", snaptrade_check),
        ("SMTP", smtp_check),
        ("Exa", key_only("EXA_API_KEY")),
        ("Tavily", key_only("TAVILY_API_KEY")),
        ("chart-img", key_only("CHART_IMG_API_KEY")),
    ]


def doctor(settings: Settings) -> int:
    """Run every check; returns how many failed."""
    failed = 0
    for name, check in [
        *_data_checks(settings),
        *_state_checks(),
        *_offsite_checks(settings),
        *_disk_checks(settings),
        *_llm_checks(settings),
        *_openrouter_checks(settings),
    ]:
        try:
            detail = check()
            print(f"  ok    {name}: {detail}")
        except Exception as e:  # noqa: BLE001 — each check reports its own failure
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {str(e)[:200]}")
    print(f"\n{failed} check(s) failed" if failed else "\nall checks passed")
    return failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ops", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("alert", help="email the tail of a failed job's log")
    a.add_argument("job")
    a.add_argument("log")
    a.add_argument("status", type=int)
    sub.add_parser("backup", help="copy the database into BACKUP_DIR, prune old logs")
    sub.add_parser("doctor", help="check keys, model ids and data sources")
    r = sub.add_parser("restore", help="download and check an off-site backup; --apply swaps it in")
    r.add_argument("name", nargs="?", help="backup file name (default: the newest)")
    r.add_argument("--to", default="~/.stock_analyzer/restore", help="where to put it")
    r.add_argument(
        "--apply",
        action="store_true",
        help="also replace the live database with it (the current one is saved first)",
    )
    sub.add_parser(
        "universe", help="rescan US stocks >= $2B with the quality rules (the nightly job does too)"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    settings = Settings.from_env()
    if args.cmd == "alert":
        alert(settings, args.job, args.log, args.status)
    elif args.cmd == "backup":
        dest = backup(settings.discover_db_path, settings.backup_dir, settings.backup_keep)
        prune_logs(_log_dirs(), settings.log_keep_days)
        if settings.backup_remote:
            upload_offsite(dest, settings.backup_remote, settings.backup_keep)
    elif args.cmd == "doctor":
        raise SystemExit(1 if doctor(settings) else 0)
    elif args.cmd == "restore":
        if not settings.backup_remote:
            raise SystemExit("BACKUP_REMOTE is not set — see README, 'Off-site backup'")
        got = restore(settings.backup_remote, args.to, args.name)
        with sqlite3.connect(got) as db:
            runs, picks = (
                db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("runs", "picks")
            )
        print(f"Restored {got} — integrity ok, {runs} runs, {picks} picks.")
        if args.apply:
            saved = apply_restore(got, settings.discover_db_path)
            print(f"Live database replaced; the previous one is saved as {saved}.")
        else:
            print("Not applied. To replace the live database with it, rerun with --apply")
            print("(the current database is saved first; best when no scheduled job is running).")
    elif args.cmd == "universe":
        from ..data.universe_base import refresh_us_2b

        print(f"Discover universe: {refresh_us_2b()} tickers written")


if __name__ == "__main__":
    main()
