"""`ops` — keeping the scheduled jobs honest. No LLM tokens are spent.

  ops alert JOB LOG STATUS   email the tail of a failed job's log
  ops backup                 copy the database, keeping the newest BACKUP_KEEP,
                             and delete logs older than LOG_KEEP_DAYS
  ops doctor                 check every key, model id and data source for free,
                             and free space on the disks the data lives on

`scripts/run_job.sh` calls `alert` when a cron job exits non-zero; before
it existed a failed job was silent until someone noticed an email missing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
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
        from ..data import yf_gateway

        bars = yf_gateway.ticker_call("SPY", "doctor", lambda t: t.history(period="5d"))
        if bars is None or bars.empty:
            raise RuntimeError("no SPY price history — daily prices and the bar store are stuck")
        trend = yf_gateway.ticker_call("AAPL", "doctor", lambda t: t.eps_trend)
        if trend is None or trend.empty:
            raise RuntimeError(
                "no AAPL EPS trend — estimates, revisions, standouts and snapshots are blind"
            )
        return f"SPY {float(bars['Close'].iloc[-1]):.2f}; AAPL next-year EPS estimate ok"

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
        ("Tavily", key_only("TAVILY_API_KEY")),
        ("chart-img", key_only("CHART_IMG_API_KEY")),
    ]


def doctor(settings: Settings) -> int:
    """Run every check; returns how many failed."""
    failed = 0
    for name, check in [*_data_checks(settings), *_disk_checks(settings), *_llm_checks(settings)]:
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
    args = parser.parse_args(argv)

    load_dotenv()
    settings = Settings.from_env()
    if args.cmd == "alert":
        alert(settings, args.job, args.log, args.status)
    elif args.cmd == "backup":
        backup(settings.discover_db_path, settings.backup_dir, settings.backup_keep)
        prune_logs(_log_dirs(), settings.log_keep_days)
    elif args.cmd == "doctor":
        raise SystemExit(1 if doctor(settings) else 0)


if __name__ == "__main__":
    main()
