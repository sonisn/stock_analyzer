"""`ops`: the backup, the failure alert, and the doctor's bookkeeping."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from stock_analyzer.cli import ops


def _db(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, note TEXT)")
        conn.execute("INSERT INTO runs (note) VALUES ('kept')")
    return path


def test_backup_is_a_readable_private_copy_and_old_ones_rotate_out(tmp_path):
    src = _db(tmp_path / "stock.db")
    out = tmp_path / "backups"
    made = [ops.backup(str(src), str(out), keep=2, now=datetime(2026, 9, d)) for d in (20, 21, 22)]

    left = sorted(p.name for p in out.iterdir())
    assert left == [made[1].name, made[2].name]
    assert made[2].stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(made[2]) as conn:
        assert conn.execute("SELECT note FROM runs").fetchone() == ("kept",)


def test_backup_refuses_a_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        ops.backup(str(tmp_path / "nope.db"), str(tmp_path / "b"), keep=3)


def test_prune_logs_deletes_only_old_log_files(tmp_path):
    import os

    now = datetime(2026, 9, 25)
    old, fresh, other = tmp_path / "old.log", tmp_path / "fresh.log", tmp_path / "old.txt"
    for f in (old, fresh, other):
        f.write_text("x")
    old_ts = datetime(2026, 6, 1).timestamp()
    os.utime(old, (old_ts, old_ts))
    os.utime(other, (old_ts, old_ts))
    os.utime(fresh, (datetime(2026, 9, 24).timestamp(),) * 2)

    assert ops.prune_logs([str(tmp_path), str(tmp_path / "missing")], 90, now=now) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fresh.log", "old.txt"]
    assert ops.prune_logs([str(tmp_path)], 0, now=now) == 0


def test_alert_mails_the_tail_of_the_log(tmp_path):
    log = tmp_path / "portfolio_20260922.log"
    log.write_text("".join(f"line {i}\n" for i in range(200)) + "Traceback: boom\n")

    class _Settings:
        email_to = "me@example.com"

    with patch("stock_analyzer.reporting.smtp.SmtpServer") as smtp:
        ops.alert(_Settings(), "portfolio", str(log), 1)

    to, subject, body = smtp.return_value.send_email.call_args.args
    assert subject == "stock-analyzer: portfolio FAILED"
    assert "Traceback: boom" in body
    assert "line 0\n" not in body  # only the tail


def test_doctor_counts_failures_without_stopping(capsys):
    def boom():
        raise RuntimeError("no key")

    with (
        patch.object(ops, "_data_checks", return_value=[("A", lambda: "fine"), ("B", boom)]),
        patch.object(ops, "_llm_checks", return_value=[("C", boom)]),
        patch.object(ops, "_disk_checks", return_value=[]),
        patch.object(ops, "_state_checks", return_value=[]),
        patch.object(ops, "_offsite_checks", return_value=[]),
    ):
        assert ops.doctor(object()) == 2
    out = capsys.readouterr().out
    assert "ok    A: fine" in out
    assert "FAIL  B: RuntimeError: no key" in out


def test_disk_space_fails_below_ten_percent_or_fifty_gb():
    from stock_analyzer.cli.ops import disk_space_problem

    tb = 1e12
    assert disk_space_problem(int(3 * tb), int(2 * tb)) is None
    assert "only 250 GB free" in disk_space_problem(int(3 * tb), int(0.25 * tb))  # < 10%
    assert disk_space_problem(int(0.4 * tb), int(60e9)) is None  # 15%, over the floor
    assert disk_space_problem(int(0.4 * tb), int(45e9)) is not None  # under 50 GB


def test_the_doctor_flags_a_missing_or_stale_universe(tmp_path):
    import os

    from stock_analyzer.cli.ops import universe_file_problem

    path = tmp_path / "us_2b_universe.txt"
    assert "missing" in universe_file_problem(path, now=1e9)
    path.write_text("NVDA\n")
    os.utime(path, (1e9, 1e9))
    assert universe_file_problem(path, now=1e9 + 2 * 86400) is None
    assert "8 days ago" in universe_file_problem(path, now=1e9 + 8 * 86400)


def test_a_backup_goes_off_site_and_old_copies_there_are_pruned(tmp_path):
    import subprocess

    from stock_analyzer.cli.ops import upload_offsite

    calls: list[list[str]] = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    dest = tmp_path / "stock-20260927-020000.db"
    upload_offsite(dest, "box-crypt:stock-analyzer/", 14, run=run)
    assert calls == [
        ["rclone", "copyto", str(dest), "box-crypt:stock-analyzer/stock-20260927-020000.db"],
        [
            "rclone",
            "delete",
            "box-crypt:stock-analyzer",
            "--min-age",
            "14d",
            "--include",
            "stock-*.db",
        ],
    ]


def test_the_doctor_reads_the_newest_off_site_copy():
    import json
    import subprocess
    from datetime import datetime

    from stock_analyzer.cli.ops import newest_offsite_age_days

    listing = [
        {"Name": "stock-20260925-020000.db", "ModTime": "2026-09-25T06:00:00Z"},
        {"Name": "stock-20260926-020000.db", "ModTime": "2026-09-26T06:00:00Z"},
        {"Name": "notes.txt", "ModTime": "2026-09-27T05:00:00Z"},
    ]

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, json.dumps(listing), "")

    now = datetime(2026, 9, 27, 6, 0, tzinfo=UTC)
    assert newest_offsite_age_days("r:", now=now, run=run) == 1.0
    empty = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "[]", "")  # noqa: E731
    assert newest_offsite_age_days("r:", now=now, run=empty) is None


def _picks_db(path, rows):
    import sqlite3

    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE picks (ticker TEXT)")
        db.executemany("INSERT INTO picks VALUES (?)", [(r,) for r in rows])
    return path


def test_restore_fetches_the_newest_backup_and_checks_it(tmp_path):
    import json
    import shutil
    import sqlite3
    import subprocess

    import pytest

    from stock_analyzer.cli.ops import restore

    cloud = tmp_path / "cloud"
    cloud.mkdir()
    _picks_db(cloud / "stock-20260925-020000.db", ["OLD"])
    _picks_db(cloud / "stock-20260926-020000.db", ["NEW"])
    (cloud / "stock-20260920-020000.db").write_bytes(b"not a database")

    def run(cmd, **kw):
        if cmd[1] == "lsjson":
            listing = [{"Name": p.name} for p in cloud.iterdir()]
            return subprocess.CompletedProcess(cmd, 0, json.dumps(listing), "")
        shutil.copy(cloud / cmd[2].split("/")[-1], cmd[3])  # copyto remote/name dest
        return subprocess.CompletedProcess(cmd, 0, "", "")

    got = restore("box-crypt:", str(tmp_path / "out"), run=run)
    assert got.name == "stock-20260926-020000.db"
    with pytest.raises(sqlite3.DatabaseError):  # a corrupt file never passes as restored
        restore("box-crypt:", str(tmp_path / "out"), "stock-20260920-020000.db", run=run)


def test_apply_restore_swaps_the_database_and_keeps_the_old_one(tmp_path):
    import sqlite3

    from stock_analyzer.cli.ops import apply_restore

    live = _picks_db(tmp_path / "stock.db", ["LIVE"])
    with sqlite3.connect(live) as db:
        db.execute("PRAGMA journal_mode=WAL")
    restored = _picks_db(tmp_path / "restored.db", ["FROM-BACKUP"])
    saved = apply_restore(restored, str(live))

    def picks(p):
        with sqlite3.connect(p) as db:
            return [r for (r,) in db.execute("SELECT ticker FROM picks")]

    assert picks(live) == ["FROM-BACKUP"] and picks(saved) == ["LIVE"]
