#!/usr/bin/env bash
# Cron wrapper: run one stock-analyzer command with a dated log, and email
# the log's tail if it fails (`ops alert`) — a failed job used to be silent.
#
#   scripts/run_job.sh <job-name> <command> [args...]
#   e.g. scripts/run_job.sh portfolio uv run analyze-portfolio
set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <job-name> <command> [args...]" >&2
    exit 2
fi
JOB="$1"
shift

# Ubuntu's cron ignores CRON_TZ and this machine's clock is UTC, so a job
# meant for a New York time is scheduled at BOTH UTC hours it can fall on
# (EDT, UTC-4, and EST, UTC-5) with NY_AT=HH:MM set, e.g.
#   30 13,14 * * 1-5 NY_AT=09:30 .../run_portfolio.sh
# Only the firing whose New York hour matches runs; the other exits quietly.
# The minute is left to cron, so a start a few seconds late still counts.
if [ -n "${NY_AT:-}" ]; then
    if [ "$(TZ=America/New_York date +%H)" != "${NY_AT%%:*}" ]; then
        exit 0
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# Cron has a minimal PATH — extend it so `uv` resolves.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Cron starts jobs with a soft limit of 1,024 open files, far below a login
# shell's; a run that fans out over ~1,900 tickers has hit it (2026-09-29).
# Raise it toward the hard limit — headroom, not a substitute for closing files.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# Named by the New York day, the calendar the Python side runs on
# (market_time.py): the 22:00 NY earnings watch is 02:00 UTC tomorrow.
LOG_FILE="$LOG_DIR/${JOB}_$(TZ=America/New_York date +%Y%m%d).log"
exec >>"$LOG_FILE" 2>&1

echo "=== ${JOB}: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z') ==="
"$@"
status=$?
echo "=== finished (status ${status}): $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z') ==="

if [ "$status" -ne 0 ]; then
    uv run ops alert "$JOB" "$LOG_FILE" "$status" || echo "alert email failed too"
fi
exit "$status"
