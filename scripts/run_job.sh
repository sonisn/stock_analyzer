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

LOG_FILE="$LOG_DIR/${JOB}_$(date +%Y%m%d).log"
exec >>"$LOG_FILE" 2>&1

echo "=== ${JOB}: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
"$@"
status=$?
echo "=== finished (status ${status}): $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

if [ "$status" -ne 0 ]; then
    uv run ops alert "$JOB" "$LOG_FILE" "$status" || echo "alert email failed too"
fi
exit "$status"
