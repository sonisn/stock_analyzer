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
