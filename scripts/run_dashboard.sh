#!/usr/bin/env bash
# Wrapper for cron: regenerate the static dashboard page (no LLM calls).
# Runs after the close so the day's prices are final. The page it writes
# is served read-only by the stock-dashboard container; nothing about
# this touches the pipeline or the brokerage beyond a positions read.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

# Cron has a minimal PATH — extend it so `uv` resolves.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

LOG_FILE="$LOG_DIR/dashboard_$(date +%Y%m%d).log"
exec >>"$LOG_FILE" 2>&1

echo "=== run_dashboard: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
uv run dashboard
echo "=== finished: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
