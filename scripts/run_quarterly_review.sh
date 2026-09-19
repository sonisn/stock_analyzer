#!/usr/bin/env bash
# Wrapper for cron: quarterly review of last quarter's advice (no LLM calls).
# Cron fires on days 1-7 of Jan/Apr/Jul/Oct; the command only acts on the
# quarter's first trading day.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

# Cron has a minimal PATH — extend it so `uv` resolves.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

LOG_FILE="$LOG_DIR/quarterly_review_$(date +%Y%m%d).log"
exec >>"$LOG_FILE" 2>&1

echo "=== run_quarterly_review: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
uv run quarterly-review
echo "=== finished: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
