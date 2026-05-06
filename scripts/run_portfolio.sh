#!/usr/bin/env bash
# Wrapper for cron: run the portfolio analysis pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

# Cron has a minimal PATH — extend it so `uv` resolves.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

LOG_FILE="$LOG_DIR/portfolio_$(date +%Y%m%d).log"
exec >>"$LOG_FILE" 2>&1

echo "=== run_portfolio: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
uv run analyze-portfolio
echo "=== finished: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
