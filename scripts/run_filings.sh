#!/usr/bin/env bash
# Cron entry point; see run_job.sh. Weekly SEC filing read (read-filings):
# the week's new 10-Q/10-K/20-F across the universe, GLM-5.3 for the stocks
# acted on, GLM-5.3-Flash for the rest. A peak earnings week is ~300-400
# filings (~$1-2), so this run gets a $5 cap; anything past it waits a week.
export OPENROUTER_DAILY_CAP_USD="${FILINGS_WEEKLY_CAP_USD:-5}"
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" filings uv run read-filings
