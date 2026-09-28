#!/usr/bin/env bash
# Cron entry point; see run_job.sh. Weekly SEC filing read (read-filings):
# the week's new 10-Q/10-K/20-F across the universe, GLM-5.3 for the stocks
# acted on, GLM-5.3-Flash for the rest. A peak earnings week is ~300-400
# filings (~$1-2), so this run gets a $5 cap; anything past it waits a week.
export OPENROUTER_DAILY_CAP_USD="${FILINGS_WEEKLY_CAP_USD:-5}"
# Before the sweep, every approved host reads a made-up filing with known
# answers; a host that fails is skipped (openrouter_hosts.py).
#
# First Saturday of the month: Claude re-reads 5 random open-model reads
# (~$0.35) and the agreement is stored per model and host — the running
# measure `ops doctor` checks. Its own run so a failure can't stop the sweep.
dir="$(dirname "${BASH_SOURCE[0]}")"
if [ "$(TZ=America/New_York date +%d)" -le 7 ]; then
  "$dir/run_job.sh" filings-spot-check uv run read-filings --spot-check "${FILINGS_SPOT_CHECK_N:-5}" || true
fi
exec "$dir/run_job.sh" filings uv run read-filings
