#!/usr/bin/env bash
# Cron entry point; see run_job.sh. IBD-style ratings before the open, then
# the dashboard, so its "Market leaders" tab is current by 9:30.
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" ibd bash -c 'uv run ibd-ratings && uv run dashboard'
