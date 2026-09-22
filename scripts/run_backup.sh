#!/usr/bin/env bash
# Cron entry point: nightly database backup; see run_job.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" backup uv run ops backup
