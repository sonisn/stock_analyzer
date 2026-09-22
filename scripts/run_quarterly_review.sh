#!/usr/bin/env bash
# Cron entry point; see run_job.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" quarterly_review uv run quarterly-review
