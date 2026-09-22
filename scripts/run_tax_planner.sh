#!/usr/bin/env bash
# Cron entry point; see run_job.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" tax_planner uv run tax-planner
