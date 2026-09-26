#!/usr/bin/env bash
# Nightly earnings-standout check (no LLM, no email). Cron: weekdays ~22:00 New York.
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" earnings-watch uv run earnings-watch
