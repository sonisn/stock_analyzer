#!/usr/bin/env bash
# Cron entry point; see run_job.sh. The portfolio's value at the day's
# closes, weekdays after the close: no model calls, no email. Keeps the
# performance and goal-pace history daily now that the email is weekly.
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" snapshot uv run analyze-portfolio --snapshot-only
