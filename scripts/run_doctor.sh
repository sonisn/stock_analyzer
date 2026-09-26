#!/usr/bin/env bash
# Weekly health check (keys, logins, model ids; no paid calls). Exits non-zero on a
# problem, and run_job.sh then emails an alert. Cron: Sundays 8 PM New York.
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" doctor uv run ops doctor
