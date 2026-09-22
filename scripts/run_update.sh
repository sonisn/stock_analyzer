#!/usr/bin/env bash
# Cron entry point: tested update of this checkout; see update.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" update "$(dirname "${BASH_SOURCE[0]}")/update.sh"
