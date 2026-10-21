#!/usr/bin/env bash
# Cron @reboot: run the scheduled jobs the machine missed while it was down
# (src/stock_analyzer/catch_up.py). The sleep lets the network and the data
# disk come up first.
sleep "${CATCH_UP_DELAY_SECONDS:-300}"
exec "$(dirname "${BASH_SOURCE[0]}")/run_job.sh" catch-up uv run ops catch-up
