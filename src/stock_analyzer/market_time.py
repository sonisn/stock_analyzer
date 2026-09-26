"""Run every command on US market time.

The server clock is UTC while cron fires on New York time, so a manual run
after ~8 pm Eastern would otherwise stamp picks, suggestions, earnings
countdowns and email subjects with tomorrow's date. Importing the package
calls `use_market_timezone()` first; `date.today()` / `datetime.now()` then
mean the New York calendar day everywhere. MARKET_TZ overrides it.
"""

from __future__ import annotations

import os
import time

DEFAULT_MARKET_TZ = "America/New_York"


def use_market_timezone() -> None:
    os.environ["TZ"] = os.environ.get("MARKET_TZ") or DEFAULT_MARKET_TZ
    if hasattr(time, "tzset"):
        time.tzset()
