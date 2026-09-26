"""Personal portfolio analyzer.

The server clock is UTC while every job runs on New York time, so the
market timezone is set here, before anything else in the package runs:
logging names its file at first use, which happens during imports, and a
switch made later inside each command's `main()` left log file names
(and their first line) in UTC while every other line was New York time.
"""

from .market_time import use_market_timezone

use_market_timezone()
