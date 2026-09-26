"""`stock-analyzer <command> [args...]` — every CLI behind one entry point.

    stock-analyzer --help            list the commands
    stock-analyzer portfolio         same as `analyze-portfolio`
    stock-analyzer ops doctor        same as `ops doctor`

The per-command scripts (`analyze-portfolio`, `ops`, ...) still exist and
cron calls them; this only makes the set discoverable from one `--help`.
"""

from __future__ import annotations

import importlib
import sys

# command -> (module under stock_analyzer.cli, its standalone script, what it does)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "portfolio": ("portfolio", "analyze-portfolio", "daily portfolio email"),
    "insiders": ("insider", "analyze-insiders", "insider / congressional / Form 4 email"),
    "discover": ("discover", "discover-stocks", "find new stocks to buy (LLM pipeline)"),
    "rebalance": ("rebalance", "rebalance-portfolio", "discover + review holdings + plan"),
    "dashboard": ("dashboard", "dashboard", "static HTML dashboard"),
    "model-review": ("model_review", "model-review", "monthly model review email"),
    "quarterly-review": ("quarterly_review", "quarterly-review", "quarterly suggestions review"),
    "tax-planner": ("tax_planner", "tax-planner", "year-end tax plan"),
    "plan-check": ("plan_check", "plan-check", "asset location + goal projection"),
    "train-model": ("train_model", "train-model", "train the forward-return model"),
    "validate-screen": ("validate_screen", "validate-screen", "does the screen score predict?"),
    "score-attribution": ("score_attribution", "score-attribution", "which score parts earn"),
    "replay-rebalance": ("replay_rebalance", "replay-rebalance", "re-run a stored rebalance plan"),
    "ops": ("ops", "ops", "backup / doctor / alert"),
}


def _usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {help_}" for name, (_, _, help_) in COMMANDS.items()]
    return (
        "usage: stock-analyzer <command> [args...]\n\ncommands:\n"
        + "\n".join(lines)
        + "\n\n`stock-analyzer <command> --help` shows a command's own options."
    )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage())
        return
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"stock-analyzer: unknown command {name!r}\n\n{_usage()}", file=sys.stderr)
        raise SystemExit(2)
    module_name, script, _ = COMMANDS[name]
    module = importlib.import_module(f"stock_analyzer.cli.{module_name}")
    # Several commands parse sys.argv themselves, so hand them theirs.
    sys.argv = [script, *rest]
    result = module.main()
    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == "__main__":
    main()
