"""The umbrella `stock-analyzer` command stays in step with the scripts."""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

from stock_analyzer.cli import main as cli

SCRIPTS = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"][
    "scripts"
]


def test_every_script_is_a_command_and_points_at_the_same_main():
    covered = {script for _, script, _ in cli.COMMANDS.values()}
    assert covered == set(SCRIPTS) - {"stock-analyzer"}
    for module_name, script, _ in cli.COMMANDS.values():
        assert SCRIPTS[script] == f"stock_analyzer.cli.{module_name}:main"
        assert callable(importlib.import_module(f"stock_analyzer.cli.{module_name}").main)


def test_dispatch_passes_the_remaining_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", list(sys.argv))  # restored after the test
    seen = {}

    def fake_main():
        seen["argv"] = list(sys.argv)
        return 0

    monkeypatch.setattr(importlib.import_module("stock_analyzer.cli.ops"), "main", fake_main)
    with pytest.raises(SystemExit) as exit_:
        cli.main(["ops", "doctor"])
    assert exit_.value.code == 0
    assert seen["argv"] == ["ops", "doctor"]


def test_help_and_unknown_command(capsys):
    cli.main(["--help"])
    assert "portfolio" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_:
        cli.main(["nope"])
    assert exit_.value.code == 2
