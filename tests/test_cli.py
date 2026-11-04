"""Contract tests for the task-oriented public research CLI."""

from __future__ import annotations

import importlib.util
import sys

from survey_audit import cli


def test_public_targets_resolve() -> None:
    for module_name in cli.COMMANDS.values():
        assert importlib.util.find_spec(module_name) is not None


def test_help_lists_research_workflows(capsys) -> None:
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "run-forecasts" in output
    assert "analyze-confirmatory" in output


def test_unknown_command_returns_nonzero(capsys) -> None:
    assert cli.main(["unrecognized-task"]) == 2
    assert "Unknown command" in capsys.readouterr().err


def test_options_are_forwarded_without_rewriting(monkeypatch) -> None:
    calls = []
    original_argv = sys.argv[:]

    def fake_run_module(module_name, *, run_name, alter_sys):
        calls.append((module_name, run_name, alter_sys, sys.argv[:]))

    monkeypatch.setattr(cli.runpy, "run_module", fake_run_module)
    assert cli.main(["api-smoke-test", "--help"]) == 0
    assert calls == [(
        "api_pilot.api_smoke_test",
        "__main__",
        True,
        ["api_pilot.api_smoke_test", "--help"],
    )]
    assert sys.argv == original_argv


def test_historical_smoke_test_import_uses_same_module() -> None:
    import api_pilot.api_smoke_test as canonical
    import api_pilot.pilot_runner as historical

    assert historical is canonical
