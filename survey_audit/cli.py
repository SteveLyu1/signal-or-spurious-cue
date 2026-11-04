"""Stable, task-oriented commands for the study's historical script entry points.

The wrapper deliberately preserves the original scripts and their arguments;
manifest paths and code provenance must not be silently rewritten.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence


COMMANDS: dict[str, str] = {
    "prepare-survey": "dataset.tools.prepare_joint_evs_wvs",
    "api-smoke-test": "api_pilot.api_smoke_test",
    "build-confirmatory-manifest": "api_pilot.build_confirmatory_core_manifest",
    "run-forecasts": "api_pilot.survey_pilot_runner",
    "analyze-confirmatory": "api_pilot.run_confirmatory_strict_inference",
    "export-results": "api_pilot.export_final_results_tables",
    "verify-completion": "api_pilot.verify_experiment_completion",
}


def _usage() -> str:
    commands = "\n".join(f"  {name}" for name in COMMANDS)
    return (
        "Usage: python -m survey_audit <command> [script options]\n\n"
        "Research workflow commands:\n"
        f"{commands}\n\n"
        "The flags after <command> are forwarded to the underlying script.\n"
        "Some scripts require locally held licensed data and frozen manifests."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a descriptive command without changing study semantics."""
    supplied = list(sys.argv[1:] if argv is None else argv)
    if not supplied or supplied[0] in {"-h", "--help"}:
        print(_usage())
        return 0

    command = supplied[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n\n{_usage()}", file=sys.stderr)
        return 2

    original_argv = sys.argv[:]
    try:
        sys.argv = [COMMANDS[command], *supplied[1:]]
        runpy.run_module(COMMANDS[command], run_name="__main__", alter_sys=True)
    finally:
        sys.argv = original_argv
    return 0
