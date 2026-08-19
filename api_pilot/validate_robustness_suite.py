#!/usr/bin/env python3
"""Offline dry-run validation for every Gate C robustness manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_robustness_manifests as builder
import survey_pilot_runner as runner


ROOT = Path(__file__).resolve().parents[1]


def dry_run_manifest(path: Path) -> dict[str, Any]:
    manifest = runner.load_task_manifest(path)
    if manifest is None:
        raise ValueError("Manifest unexpectedly absent")
    profiles = runner.load_profiles(
        ROOT / manifest["source"]["profile_file"],
        max_per_country=manifest["source"]["profiles_per_country"],
    )
    profiles = runner.apply_task_manifest(profiles, manifest)
    if manifest["runner_configuration"]["reverse_target_options"]:
        profiles = [dict(profile) for profile in profiles]
        for profile in profiles:
            profile["target_options"] = {
                item_id: list(reversed(options))
                for item_id, options in profile["target_options"].items()
            }
    bundle = runner.load_prompt_bundle(ROOT / manifest["prompt"]["file"])
    tasks = runner.build_tasks(
        profiles,
        list(manifest["provider_models"]),
        manifest["provider_models"],
        bundle=bundle,
        target_batch_size=manifest["runner_configuration"]["target_batch_size"],
        cells=manifest["parsed_cells"],
        target_batch_ids=tuple(manifest["target_batches"]),
        experiment_id=manifest["experiment_id"],
    )
    tasks = runner.apply_task_units(tasks, manifest)
    by_provider = {
        provider: len([task for task in tasks if task["provider"] == provider])
        for provider in manifest["provider_models"]
    }
    if any(
        count != manifest["expected"]["calls_per_provider"]
        for count in by_provider.values()
    ):
        raise ValueError(f"Dry-run count mismatch: {by_provider}")
    if len(tasks) != manifest["expected"]["total_calls"]:
        raise ValueError("Dry-run total does not match manifest")
    call_ids = {task["call_id"] for task in tasks}
    if len(call_ids) != len(tasks):
        raise ValueError("Dry-run call IDs are not unique")
    if manifest["task_mode"] == "repeat_identical":
        if any(task["prompt"] == "" or not task.get("source_call_id") for task in tasks):
            raise ValueError("Repeat dry-run lacks source call binding")
        if any(not task["call_id"].endswith("/ROBUST-REPEAT-01") for task in tasks):
            raise ValueError("Repeat dry-run lacks repeat call suffix")
    return {
        "manifest": path.relative_to(ROOT).as_posix(),
        "experiment_id": manifest["experiment_id"],
        "providers": by_provider,
        "planned_calls": len(tasks),
        "unique_call_ids": len(call_ids),
        "prompt_version": bundle["version"],
        "output": manifest["immutable_output_path"],
        "checkpoint": manifest["immutable_checkpoint_path"],
        "valid": True,
    }


def validate_suite(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = []
    for entry in summary["manifests"].values():
        path = ROOT / entry["file"]
        if builder.sha256(path) != entry["sha256"]:
            raise ValueError(f"Suite manifest hash mismatch: {path}")
        results.append(dry_run_manifest(path))
    return {
        "version": "gate-c-robustness-dry-run-v2",
        "status": "valid; no API request made",
        "suite_file": summary_path.relative_to(ROOT).as_posix(),
        "suite_sha256": builder.sha256(summary_path),
        "total_planned_calls": sum(result["planned_calls"] for result in results),
        "experiments": results,
        "formal_run_schema_recommendation": {
            "schema_version": builder.SCHEMA_VERSION,
            "note": (
                "Use these runs only as Gate C measurement checks. The v4 transfer task "
                "binds the formal prompt without duplicating a formal outcome analysis."
            ),
            "required_binding": [
                "source profile and parent manifest hashes",
                "prompt file, version and hash",
                "official provider/model roster",
                "explicit profile-condition-evidence-batch units",
                "selection seed and method",
                "expected counts",
                "immutable output and checkpoint paths",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "dataset" / "analysis" / "gate_c_robustness_suite_v2.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "api_pilot" / "outputs" / "gate_c_robustness_dry_run_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_suite(args.suite.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"status={result['status']} experiments={len(result['experiments'])} "
        f"calls={result['total_planned_calls']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
