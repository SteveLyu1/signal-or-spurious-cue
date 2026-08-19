#!/usr/bin/env python3
"""Offline dry-run validation for the three formal sentinel waves."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_confirmatory_sentinel_manifests as builder
import survey_pilot_runner as runner


ROOT = Path(__file__).resolve().parents[1]


def dry_run_manifest(path: Path) -> dict[str, Any]:
    manifest = runner.load_task_manifest(path)
    if manifest is None:
        raise ValueError("Sentinel manifest unexpectedly absent")
    runner.validate_v2_source_bindings(
        manifest, ROOT / manifest["source"]["analysis_manifest_file"]
    )
    profiles = runner.load_profiles(
        ROOT / manifest["source"]["profile_file"],
        max_per_country=manifest["source"]["profiles_per_country"],
    )
    profiles = runner.apply_task_manifest(profiles, manifest)
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
    by_provider = Counter(task["provider"] for task in tasks)
    expected_by_provider = {
        provider: manifest["expected"]["calls_per_provider"]
        for provider in manifest["provider_models"]
    }
    if dict(by_provider) != expected_by_provider:
        raise ValueError(f"Sentinel dry-run count mismatch: {by_provider}")
    if len(tasks) != manifest["expected"]["total_calls"]:
        raise ValueError("Sentinel dry-run total differs from manifest")
    if len({task["call_id"] for task in tasks}) != len(tasks):
        raise ValueError("Sentinel dry-run call IDs are not unique")
    if any(
        task.get("source_call_id")
        != task["call_id"].removesuffix("/ROBUST-REPEAT-01")
        for task in tasks
    ):
        raise ValueError("Sentinel dry-run is not bound to exact core call IDs")
    return {
        "wave": manifest["selection"]["wave"],
        "manifest": path.relative_to(ROOT).as_posix(),
        "manifest_sha256": builder.sha256(path),
        "planned_calls": len(tasks),
        "providers": dict(sorted(by_provider.items())),
        "unique_call_ids": len({task["call_id"] for task in tasks}),
        "prompt_version": bundle["version"],
        "immutable_output_path": manifest["immutable_output_path"],
        "immutable_checkpoint_path": manifest["immutable_checkpoint_path"],
        "valid": True,
        "api_requests_made": 0,
    }


def validate_suite(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["version"] != builder.SUITE_VERSION:
        raise ValueError("Unexpected sentinel suite version")
    if builder.sha256(builder.CORE_TASK_MANIFEST_PATH) != summary[
        "core_manifest_sha256"
    ]:
        raise ValueError("Sentinel suite/core manifest hash mismatch")
    results = []
    for wave in builder.WAVES:
        entry = summary["waves"][wave]
        manifest_path = ROOT / entry["manifest"]
        if builder.sha256(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"Sentinel manifest hash mismatch: {wave}")
        results.append(dry_run_manifest(manifest_path))
    if sum(row["planned_calls"] for row in results) != 576:
        raise ValueError("Sentinel suite must contain exactly 576 calls")
    return {
        "version": "confirmatory-sentinel-dry-run-v1",
        "status": "valid; no API request made",
        "suite_file": path.relative_to(ROOT).as_posix(),
        "suite_sha256": builder.sha256(path),
        "total_planned_calls": 576,
        "waves": results,
        "api_requests_made": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=builder.SUITE_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            ROOT / "api_pilot/outputs/confirmatory_sentinel_dry_run_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_suite(args.suite.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("status=valid waves=3 calls=576 api_requests_made=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
