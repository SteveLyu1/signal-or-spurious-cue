#!/usr/bin/env python3
"""Offline, fail-closed runtime-quality summary for frozen formal calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import survey_pilot_runner as runner
except ImportError:  # Direct script execution.
    import survey_pilot_runner as runner


ROOT = Path(__file__).resolve().parents[1]
CORE = "dataset/analysis/confirmatory_core_task_manifest_v2.json"
ROBUSTNESS = "dataset/analysis/gate_c_robustness_suite_v2.json"
SENTINELS = "dataset/analysis/confirmatory_sentinel_suite_v1.json"
OUTPUT_JSON = "api_pilot/outputs/formal_runtime_quality.json"
OUTPUT_CSV = "api_pilot/outputs/formal_runtime_quality_by_provider.csv"
EXPECTED_TOTAL = 37_344
TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def manifest_roster(root: Path) -> list[tuple[str, Path, str | None]]:
    roster = [("core", root / CORE, None)]
    robustness_path = root / ROBUSTNESS
    robustness = load_json(robustness_path)
    for name, entry in robustness.get("manifests", {}).items():
        roster.append((f"robustness_{name}", root / entry["file"], entry["sha256"]))
    sentinel_path = root / SENTINELS
    sentinels = load_json(sentinel_path)
    for wave, entry in sentinels.get("waves", {}).items():
        roster.append((f"sentinel_{wave}", root / entry["manifest"], entry["manifest_sha256"]))
    names = [name for name, _, _ in roster]
    required = {
        "core", "robustness_repeat_identical", "robustness_single_item",
        "robustness_wording_b", "robustness_option_reverse",
        "sentinel_before", "sentinel_mid", "sentinel_after",
    }
    if set(names) != required or len(names) != len(required):
        raise ValueError(f"Frozen runtime roster mismatch: {sorted(names)}")
    return roster


def _verify_source_hashes(root: Path, manifest: dict[str, Any]) -> None:
    for key, relative in manifest["source"].items():
        hash_key = key + "_sha256"
        if key.endswith("_file") and hash_key in manifest["source"]:
            path = root / relative
            if not path.is_file() or sha256(path) != manifest["source"][hash_key]:
                raise ValueError(f"Frozen source hash mismatch: {relative}")
    prompt = manifest["prompt"]
    if sha256(root / prompt["file"]) != prompt["sha256"]:
        raise ValueError(f"Frozen prompt hash mismatch: {prompt['file']}")


def expected_calls(root: Path, path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    manifest = runner.load_task_manifest(path)
    if manifest is None or manifest.get("schema_version") != "survey-task-manifest-v2":
        raise ValueError(f"Expected a version-2 task manifest: {path}")
    _verify_source_hashes(root, manifest)
    profiles = runner.load_profiles(
        root / manifest["source"]["profile_file"],
        manifest["source"]["profiles_per_country"],
    )
    profiles = runner.apply_task_manifest(profiles, manifest)
    if manifest["runner_configuration"]["reverse_target_options"]:
        profiles = [dict(profile) for profile in profiles]
        for profile in profiles:
            profile["target_options"] = {
                item: list(reversed(options))
                for item, options in profile["target_options"].items()
            }
    bundle = runner.load_prompt_bundle(root / manifest["prompt"]["file"])
    tasks = runner.build_tasks(
        profiles, list(manifest["provider_models"]), manifest["provider_models"],
        bundle=bundle,
        target_batch_size=manifest["runner_configuration"]["target_batch_size"],
        cells=manifest["parsed_cells"],
        target_batch_ids=tuple(manifest["target_batches"]),
        experiment_id=manifest["experiment_id"],
    )
    tasks = runner.apply_task_units(tasks, manifest)
    index = {
        task["call_id"]: {"provider": task["provider"], "model": task["model"]}
        for task in tasks
    }
    expected = int(manifest["expected"]["total_calls"])
    if len(tasks) != expected or len(index) != expected:
        raise ValueError(f"Generated call roster mismatch: {path}")
    return index, manifest


def _percentile(values: list[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(probability * len(ordered)) - 1)])


def _blank_bucket(planned: int = 0) -> dict[str, Any]:
    return {
        "planned_calls": planned, "observed_calls": 0, "valid_calls": 0,
        "successful_calls": 0, "failed_calls": 0, "terminal_failures": 0,
        "missing_calls": 0, "recorded_attempts": 0, "recorded_retries": 0,
        "attempt_latencies": [], "final_latencies": [],
        "http_status_attempts": Counter(), "http_status_final": Counter(),
        "error_types": Counter(),
        **{f"{field}_total": 0 for field in TOKEN_FIELDS},
        **{f"{field}_reported_calls": 0 for field in TOKEN_FIELDS},
    }


def _finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in bucket.items() if not key.endswith("latencies")}
    result["valid_rate"] = (
        bucket["valid_calls"] / bucket["planned_calls"]
        if bucket["planned_calls"] else None
    )
    for prefix, values in (("attempt_latency_ms", bucket["attempt_latencies"]),
                           ("final_latency_ms", bucket["final_latencies"])):
        result[f"{prefix}_mean"] = statistics.fmean(values) if values else None
        result[f"{prefix}_p50"] = _percentile(values, 0.50)
        result[f"{prefix}_p95"] = _percentile(values, 0.95)
        result[f"{prefix}_max"] = float(max(values)) if values else None
    for key in ("http_status_attempts", "http_status_final", "error_types"):
        result[key] = dict(sorted(bucket[key].items(), key=lambda pair: str(pair[0])))
    for field in TOKEN_FIELDS:
        result[f"{field}_missing_calls"] = (
            bucket["observed_calls"] - bucket[f"{field}_reported_calls"]
        )
    return result


def audit_output(path: Path, expected: dict[str, dict[str, str]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    providers = sorted({value["provider"] for value in expected.values()})
    buckets = {provider: _blank_bucket(sum(value["provider"] == provider for value in expected.values())) for provider in providers}
    errors: list[str] = []
    seen: set[str] = set()
    malformed = duplicates = unknown = 0
    if path.is_file():
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    else:
        lines = []
        errors.append("output file is absent")
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            errors.append(f"line {line_number}: malformed JSON")
            continue
        call_id = row.get("call_id")
        if call_id in seen:
            duplicates += 1
            errors.append(f"line {line_number}: duplicate call_id")
            continue
        seen.add(call_id)
        spec = expected.get(call_id)
        if spec is None:
            unknown += 1
            errors.append(f"line {line_number}: unknown call_id")
            continue
        provider = spec["provider"]
        bucket = buckets[provider]
        bucket["observed_calls"] += 1
        structurally_valid = row.get("provider") == provider and row.get("model_requested") == spec["model"]
        attempts = row.get("attempts")
        history = row.get("attempt_history")
        if not isinstance(attempts, int) or attempts < 1 or not isinstance(history, list) or len(history) != attempts:
            structurally_valid = False
            errors.append(f"line {line_number}: invalid attempt contract")
            attempts = len(history) if isinstance(history, list) else 0
            history = history if isinstance(history, list) else []
        bucket["recorded_attempts"] += attempts
        bucket["recorded_retries"] += max(attempts - 1, 0)
        for attempt in history:
            status = attempt.get("status_code")
            latency = attempt.get("latency_ms")
            bucket["http_status_attempts"][str(status)] += 1
            if isinstance(latency, (int, float)) and latency >= 0:
                bucket["attempt_latencies"].append(int(latency))
        status = row.get("status_code")
        bucket["http_status_final"][str(status)] += 1
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)) and latency >= 0:
            bucket["final_latencies"].append(int(latency))
        success = row.get("success") is True
        bucket["successful_calls"] += int(success)
        bucket["failed_calls"] += int(not success)
        bucket["terminal_failures"] += int(row.get("terminal_failure") is True)
        if row.get("error_type"):
            bucket["error_types"][str(row["error_type"])] += 1
        for field in TOKEN_FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)) and value >= 0:
                bucket[f"{field}_total"] += int(value)
                bucket[f"{field}_reported_calls"] += 1
        if structurally_valid and success and not row.get("terminal_failure"):
            bucket["valid_calls"] += 1
    missing = set(expected).difference(seen)
    for call_id in missing:
        buckets[expected[call_id]["provider"]]["missing_calls"] += 1
    complete = not errors and not missing and not malformed and not duplicates and not unknown and all(
        bucket["valid_calls"] == bucket["planned_calls"] for bucket in buckets.values()
    )
    audit = {
        "path": path.as_posix(), "expected_calls": len(expected),
        "observed_known_unique_calls": sum(bucket["observed_calls"] for bucket in buckets.values()),
        "missing_calls": len(missing), "malformed_rows": malformed,
        "duplicate_call_ids": duplicates, "unknown_call_ids": unknown,
        "complete": complete, "sha256": sha256(path) if path.is_file() else None,
        "errors": errors[:100],
    }
    return audit, {provider: _finish_bucket(bucket) for provider, bucket in buckets.items()}


def summarize(root: Path = ROOT) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    aggregate: dict[str, dict[str, Any]] = defaultdict(_blank_bucket)
    roster = manifest_roster(root)
    for name, manifest_path, frozen_hash in roster:
        observed_hash = sha256(manifest_path)
        if frozen_hash and observed_hash != frozen_hash:
            raise ValueError(f"Frozen manifest hash mismatch: {name}")
        expected, manifest = expected_calls(root, manifest_path)
        output = root / manifest["immutable_output_path"]
        audit, by_provider = audit_output(output, expected)
        sources[name] = {
            "manifest": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": observed_hash,
            "output": manifest["immutable_output_path"],
            "output_sha256": audit["sha256"], "audit": audit,
            "by_provider": by_provider,
        }
        for provider, values in by_provider.items():
            target = aggregate[provider]
            for key in (
                "planned_calls", "observed_calls", "valid_calls", "successful_calls",
                "failed_calls", "terminal_failures", "missing_calls",
                "recorded_attempts", "recorded_retries",
            ):
                target[key] += values[key]
            for field in TOKEN_FIELDS:
                target[f"{field}_total"] += values[f"{field}_total"]
                target[f"{field}_reported_calls"] += values[f"{field}_reported_calls"]
            # Exact pooled latency/status summaries are rebuilt from source JSONLs below.
        
    # Re-audit rows once to pool distributions without retaining response text.
    pooled_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources.values():
        output_path = root / source["output"]
        if not output_path.is_file():
            continue
        for line in output_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                pooled_rows[row.get("provider", "unknown")].append(row)
    for provider, rows in pooled_rows.items():
        bucket = aggregate[provider]
        for row in rows:
            bucket["final_latencies"].append(int(row.get("latency_ms") or 0))
            bucket["http_status_final"][str(row.get("status_code"))] += 1
            if row.get("error_type"):
                bucket["error_types"][str(row["error_type"])] += 1
            for attempt in row.get("attempt_history") or []:
                bucket["attempt_latencies"].append(int(attempt.get("latency_ms") or 0))
                bucket["http_status_attempts"][str(attempt.get("status_code"))] += 1
    providers = {provider: _finish_bucket(bucket) for provider, bucket in aggregate.items()}
    total_planned = sum(row["planned_calls"] for row in providers.values())
    complete = total_planned == EXPECTED_TOTAL and all(source["audit"]["complete"] for source in sources.values())
    return {
        "version": "formal-runtime-quality-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "incomplete", "complete": complete,
        "offline_only": True, "api_requests_made": 0,
        "cost": {"status": "pending_not_estimated", "reason": "Provider prices are time-varying and are not frozen inputs."},
        "expected_total_calls": EXPECTED_TOTAL, "observed_planned_calls": total_planned,
        "sources": sources, "by_provider": providers,
    }


def write_outputs(root: Path, report: dict[str, Any]) -> None:
    json_path, csv_path = root / OUTPUT_JSON, root / OUTPUT_CSV
    rows = []
    for provider, values in sorted(report["by_provider"].items()):
        row = {"provider": provider, **values}
        for field in ("http_status_attempts", "http_status_final", "error_types"):
            row[field] = json.dumps(row[field], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    fieldnames = list(rows[0]) if rows else ["provider"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, csv_path)
    report["artifacts"] = {"provider_csv": OUTPUT_CSV, "provider_csv_sha256": sha256(csv_path)}
    atomic_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize(args.root.resolve())
    write_outputs(args.root.resolve(), report)
    print(json.dumps({"status": report["status"], "calls": report["observed_planned_calls"]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
