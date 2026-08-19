#!/usr/bin/env python3
"""Build GPT-5.4 parity overlays from the frozen four-provider manifests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = {"openai": "gpt-5.4"}
SOURCES = {
    "repeat_identical": "dataset/analysis/gate_c_repeat_identical_manifest_v2.json",
    "single_item": "dataset/analysis/gate_c_single_item_manifest_v2.json",
    "wording_b": "dataset/analysis/gate_c_prompt_v4_transfer_manifest_v2.json",
    "option_reverse": "dataset/analysis/gate_c_option_reverse_manifest_v2.json",
    "confirmatory_core": "dataset/analysis/confirmatory_core_task_manifest_v2.json",
    "sentinel_before": "dataset/analysis/confirmatory_sentinel_before_manifest_v1.json",
    "sentinel_mid": "dataset/analysis/confirmatory_sentinel_mid_manifest_v1.json",
    "sentinel_after": "dataset/analysis/confirmatory_sentinel_after_manifest_v1.json",
    "batch_context": "dataset/analysis/confirmatory_batch_context_replication_v1.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(name: str, source_rel: str) -> tuple[str, dict[str, Any]]:
    source_path = ROOT / source_rel
    source = json.loads(source_path.read_text(encoding="utf-8"))
    result = deepcopy(source)
    calls = int(source["expected"]["calls_per_provider"])
    result["version"] = f"{source['version']}-openai-parity-v1"
    result["status"] = (
        "researcher-directed prospective roster amendment; task units and inputs "
        "are identical to the frozen source manifest"
    )
    result["experiment_id"] = f"{source['experiment_id']}_openai_parity"
    result["provider_models"] = MODEL
    result["expected"] = {"providers": 1, "calls_per_provider": calls, "total_calls": calls}
    output = f"api_pilot/restricted_outputs/{result['experiment_id']}.jsonl"
    checkpoint = f"api_pilot/restricted_outputs/{result['experiment_id']}.checkpoint.sqlite3"
    result["immutable_output_path"] = output
    result["immutable_checkpoint_path"] = checkpoint
    result["roster_amendment"] = {
        "file": "dataset/analysis/openai_parity_roster_amendment_v1.json",
        "source_manifest": source_rel,
        "source_manifest_sha256": sha256(source_path),
        "changed_fields": [
            "version", "status", "experiment_id", "provider_models", "expected",
            "immutable_output_path", "immutable_checkpoint_path", "roster_amendment",
        ],
        "unchanged_scientific_fields": [
            "task_mode", "profile_ids", "cells", "target_batches", "task_units",
            "profile_target_batches", "source", "prompt", "runner_configuration",
            "selection", "analysis_contract", "runtime_policy",
        ],
    }
    target_rel = f"dataset/analysis/{result['experiment_id']}_manifest_v1.json"
    target = ROOT / target_rel
    write_json(target, result)
    for field in result["roster_amendment"]["unchanged_scientific_fields"]:
        if source.get(field) != result.get(field):
            raise ValueError(f"Scientific field changed in parity manifest: {name}/{field}")
    return target_rel, result


def main() -> int:
    built = {name: build(name, path) for name, path in SOURCES.items()}
    record = {
        "schema_version": "openai-parity-roster-amendment-v1",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "decision": "include researcher-verified gpt-5.4 as a fifth formal model",
        "researcher_confirmation": (
            "The researcher confirms that the route serves GPT-5.4 and directs "
            "equal data volume and experimental operations."
        ),
        "model": MODEL,
        "existing_gate_c_calls_retained": 864,
        "additional_calls": sum(
            value[1]["expected"]["total_calls"] for value in built.values()
        ),
        "total_calls_for_openai": 864 + sum(
            value[1]["expected"]["total_calls"] for value in built.values()
        ),
        "selection_rule": "No task unit was selected using GPT parity outcomes.",
        "manifests": {
            name: {
                "file": value[0],
                "sha256": sha256(ROOT / value[0]),
                "calls": value[1]["expected"]["total_calls"],
            }
            for name, value in built.items()
        },
    }
    write_json(ROOT / "dataset/analysis/openai_parity_roster_amendment_v1.json", record)
    print(json.dumps({"manifests": len(built), "additional_calls": record["additional_calls"], "total_openai": record["total_calls_for_openai"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
