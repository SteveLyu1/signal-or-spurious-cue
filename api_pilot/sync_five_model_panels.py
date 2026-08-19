#!/usr/bin/env python3
"""Build five-model manifests and merged JSONL panels without API calls."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "dataset/analysis/openai_parity_roster_amendment_v1.json"
SOURCE_BY_NAME = {
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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    latest = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["call_id"]] = row
    return list(latest.values())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(values, key=lambda value: value["call_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    amendment = load(AMENDMENT)
    summary = {}
    for name, source_rel in SOURCE_BY_NAME.items():
        source_path = ROOT / source_rel
        parity_path = ROOT / amendment["manifests"][name]["file"]
        source, parity = load(source_path), load(parity_path)
        for field in (
            "task_mode", "profile_ids", "cells", "target_batches", "task_units",
            "profile_target_batches", "source", "prompt", "runner_configuration",
            "selection", "analysis_contract", "runtime_policy",
        ):
            if source.get(field) != parity.get(field):
                raise ValueError(f"Parity scientific field mismatch: {name}/{field}")
        combined = deepcopy(source)
        combined["version"] = f"{source['version']}-five-model-v1"
        combined["status"] = "five-model roster amendment; offline union of identical task panels"
        combined["experiment_id"] = f"{source['experiment_id']}_five_model"
        combined["provider_models"] = {**source["provider_models"], **parity["provider_models"]}
        per_provider = int(source["expected"]["calls_per_provider"])
        combined["expected"] = {
            "providers": 5, "calls_per_provider": per_provider,
            "total_calls": per_provider * 5,
        }
        combined["immutable_output_path"] = (
            f"api_pilot/restricted_outputs/{combined['experiment_id']}.jsonl"
        )
        combined["immutable_checkpoint_path"] = (
            f"api_pilot/restricted_outputs/{combined['experiment_id']}.offline_union.sqlite3"
        )
        combined["roster_amendment"] = {
            **parity["roster_amendment"],
            "source_experiment_id": source["experiment_id"],
            "openai_source_experiment_id": parity["experiment_id"],
        }
        combined_path = ROOT / f"dataset/analysis/{combined['experiment_id']}_manifest_v1.json"
        atomic_json(combined_path, combined)

        source_rows = rows(ROOT / source["immutable_output_path"])
        parity_rows = rows(ROOT / parity["immutable_output_path"])
        complete = (
            len(source_rows) == source["expected"]["total_calls"]
            and len(parity_rows) == parity["expected"]["total_calls"]
            and all(row.get("success") for row in source_rows + parity_rows)
        )
        output_path = ROOT / combined["immutable_output_path"]
        if complete:
            merged = source_rows + parity_rows
            if len({row["call_id"] for row in merged}) != combined["expected"]["total_calls"]:
                raise ValueError(f"Duplicate or missing five-model calls: {name}")
            atomic_jsonl(output_path, merged)
        summary[name] = {
            "manifest": combined_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256(combined_path),
            "output": output_path.relative_to(ROOT).as_posix(),
            "complete": complete,
            "source_rows": len(source_rows), "openai_rows": len(parity_rows),
            "expected": combined["expected"]["total_calls"],
        }
    status = ROOT / "api_pilot/outputs/five_model_panel_sync.json"
    robustness = {
        "version": "gate-c-robustness-suite-five-model-v1",
        "provider_models": {"deepseek": "deepseek-v4-pro", "gemini": "gemini-3.6-flash", "mistral": "mistral-medium-3-5", "qwen": "qwen3.7-max-2026-06-08", "openai": "gpt-5.4"},
        "manifests": {
            name: {"file": summary[name]["manifest"], "sha256": summary[name]["manifest_sha256"]}
            for name in ("repeat_identical", "single_item", "wording_b", "option_reverse")
        },
    }
    robustness_path = ROOT / "dataset/analysis/gate_c_robustness_suite_five_model_v1.json"
    atomic_json(robustness_path, robustness)
    core = summary["confirmatory_core"]
    sentinel = {
        "version": "confirmatory-sentinel-suite-five-model-v1",
        "provider_models": robustness["provider_models"],
        "core_manifest_file": core["manifest"],
        "core_manifest_sha256": core["manifest_sha256"],
        "waves": {
            wave: {
                "manifest": summary[f"sentinel_{wave}"]["manifest"],
                "manifest_sha256": summary[f"sentinel_{wave}"]["manifest_sha256"],
            }
            for wave in ("before", "mid", "after")
        },
    }
    sentinel_path = ROOT / "dataset/analysis/confirmatory_sentinel_suite_five_model_v1.json"
    atomic_json(sentinel_path, sentinel)
    atomic_json(status, {"status": "complete" if all(v["complete"] for v in summary.values()) else "pending", "panels": summary, "robustness_suite": robustness_path.relative_to(ROOT).as_posix(), "sentinel_suite": sentinel_path.relative_to(ROOT).as_posix()})
    print(json.dumps({name: value["complete"] for name, value in summary.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
