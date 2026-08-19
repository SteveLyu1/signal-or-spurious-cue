#!/usr/bin/env python3
"""Freeze three prompt-identical sentinel waves for the formal run.

This builder is offline.  It selects records and writes manifests, but never
calls an LLM endpoint.  Each wave repeats 48 cached-core task units on all four
official providers.  The three waves use disjoint profiles and are intended to
run near the beginning, middle and end of formal inference.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl"
ANALYSIS_MANIFEST_PATH = ROOT / "dataset/analysis/confirmatory_manifest_v2.json"
LABEL_ASSIGNMENT_PATH = ROOT / "dataset/analysis/confirmatory_label_assignment_v1.json"
CORE_TASK_MANIFEST_PATH = (
    ROOT / "dataset/analysis/confirmatory_core_task_manifest_v2.json"
)
PROMPT_PATH = ROOT / "api_pilot/prompts/survey-confirmatory-v4.json"
GATE_STATUS_PATH = ROOT / "api_pilot/outputs/gate_c_v3_postprocess_status.json"
OUTPUT_DIR = ROOT / "dataset/analysis"
SUITE_PATH = OUTPUT_DIR / "confirmatory_sentinel_suite_v1.json"

SCHEMA_VERSION = "survey-task-manifest-v2"
SUITE_VERSION = "confirmatory-sentinel-suite-v1"
SEED = "aaai-confirmatory-runtime-sentinel-v1"
COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")
CELLS = (
    ("M0", "correct"),
    ("MO", "correct"),
    ("MR", "correct"),
    ("MV", "correct"),
)
OFFICIAL_PROVIDER_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.6-flash",
    "mistral": "mistral-medium-3-5",
    "qwen": "qwen3.7-max-2026-06-08",
}
WAVES = ("before", "mid", "after")
BATCH_SCHEDULE = {
    ("before", 1): {
        "CN": "B01", "US": "B01", "JO": "B02",
        "FR": "B02", "GB": "B03", "IT": "B04",
    },
    ("before", 2): {
        "CN": "B03", "US": "B04", "JO": "B03",
        "FR": "B04", "GB": "B01", "IT": "B02",
    },
    ("mid", 1): {
        "CN": "B02", "US": "B02", "JO": "B03",
        "FR": "B03", "GB": "B04", "IT": "B01",
    },
    ("mid", 2): {
        "CN": "B04", "US": "B01", "JO": "B04",
        "FR": "B01", "GB": "B02", "IT": "B03",
    },
    ("after", 1): {
        "CN": "B03", "US": "B03", "JO": "B04",
        "FR": "B04", "GB": "B01", "IT": "B02",
    },
    ("after", 2): {
        "CN": "B01", "US": "B02", "JO": "B01",
        "FR": "B02", "GB": "B03", "IT": "B04",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def stable_digest(*parts: str) -> bytes:
    return hashlib.sha256("\x1f".join((SEED, *parts)).encode()).digest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_profiles() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PROFILE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 504:
        raise ValueError(f"Expected 504 formal profiles, found {len(rows)}")
    if Counter(row["true_country_code"] for row in rows) != Counter(
        {country: 84 for country in COUNTRIES}
    ):
        raise ValueError("Formal profile panel is not balanced 6 x 84")
    if len({row["profile_id"] for row in rows}) != len(rows):
        raise ValueError("Formal profile IDs are not unique")
    return rows


def select_waves(
    profiles: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Select two country-label perfect matchings per disjoint wave.

    Each matching contributes one profile per true country and displayed label.
    Within a wave, the second matching gives each displayed label a different
    batch.  Selection uses no LLM outcome.
    """

    labels = sorted({row["balanced_random_country_label"] for row in profiles})
    if len(labels) != 6:
        raise ValueError("Expected six displayed labels")
    used_profiles: set[str] = set()
    selected: dict[str, list[dict[str, Any]]] = {}
    for wave in WAVES:
        wave_profiles: list[dict[str, Any]] = []
        wave_label_batches: dict[str, set[str]] = defaultdict(set)
        for round_index in (1, 2):
            candidates: list[tuple[bytes, list[dict[str, Any]]]] = []
            for permutation in itertools.permutations(labels):
                chosen: list[dict[str, Any]] = []
                for country, label in zip(COUNTRIES, permutation):
                    batch_id = BATCH_SCHEDULE[(wave, round_index)][country]
                    if batch_id in wave_label_batches[label]:
                        break
                    bucket = [
                        row for row in profiles
                        if row["true_country_code"] == country
                        and row["balanced_random_country_label"] == label
                        and row["profile_id"] not in used_profiles
                        and row["profile_id"] not in {
                            value["profile_id"] for value in wave_profiles
                        }
                    ]
                    if not bucket:
                        break
                    bucket.sort(
                        key=lambda row: stable_digest(
                            "profile", wave, str(round_index), country, label,
                            batch_id, row["profile_id"]
                        )
                    )
                    value = dict(bucket[0])
                    value["sentinel_batch_id"] = batch_id
                    value["sentinel_round"] = round_index
                    chosen.append(value)
                if len(chosen) == len(COUNTRIES):
                    candidates.append(
                        (
                            stable_digest(
                                "matching", wave, str(round_index),
                                *(row["profile_id"] for row in chosen),
                            ),
                            chosen,
                        )
                    )
            if not candidates:
                raise ValueError(
                    f"No balanced sentinel matching exists for {wave}/{round_index}"
                )
            chosen = min(candidates, key=lambda value: value[0])[1]
            wave_profiles.extend(chosen)
            for row in chosen:
                wave_label_batches[row["balanced_random_country_label"]].add(
                    row["sentinel_batch_id"]
                )
        selected[wave] = wave_profiles
        used_profiles.update(row["profile_id"] for row in wave_profiles)
        if any(len(value) != 2 for value in wave_label_batches.values()):
            raise ValueError(f"Displayed-label batch exposure is invalid in {wave}")
    if len(used_profiles) != 36:
        raise ValueError("Sentinel waves are not profile-disjoint")
    return selected


def wave_units(wave: str, profiles: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = [
        {
            "profile_id": row["profile_id"],
            "condition": condition,
            "evidence_variant": evidence,
            "batch_id": row["sentinel_batch_id"],
        }
        for row in profiles
        for condition, evidence in CELLS
    ]
    return sorted(
        result,
        key=lambda unit: stable_digest(
            "task", wave, unit["profile_id"], unit["condition"],
            unit["evidence_variant"], unit["batch_id"],
        ),
    )


def build_manifest(
    wave: str, profiles: list[dict[str, Any]], core_manifest_sha256: str
) -> dict[str, Any]:
    prompt = load_json(PROMPT_PATH)
    core_manifest = load_json(CORE_TASK_MANIFEST_PATH)
    units = wave_units(wave, profiles)
    experiment_id = f"confirmatory_sentinel_{wave}_v1"
    output_path = ROOT / f"api_pilot/restricted_outputs/{experiment_id}.jsonl"
    checkpoint_path = output_path.with_suffix(".checkpoint.sqlite3")
    batches = sorted({unit["batch_id"] for unit in units})
    batch_counts = Counter(unit["batch_id"] for unit in units)
    if len(units) != 48 or set(batch_counts.values()) != {12}:
        raise ValueError(f"Invalid task-unit balance for {wave}: {batch_counts}")
    return {
        "schema_version": SCHEMA_VERSION,
        "version": f"confirmatory-sentinel-{wave}-v1",
        "status": (
            "prospectively frozen runtime sentinel; offline definition only; "
            "execution requires the aggregate Gate C pass and respondent-data acknowledgement"
        ),
        "suite_version": SUITE_VERSION,
        "experiment_id": experiment_id,
        "task_mode": "repeat_identical",
        "profile_ids": sorted(row["profile_id"] for row in profiles),
        "cells": [list(cell) for cell in CELLS],
        "target_batches": batches,
        "profile_target_batches": {
            row["profile_id"]: core_manifest["profile_target_batches"][row["profile_id"]]
            for row in profiles
        },
        "task_units": units,
        "source": {
            "profile_file": relative(PROFILE_PATH),
            "profile_file_sha256": sha256(PROFILE_PATH),
            "profile_count": 504,
            "profiles_per_country": 84,
            "analysis_manifest_file": relative(ANALYSIS_MANIFEST_PATH),
            "analysis_manifest_sha256": sha256(ANALYSIS_MANIFEST_PATH),
            "label_assignment_file": relative(LABEL_ASSIGNMENT_PATH),
            "label_assignment_sha256": sha256(LABEL_ASSIGNMENT_PATH),
            "core_manifest_file": relative(CORE_TASK_MANIFEST_PATH),
            "core_manifest_sha256": core_manifest_sha256,
        },
        "prompt": {
            "file": relative(PROMPT_PATH),
            "version": prompt["version"],
            "sha256": sha256(PROMPT_PATH),
        },
        "provider_models": OFFICIAL_PROVIDER_MODELS,
        "runner_configuration": {
            "target_batch_size": 5,
            "reverse_target_options": False,
        },
        "selection": {
            "seed": SEED,
            "wave": wave,
            "intended_runtime_position": wave,
            "method": (
                "stable SHA-256 country-label perfect matching before formal outcomes; "
                "profiles are disjoint across waves"
            ),
            "profile_count": 12,
            "profiles_by_true_country": dict(
                sorted(Counter(row["true_country_code"] for row in profiles).items())
            ),
            "profiles_by_displayed_label": dict(
                sorted(
                    Counter(
                        row["balanced_random_country_label"] for row in profiles
                    ).items()
                )
            ),
            "batch_by_profile": {
                row["profile_id"]: row["sentinel_batch_id"]
                for row in sorted(profiles, key=lambda value: value["profile_id"])
            },
            "task_units_per_provider": 48,
        },
        "expected": {
            "providers": 4,
            "calls_per_provider": 48,
            "total_calls": 192,
        },
        "immutable_output_path": relative(output_path),
        "immutable_checkpoint_path": relative(checkpoint_path),
        "analysis_contract": {
            "role": "runtime drift sentinel, not a primary hypothesis test",
            "source_pairing": (
                "each repeated call must name the exact formal core call_id obtained "
                "from the same prompt, profile, cell, batch, provider and model"
            ),
            "primary_discrepancy": (
                "total variation after alignment by answer_code"
            ),
            "cluster_unit": "profile_id",
            "summaries": ["overall", "wave", "provider", "cell"],
            "core_results_required": 32_256,
            "core_experiment_id": "confirmatory_core_v2_raw",
            "all_three_waves_required": True,
            "execution_gate_status_file": relative(GATE_STATUS_PATH),
            "execution_gate_json_path": "gate.full_gate_c_pass",
            "execution_gate_required_value": True,
            "data_processing_acknowledgement_required": True,
            "result_probability_rule": (
                "reject sum error above 0.01; otherwise normalize a local analysis "
                "copy and retain the raw response"
            ),
        },
        "runtime_policy": core_manifest["runtime_policy"],
    }


def validate_suite(suite: dict[str, dict[str, Any]]) -> None:
    if set(suite) != set(WAVES):
        raise ValueError("Sentinel suite must contain before, mid and after")
    all_profiles = []
    batch_totals: Counter[str] = Counter()
    label_batches: dict[str, set[str]] = defaultdict(set)
    profiles = {row["profile_id"]: row for row in load_profiles()}
    for wave, manifest in suite.items():
        if manifest["expected"] != {
            "providers": 4, "calls_per_provider": 48, "total_calls": 192
        }:
            raise ValueError(f"Invalid expected counts for {wave}")
        if manifest["provider_models"] != OFFICIAL_PROVIDER_MODELS:
            raise ValueError("Official provider roster drifted")
        all_profiles.extend(manifest["profile_ids"])
        units = manifest["task_units"]
        if len(units) != len({
            (row["profile_id"], row["condition"], row["evidence_variant"], row["batch_id"])
            for row in units
        }):
            raise ValueError(f"Duplicate sentinel task unit in {wave}")
        by_profile = Counter(row["profile_id"] for row in units)
        if set(by_profile.values()) != {4}:
            raise ValueError(f"Every sentinel profile must cover all four cells in {wave}")
        for profile_id, batch_id in manifest["selection"]["batch_by_profile"].items():
            batch_totals[batch_id] += 1
            label_batches[profiles[profile_id]["balanced_random_country_label"]].add(
                batch_id
            )
    if len(all_profiles) != 36 or len(set(all_profiles)) != 36:
        raise ValueError("Sentinel profiles overlap across waves")
    if batch_totals != Counter({"B01": 9, "B02": 9, "B03": 9, "B04": 9}):
        raise ValueError(f"Unexpected cross-wave profile batch balance: {batch_totals}")
    if any(len(value) < 3 for value in label_batches.values()):
        raise ValueError("Each displayed label must receive at least three batches")


def build_suite() -> dict[str, dict[str, Any]]:
    core_hash = sha256(CORE_TASK_MANIFEST_PATH)
    selected = select_waves(load_profiles())
    suite = {
        wave: build_manifest(wave, selected[wave], core_hash)
        for wave in WAVES
    }
    validate_suite(suite)
    return suite


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suite = build_suite()
    summary: dict[str, Any] = {
        "version": SUITE_VERSION,
        "status": "offline task definitions frozen; no API request made",
        "seed": SEED,
        "provider_models": OFFICIAL_PROVIDER_MODELS,
        "core_manifest_file": relative(CORE_TASK_MANIFEST_PATH),
        "core_manifest_sha256": sha256(CORE_TASK_MANIFEST_PATH),
        "total_planned_calls": 576,
        "waves": {},
    }
    for wave, manifest in suite.items():
        path = OUTPUT_DIR / f"confirmatory_sentinel_{wave}_manifest_v1.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["waves"][wave] = {
            "manifest": relative(path),
            "manifest_sha256": sha256(path),
            **manifest["expected"],
            "immutable_output_path": manifest["immutable_output_path"],
            "immutable_checkpoint_path": manifest["immutable_checkpoint_path"],
        }
    SUITE_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"suite={relative(SUITE_PATH)} waves=3 calls=576 "
        f"sha256={sha256(SUITE_PATH)} api_requests_made=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
