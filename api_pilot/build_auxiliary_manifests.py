#!/usr/bin/env python3
"""Freeze the formal 72-profile confirmatory auxiliary tasks without API calls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl"
ANALYSIS_MANIFEST_PATH = ROOT / "dataset/analysis/confirmatory_manifest_v2.json"
LABEL_ASSIGNMENT_PATH = ROOT / "dataset/analysis/confirmatory_label_assignment_v1.json"
CORE_TASK_MANIFEST_PATH = (
    ROOT / "dataset/analysis/confirmatory_core_task_manifest_v1.json"
)
CORE_PROMPT_PATH = ROOT / "api_pilot/prompts/survey-confirmatory-v3.json"
WORDING_PROMPT_PATH = (
    ROOT / "api_pilot/prompts/survey-confirmatory-v3-wording-b.json"
)
GATE_STATUS_PATH = ROOT / "api_pilot/outputs/gate_c_v3_postprocess_status.json"
OUTPUT_DIR = ROOT / "dataset/analysis"

SCHEMA_VERSION = "survey-task-manifest-v2"
SUITE_VERSION = "confirmatory-auxiliary-suite-v2"
SEED = "aaai-country-metadata-auxiliary-v1"
COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")
OFFICIAL_PROVIDER_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.6-flash",
    "mistral": "mistral-medium-3-5",
    "qwen": "qwen3.7-max-2026-06-08",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_order(profile_id: str) -> bytes:
    return hashlib.sha256(f"{SEED}\x1f{profile_id}".encode()).digest()


def load_profiles() -> list[dict[str, Any]]:
    profiles = [
        json.loads(line)
        for line in PROFILE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(profiles) != 504:
        raise ValueError(f"Expected 504 confirmatory profiles, found {len(profiles)}")
    if Counter(row["true_country_code"] for row in profiles) != Counter(
        {country: 84 for country in COUNTRIES}
    ):
        raise ValueError("Confirmatory profiles are not balanced 6 x 84")
    if len({row["profile_id"] for row in profiles}) != len(profiles):
        raise ValueError("Confirmatory profile IDs are not unique")
    analysis = load_json(ANALYSIS_MANIFEST_PATH)
    target_ids = [
        item_id
        for batch in analysis["target_batches"].values()
        for item_id in batch
    ]
    if len(target_ids) != 20 or len(set(target_ids)) != 20:
        raise ValueError("Confirmatory analysis manifest must freeze 20 targets")
    assignment_rows = load_json(LABEL_ASSIGNMENT_PATH)["assignments"]
    assignments = {
        row["profile_id"]: (
            row["true_country_code"], row["displayed_country_label"]
        )
        for row in assignment_rows
    }
    if len(assignments) != len(assignment_rows) or len(assignments) != 504:
        raise ValueError("Confirmatory label assignment is incomplete or duplicated")
    for profile in profiles:
        if assignments.get(profile["profile_id"]) != (
            profile["true_country_code"],
            profile["balanced_random_country_label"],
        ):
            raise ValueError(
                f"Profile/label assignment mismatch: {profile['profile_id']}"
            )
        if [target["item_id"] for target in profile["targets"]] != target_ids:
            raise ValueError(
                f"Profile target order differs from analysis manifest: "
                f"{profile['profile_id']}"
            )
    return profiles


def select_profiles(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select two profiles from every true-country x displayed-label cell."""
    selected: list[dict[str, Any]] = []
    for country in COUNTRIES:
        labels = sorted(
            {
                row["balanced_random_country_label"]
                for row in profiles
                if row["true_country_code"] == country
            }
        )
        if len(labels) != 6:
            raise ValueError(f"Expected six displayed labels in {country}")
        for label in labels:
            candidates = sorted(
                (
                    row
                    for row in profiles
                    if row["true_country_code"] == country
                    and row["balanced_random_country_label"] == label
                ),
                key=lambda row: stable_order(row["profile_id"]),
            )
            if len(candidates) != 14:
                raise ValueError(
                    f"Expected 14 profiles in {country}/{label}, found {len(candidates)}"
                )
            selected.extend(candidates[:2])
    selected.sort(key=lambda row: row["profile_id"])
    if len(selected) != 72 or len({row["profile_id"] for row in selected}) != 72:
        raise ValueError("Auxiliary selection is not 72 unique profiles")
    return selected


def build_units(
    selected: Iterable[dict[str, Any]],
    cells: Iterable[tuple[str, str]],
    batches: Iterable[str],
) -> list[dict[str, str]]:
    units = [
        {
            "profile_id": row["profile_id"],
            "condition": condition,
            "evidence_variant": evidence,
            "batch_id": batch_id,
        }
        for row in selected
        for condition, evidence in cells
        for batch_id in batches
    ]
    keys = {
        (
            row["profile_id"],
            row["condition"],
            row["evidence_variant"],
            row["batch_id"],
        )
        for row in units
    }
    if len(keys) != len(units):
        raise ValueError("Auxiliary task units are duplicated")
    return units


def selection_audit(
    selected: list[dict[str, Any]], task_units: list[dict[str, str]]
) -> dict[str, Any]:
    pair_counts = Counter(
        (row["true_country_code"], row["balanced_random_country_label"])
        for row in selected
    )
    if set(pair_counts.values()) != {2} or len(pair_counts) != 36:
        raise ValueError("Auxiliary selection is not balanced over all 36 label cells")
    return {
        "seed": SEED,
        "method": (
            "stable SHA-256 order, selecting two profiles in every true-country "
            "x displayed-label cell before auxiliary outcomes"
        ),
        "profile_count": 72,
        "profile_ids_sha256": hashlib.sha256(
            "\n".join(row["profile_id"] for row in selected).encode()
        ).hexdigest(),
        "profiles_by_true_country": dict(
            sorted(Counter(row["true_country_code"] for row in selected).items())
        ),
        "profiles_by_displayed_label": dict(
            sorted(
                Counter(
                    row["balanced_random_country_label"] for row in selected
                ).items()
            )
        ),
        "profiles_per_true_country_and_displayed_label": 2,
        "task_units_per_provider": len(task_units),
    }


def build_manifest(
    selected: list[dict[str, Any]],
    *,
    version: str,
    experiment_id: str,
    cells: tuple[tuple[str, str], ...],
    target_batches: tuple[str, ...],
    prompt_path: Path,
    result_path: Path,
    analysis_contract: dict[str, Any],
) -> dict[str, Any]:
    prompt = load_json(prompt_path)
    analysis = load_json(ANALYSIS_MANIFEST_PATH)
    if any(batch_id not in analysis["target_batches"] for batch_id in target_batches):
        raise ValueError("Auxiliary manifest references an unfrozen target batch")
    task_units = build_units(selected, cells, target_batches)
    checkpoint_path = result_path.with_suffix(".checkpoint.sqlite3")
    return {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "status": (
            "frozen offline task definition; execution blocked until aggregate "
            "Gate C passes and respondent-data processing is acknowledged"
        ),
        "suite_version": SUITE_VERSION,
        "experiment_id": experiment_id,
        "task_mode": "standard",
        "profile_ids": [row["profile_id"] for row in selected],
        "cells": [list(cell) for cell in cells],
        "target_batches": list(target_batches),
        "task_units": task_units,
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
            "core_manifest_sha256": sha256(CORE_TASK_MANIFEST_PATH),
        },
        "prompt": {
            "file": relative(prompt_path),
            "version": prompt["version"],
            "sha256": sha256(prompt_path),
        },
        "provider_models": OFFICIAL_PROVIDER_MODELS,
        "runner_configuration": {
            "target_batch_size": 5,
            "reverse_target_options": False,
        },
        "selection": selection_audit(selected, task_units),
        "expected": {
            "providers": len(OFFICIAL_PROVIDER_MODELS),
            "calls_per_provider": len(task_units),
            "total_calls": len(task_units) * len(OFFICIAL_PROVIDER_MODELS),
        },
        "immutable_output_path": relative(result_path),
        "immutable_checkpoint_path": relative(checkpoint_path),
        "analysis_contract": {
            **analysis_contract,
            "execution_gate_status_file": relative(GATE_STATUS_PATH),
            "execution_gate_json_path": "gate.full_gate_c_pass",
            "execution_gate_required_value": True,
            "data_processing_acknowledgement_required": True,
            "result_probability_rule": (
                "reject sum error above 0.01; otherwise normalize a local analysis "
                "copy to sum exactly one and retain the raw response"
            ),
        },
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    expected_units = len(manifest["profile_ids"]) * len(manifest["cells"]) * len(
        manifest["target_batches"]
    )
    if len(manifest["task_units"]) != expected_units:
        raise ValueError("Auxiliary task grid is incomplete")
    if manifest["expected"]["calls_per_provider"] != expected_units:
        raise ValueError("Auxiliary per-provider count is inconsistent")
    if manifest["expected"]["total_calls"] != expected_units * 4:
        raise ValueError("Auxiliary total call count is inconsistent")
    if manifest["provider_models"] != OFFICIAL_PROVIDER_MODELS:
        raise ValueError("Auxiliary official provider roster drifted")


def build_suite() -> dict[str, dict[str, Any]]:
    selected = select_profiles(load_profiles())
    suite = {
        "confirmatory_mechanism_manifest_v1.json": build_manifest(
            selected,
            version="confirmatory-mechanism-task-v2",
            experiment_id="confirmatory_mechanism_v1",
            cells=(("MO", "shuffled"), ("MV", "shuffled")),
            target_batches=("B01", "B02", "B03", "B04"),
            prompt_path=CORE_PROMPT_PATH,
            result_path=(
                ROOT / "api_pilot/restricted_outputs/confirmatory_mechanism_v1.jsonl"
            ),
            analysis_contract={
                "role": "secondary mechanism completion",
                "pair_with_core_cells": [
                    ["M0", "correct"],
                    ["M0", "shuffled"],
                    ["MO", "correct"],
                    ["MR", "correct"],
                    ["MR", "shuffled"],
                    ["MV", "correct"],
                ],
                "estimands": [
                    "G_O = Brier(MO-shuffled) - Brier(MO-correct)",
                    "G_V = Brier(MV-shuffled) - Brier(MV-correct)",
                    "metadata-by-evidence interactions across M0, MO, MR and MV",
                ],
                "cluster_unit": "profile_id",
            },
        ),
        "confirmatory_wording_b_manifest_v1.json": build_manifest(
            selected,
            version="confirmatory-wording-b-task-v2",
            experiment_id="confirmatory_wording_b_v1",
            cells=(("MO", "correct"), ("MR", "correct")),
            target_batches=("B01", "B02"),
            prompt_path=WORDING_PROMPT_PATH,
            result_path=(
                ROOT / "api_pilot/restricted_outputs/confirmatory_wording_b_v1.jsonl"
            ),
            analysis_contract={
                "role": "secondary equivalent-wording robustness",
                "pair_with_core_prompt_version": "survey-confirmatory-v3",
                "pairing_unit": "provider, profile, condition, evidence, item",
                "primary_discrepancy": (
                    "total variation after alignment by answer code"
                ),
                "contrast_checks": [
                    "MO minus M0 country-directed shift stability",
                    "MO minus MR disclosure attenuation stability",
                ],
                "cluster_unit": "profile_id",
            },
        ),
    }
    for manifest in suite.values():
        validate_manifest(manifest)
    return suite


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suite = build_suite()
    summary = {
        "version": SUITE_VERSION,
        "status": "offline task definitions only; no API request made",
        "seed": SEED,
        "provider_models": OFFICIAL_PROVIDER_MODELS,
        "manifests": {},
    }
    for filename, manifest in suite.items():
        path = OUTPUT_DIR / filename
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["manifests"][manifest["experiment_id"]] = {
            "file": relative(path),
            "sha256": sha256(path),
            **manifest["expected"],
            "immutable_output_path": manifest["immutable_output_path"],
            "immutable_checkpoint_path": manifest["immutable_checkpoint_path"],
        }
        print(
            f"{manifest['experiment_id']}: "
            f"{manifest['expected']['calls_per_provider']} calls/provider, "
            f"{manifest['expected']['total_calls']} total"
        )
    selection = next(iter(suite.values()))["selection"]
    audit = {
        "version": "confirmatory-auxiliary-selection-audit-v2",
        "seed": SEED,
        **selection,
        "label_assignment_file": relative(LABEL_ASSIGNMENT_PATH),
        "label_assignment_file_sha256": sha256(LABEL_ASSIGNMENT_PATH),
        "analysis_manifest_file": relative(ANALYSIS_MANIFEST_PATH),
        "analysis_manifest_sha256": sha256(ANALYSIS_MANIFEST_PATH),
    }
    audit_path = OUTPUT_DIR / "confirmatory_auxiliary_selection_audit_v1.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_path = OUTPUT_DIR / "confirmatory_auxiliary_suite_v2.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"suite={relative(summary_path)} sha256={sha256(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
