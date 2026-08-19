#!/usr/bin/env python3
"""Build the offline Gate C robustness-suite manifests.

The builder reads no LLM outcomes beyond hashing the immutable core export.  It
selects task units from the frozen Gate C profiles with stable hash ordering and
binds every auxiliary run to an exact prompt, provider/model roster and output
path.  Running this file never makes a network request.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "dataset" / "derived" / "gate_c_profiles_6x12.jsonl"
CORE_MANIFEST_PATH = ROOT / "dataset" / "analysis" / "gate_c_manifest_v1.json"
CORE_OUTPUT_PATH = ROOT / "api_pilot" / "restricted_outputs" / "gate_c_v3_complete.jsonl"
PROMPT_A_PATH = ROOT / "api_pilot" / "prompts" / "survey-confirmatory-v3.json"
PROMPT_V4_PATH = ROOT / "api_pilot" / "prompts" / "survey-confirmatory-v4.json"
OUTPUT_DIR = ROOT / "dataset" / "analysis"
AMENDMENT_PATH = OUTPUT_DIR / "gate_c_robustness_amendment_v2.json"

SCHEMA_VERSION = "survey-task-manifest-v2"
# V2 is a prospective amendment.  It was frozen before any robustness JSONL
# outcome existed and leaves the completed v3 Gate C core export untouched.
SUITE_VERSION = "gate-c-robustness-suite-v2"
SEED = "aaai-gate-c-robustness-v1"
COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")
CORE_BATCHES = ("B01", "B02")
CORE_CELLS = (
    ("M0", "correct"),
    ("MO", "correct"),
    ("MR", "correct"),
    ("MV", "correct"),
    ("M0", "shuffled"),
    ("MR", "shuffled"),
)
OFFICIAL_PROVIDER_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.6-flash",
    "mistral": "mistral-medium-3-5",
    "qwen": "qwen3.7-max-2026-06-08",
}
EQUIVALENCE_TV_MARGINS = (0.02, 0.05, 0.10)
GATE_TV_MARGIN = 0.05
GATE_DIRECTION_AGREEMENT_LOWER_BOUND = 0.75


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
    if len(rows) != 72 or Counter(row["true_country_code"] for row in rows) != Counter(
        {country: 12 for country in COUNTRIES}
    ):
        raise ValueError("Gate C profile panel is not the expected balanced 6 x 12 panel")
    if len({row["profile_id"] for row in rows}) != len(rows):
        raise ValueError("Gate C profile IDs are not unique")
    return rows


def _matching_round(
    profiles: list[dict[str, Any]], used: set[str], round_index: int
) -> list[dict[str, Any]]:
    """Choose one profile per true country and displayed label.

    Only 6! label permutations are examined.  The objective is a stable hash,
    not an LLM outcome, and used profiles are excluded between rounds.
    """

    labels = sorted({row["balanced_random_country_label"] for row in profiles})
    if len(labels) != len(COUNTRIES):
        raise ValueError("Expected six displayed country labels")
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for country in COUNTRIES:
        for label in labels:
            bucket = [
                row
                for row in profiles
                if row["true_country_code"] == country
                and row["balanced_random_country_label"] == label
                and row["profile_id"] not in used
            ]
            bucket.sort(
                key=lambda row: stable_digest(
                    "profile", str(round_index), country, label, row["profile_id"]
                )
            )
            by_pair[(country, label)] = bucket
    candidates = []
    for permutation in itertools.permutations(labels):
        chosen = []
        for country, label in zip(COUNTRIES, permutation):
            bucket = by_pair[(country, label)]
            if not bucket:
                break
            chosen.append(bucket[0])
        if len(chosen) == len(COUNTRIES):
            score = stable_digest(
                "matching",
                str(round_index),
                *(row["profile_id"] for row in chosen),
            )
            candidates.append((score, chosen))
    if not candidates:
        raise ValueError(f"No balanced matching available for round {round_index}")
    return min(candidates, key=lambda value: value[0])[1]


def balanced_profiles(profiles: list[dict[str, Any]], rounds: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for round_index in range(1, rounds + 1):
        chosen = _matching_round(profiles, used, round_index)
        selected.extend(chosen)
        used.update(row["profile_id"] for row in chosen)
    return selected


def units(
    profiles: Iterable[dict[str, Any]],
    cells: Iterable[tuple[str, str]],
    batches: Iterable[str],
) -> list[dict[str, str]]:
    result = [
        {
            "profile_id": row["profile_id"],
            "condition": condition,
            "evidence_variant": evidence,
            "batch_id": batch,
        }
        for row in profiles
        for condition, evidence in cells
        for batch in batches
    ]
    return sorted(
        result,
        key=lambda unit: stable_digest(
            "unit",
            unit["profile_id"],
            unit["condition"],
            unit["evidence_variant"],
            unit["batch_id"],
        ),
    )


def selection_audit(
    selected: list[dict[str, Any]], task_units: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "seed": SEED,
        "method": "stable SHA-256 ordering before any robustness outcomes",
        "profile_count": len(selected),
        "profile_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["profile_id"] for row in selected)).encode()
        ).hexdigest(),
        "profiles_by_true_country": dict(
            sorted(Counter(row["true_country_code"] for row in selected).items())
        ),
        "profiles_by_displayed_label": dict(
            sorted(
                Counter(row["balanced_random_country_label"] for row in selected).items()
            )
        ),
        "task_units_per_provider": len(task_units),
    }


def build_manifest(
    *,
    name: str,
    experiment_id: str,
    task_mode: str,
    selected: list[dict[str, Any]],
    task_units: list[dict[str, str]],
    prompt_path: Path,
    target_batch_size: int,
    reverse_target_options: bool,
    output_name: str,
    item_to_core_batch: dict[str, str] | None = None,
    gate_forecast_tv_overall: bool = True,
    gate_treatment_contrasts: tuple[str, ...] = (),
) -> dict[str, Any]:
    prompt = load_json(prompt_path)
    output_path = ROOT / "api_pilot" / "restricted_outputs" / output_name
    checkpoint_path = output_path.with_suffix(".checkpoint.sqlite3")
    cells = sorted(
        {(
            unit["condition"],
            unit["evidence_variant"],
        ) for unit in task_units}
    )
    batches = sorted({unit["batch_id"] for unit in task_units})
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "version": name,
        "status": (
            "frozen Gate C prerequisite offline; no robustness LLM outcome inspected; "
            "does not replace the formal 504-profile auxiliary manifests"
        ),
        "suite_version": SUITE_VERSION,
        "experiment_id": experiment_id,
        "task_mode": task_mode,
        "profile_ids": sorted(row["profile_id"] for row in selected),
        "cells": [list(cell) for cell in cells],
        "target_batches": batches,
        "task_units": task_units,
        "source": {
            "profile_file": relative(PROFILE_PATH),
            "profile_file_sha256": sha256(PROFILE_PATH),
            "profile_count": 72,
            "profiles_per_country": 12,
            "core_manifest_file": relative(CORE_MANIFEST_PATH),
            "core_manifest_sha256": sha256(CORE_MANIFEST_PATH),
            "core_output_file": relative(CORE_OUTPUT_PATH),
            "core_output_sha256": sha256(CORE_OUTPUT_PATH),
        },
        "prompt": {
            "file": relative(prompt_path),
            "version": prompt["version"],
            "sha256": sha256(prompt_path),
        },
        "provider_models": OFFICIAL_PROVIDER_MODELS,
        "runner_configuration": {
            "target_batch_size": target_batch_size,
            "reverse_target_options": reverse_target_options,
        },
        "selection": selection_audit(selected, task_units),
        "expected": {
            "providers": len(OFFICIAL_PROVIDER_MODELS),
            "calls_per_provider": len(task_units),
            "total_calls": len(task_units) * len(OFFICIAL_PROVIDER_MODELS),
        },
        "immutable_output_path": relative(output_path),
        "immutable_checkpoint_path": relative(checkpoint_path),
        "analysis_contract": {
            "pairing_unit": "provider, profile, condition, evidence, item",
            "primary_discrepancy": "total variation after alignment by answer code",
            "cluster_unit": "profile_id",
            "equivalence_tv_margin_grid": list(EQUIVALENCE_TV_MARGINS),
            "gate_margin": GATE_TV_MARGIN,
            "gate_direction_agreement_lower_bound": (
                GATE_DIRECTION_AGREEMENT_LOWER_BOUND
            ),
            "gate_forecast_tv_overall": gate_forecast_tv_overall,
            "gate_treatment_contrasts": list(gate_treatment_contrasts),
            "gate_rule": (
                "for every listed treatment contrast, the country-stratified "
                "profile-bootstrap 95% CI upper bound for discrepancy must be "
                "below 0.05 and the 95% CI lower bound for direction agreement "
                "must exceed 0.75; forecast TV is additionally gated only when "
                "gate_forecast_tv_overall is true"
            ),
            "aggregation": (
                "survey-weighted within true survey country, followed by equal "
                "country, item/condition and official-provider macro averaging"
            ),
            "bootstrap": (
                "profile resampling stratified within true survey country with "
                "shared draws across providers, conditions and contrasts"
            ),
            "design_estimand_stability": {
                "implemented_in_this_gate": False,
                "not_directly_reestimated": [
                    "beta_O",
                    "beta_O_minus_beta_R",
                    "UtilityV",
                ],
                "restriction": (
                    "TV and treatment-contrast checks are measurement diagnostics; "
                    "they cannot establish invariance of the formal estimands"
                ),
            },
            "execution_gate_status_file": (
                "api_pilot/outputs/gate_c_v3_postprocess_status.json"
            ),
            "execution_gate_json_path": "data_processing_permission.passes",
            "execution_gate_required_value": True,
            "data_processing_acknowledgement_required": True,
        },
    }
    if item_to_core_batch is not None:
        manifest["item_to_core_batch"] = item_to_core_batch
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unexpected robustness manifest schema")
    units_seen = {
        (
            unit["profile_id"],
            unit["condition"],
            unit["evidence_variant"],
            unit["batch_id"],
        )
        for unit in manifest["task_units"]
    }
    if len(units_seen) != len(manifest["task_units"]):
        raise ValueError(f"Duplicate task unit in {manifest['version']}")
    if manifest["expected"]["calls_per_provider"] != len(units_seen):
        raise ValueError("Expected per-provider call count does not match task units")
    expected_total = len(units_seen) * len(manifest["provider_models"])
    if manifest["expected"]["total_calls"] != expected_total:
        raise ValueError("Expected total call count does not match provider roster")
    if manifest["source"]["profile_file_sha256"] != sha256(PROFILE_PATH):
        raise ValueError("Profile file changed while building robustness suite")
    if manifest["source"]["core_manifest_sha256"] != sha256(CORE_MANIFEST_PATH):
        raise ValueError("Core manifest changed while building robustness suite")
    if manifest["source"]["core_output_sha256"] != sha256(CORE_OUTPUT_PATH):
        raise ValueError("Core output changed while building robustness suite")
    if manifest["prompt"]["sha256"] != sha256(ROOT / manifest["prompt"]["file"]):
        raise ValueError("Prompt changed while building robustness suite")


def build_suite() -> dict[str, dict[str, Any]]:
    profiles = load_profiles()
    repeat_profiles = balanced_profiles(profiles, rounds=1)
    compact_profiles = balanced_profiles(profiles, rounds=2)
    reverse_profiles = balanced_profiles(profiles, rounds=4)
    all_profiles = sorted(profiles, key=lambda row: row["profile_id"])
    core_manifest = load_json(CORE_MANIFEST_PATH)
    item_to_core_batch = {
        item_id: batch_id
        for batch_id, item_ids in core_manifest["target_batches"].items()
        for item_id in item_ids
    }
    single_batches = [f"B{index:02d}" for index in range(1, 11)]
    ordered_items = [
        item_id
        for batch_id in CORE_BATCHES
        for item_id in core_manifest["target_batches"][batch_id]
    ]
    single_item_map = {
        f"B{index:02d}": item_id
        for index, item_id in enumerate(ordered_items, start=1)
    }
    suite = {
        "gate_c_repeat_identical_manifest_v2.json": build_manifest(
            name="gate-c-repeat-identical-v2",
            experiment_id="repeat_identical",
            task_mode="repeat_identical",
            selected=repeat_profiles,
            task_units=units(repeat_profiles, CORE_CELLS, CORE_BATCHES),
            prompt_path=PROMPT_A_PATH,
            target_batch_size=5,
            reverse_target_options=False,
            output_name="gate_c_robustness_repeat_v2.jsonl",
        ),
        "gate_c_single_item_manifest_v2.json": build_manifest(
            name="gate-c-single-item-v2",
            experiment_id="single_item",
            task_mode="standard",
            selected=compact_profiles,
            task_units=units(
                compact_profiles,
                (
                    ("M0", "correct"),
                    ("MO", "correct"),
                    ("MR", "correct"),
                    ("MV", "correct"),
                ),
                single_batches,
            ),
            prompt_path=PROMPT_A_PATH,
            target_batch_size=1,
            reverse_target_options=False,
            output_name="gate_c_robustness_single_item_v2.jsonl",
            item_to_core_batch=item_to_core_batch,
            gate_treatment_contrasts=("MO_minus_MR", "MV_minus_M0"),
        ),
        "gate_c_prompt_v4_transfer_manifest_v2.json": build_manifest(
            name="gate-c-prompt-v4-transfer-v2",
            experiment_id="wording_b",
            task_mode="standard",
            selected=all_profiles,
            task_units=units(
                all_profiles,
                (("MO", "correct"), ("MR", "correct")),
                CORE_BATCHES,
            ),
            prompt_path=PROMPT_V4_PATH,
            target_batch_size=5,
            reverse_target_options=False,
            output_name="gate_c_robustness_prompt_v4_transfer_v2.jsonl",
            gate_forecast_tv_overall=False,
            gate_treatment_contrasts=("MO_minus_MR",),
        ),
        "gate_c_option_reverse_manifest_v2.json": build_manifest(
            name="gate-c-option-reverse-v2",
            experiment_id="option_reverse",
            task_mode="standard",
            selected=reverse_profiles,
            task_units=units(reverse_profiles, CORE_CELLS, CORE_BATCHES),
            prompt_path=PROMPT_A_PATH,
            target_batch_size=5,
            reverse_target_options=True,
            output_name="gate_c_robustness_option_reverse_v2.jsonl",
            gate_treatment_contrasts=("MO_minus_MR", "MV_minus_M0"),
        ),
    }
    for manifest in suite.values():
        if manifest["version"] == "gate-c-single-item-v2":
            manifest["single_item_batch_to_item"] = single_item_map
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
        "prospective_amendment": {
            "file": relative(AMENDMENT_PATH),
        },
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
    amendment = {
        "schema_version": "gate-c-robustness-amendment-v1",
        "version": SUITE_VERSION,
        "frozen_at_utc": "2026-07-25T02:40:15Z",
        "status": "prospectively frozen before any robustness JSONL outcome existed",
        "supersedes": {
            "suite": "dataset/analysis/gate_c_robustness_suite_v1.json",
            "suite_sha256_at_amendment": (
                "ec8aef82621f4218d5567f645a457d8c8c03d0d0bc6be7552d6418b95c14176f"
            ),
            "observed_robustness_jsonl_outputs": 0,
        },
        "changes": [
            {
                "component": "option_reverse",
                "before": "72 balanced profiles and 3456 calls",
                "after": "24 balanced profiles and 1152 calls",
                "reason": (
                    "retain four profiles per true country and per displayed label "
                    "while avoiding a full-panel auxiliary run"
                ),
            },
            {
                "component": "prompt_transfer",
                "before": "alternate v3 wording",
                "after": "frozen v4 MO/MR prompt on all 72 profiles",
                "reason": "test transfer from the completed v3 core to the formal v4 prompt",
            },
            {
                "component": "analysis",
                "before": "pooled row mean and unstratified profile bootstrap",
                "after": (
                    "survey-weighted within country, equal macro averaging, and "
                    "country-stratified shared-draw bootstrap"
                ),
                "reason": "match the pre-specified estimand hierarchy",
            },
            {
                "component": "direction_gate",
                "before": "no direction-agreement confidence-bound criterion",
                "after": (
                    "treatment discrepancy 95% CI upper below 0.05 and weighted "
                    "direction-agreement 95% CI lower above 0.75"
                ),
                "reason": (
                    "require at least three-quarters directional retention even at "
                    "the lower confidence limit, rather than allowing low TV alone "
                    "to define protocol stability"
                ),
            },
        ],
        "call_totals": {"v1": 6816, "v2": 4512},
        "note": (
            "The pre-existing empty v1 checkpoint is an engineering dry-run artifact; "
            "v2 uses new immutable output and checkpoint paths."
        ),
    }
    AMENDMENT_PATH.write_text(
        json.dumps(amendment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["prospective_amendment"]["sha256"] = sha256(AMENDMENT_PATH)
    summary["total_planned_calls"] = sum(
        row["total_calls"] for row in summary["manifests"].values()
    )
    summary_path = OUTPUT_DIR / "gate_c_robustness_suite_v2.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"suite={relative(summary_path)} sha256={sha256(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
