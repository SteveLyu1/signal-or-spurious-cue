#!/usr/bin/env python3
"""Freeze the cached 504-profile confirmatory core v2 panel without API calls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl"
ANALYSIS_MANIFEST_PATH = ROOT / "dataset/analysis/confirmatory_manifest_v2.json"
LABEL_ASSIGNMENT_PATH = (
    ROOT / "dataset/analysis/confirmatory_label_assignment_v1.json"
)
DONOR_ASSIGNMENT_PATH = (
    ROOT / "dataset/analysis/confirmatory_donor_pair_assignment_v2.json"
)
PROMPT_PATH = ROOT / "api_pilot/prompts/survey-confirmatory-v4.json"
OUTPUT_PATH = ROOT / "dataset/analysis/confirmatory_core_task_manifest_v2.json"
RESULT_PATH = ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_raw.jsonl"
CHECKPOINT_PATH = (
    ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_raw.checkpoint.sqlite3"
)
GATE_STATUS_PATH = ROOT / "api_pilot/outputs/gate_c_v3_postprocess_status.json"

COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")
CELLS = (
    ("M0", "correct"),
    ("MO", "correct"),
    ("MR", "correct"),
    ("MV", "correct"),
)
LOGICAL_CELLS = CELLS + tuple((condition, "shuffled") for condition, _ in CELLS)
OFFICIAL_PROVIDER_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.6-flash",
    "mistral": "mistral-medium-3-5",
    "qwen": "qwen3.7-max-2026-06-08",
}
RUNTIME_AMENDMENT_PATH = (
    ROOT / "dataset/analysis/confirmatory_protocol_amendment_v3.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_profiles() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PROFILE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 504:
        raise ValueError(f"Expected 504 profiles, found {len(rows)}")
    if Counter(row["true_country_code"] for row in rows) != Counter(
        {country: 84 for country in COUNTRIES}
    ):
        raise ValueError("Confirmatory profiles are not balanced 6 x 84")
    if len({row["profile_id"] for row in rows}) != len(rows):
        raise ValueError("Confirmatory profile IDs are not unique")
    return rows


def balanced_profile_batches(
    profiles: list[dict[str, Any]], target_ids: list[str]
) -> dict[str, dict[str, list[str]]]:
    """Assign reciprocal pairs deterministic balanced five-item templates."""
    by_country: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for profile in profiles:
        by_country[profile["true_country_code"]][profile["pair_id"]].append(profile)
    assignments: dict[str, dict[str, list[str]]] = {}
    for country_index, country in enumerate(COUNTRIES):
        pairs = by_country[country]
        if len(pairs) != 42 or any(len(rows) != 2 for rows in pairs.values()):
            raise ValueError(f"Expected 42 reciprocal pairs in {country}")
        ordered_pairs = sorted(
            pairs,
            key=lambda pair_id: hashlib.sha256(
                f"balanced-batch-v3\x1f{country}\x1f{pair_id}".encode()
            ).digest(),
        )
        for rank, pair_id in enumerate(ordered_pairs):
            # Reciprocal pairs must share a template, so each exposure count is
            # even.  Country-shifted rotations achieve the mathematical optimum:
            # 100 or 102 records per answer position over 504 profiles.
            offset = (rank + country_index * 2) % 5
            ordered_targets = [
                target_ids[(position + offset) % 20]
                for position in range(20)
            ]
            batches = {
                f"B{batch + 1:02d}": ordered_targets[batch * 5:(batch + 1) * 5]
                for batch in range(4)
            }
            for profile in pairs[pair_id]:
                assignments[profile["profile_id"]] = batches
    if len(assignments) != len(profiles):
        raise ValueError("Balanced batch assignment is incomplete")
    return assignments


def build_manifest() -> dict[str, Any]:
    profiles = load_profiles()
    analysis = load_json(ANALYSIS_MANIFEST_PATH)
    assignment = load_json(LABEL_ASSIGNMENT_PATH)
    donor_assignment = load_json(DONOR_ASSIGNMENT_PATH)
    prompt = load_json(PROMPT_PATH)
    batches = analysis["target_batches"]
    if list(batches) != ["B01", "B02", "B03", "B04"]:
        raise ValueError("Confirmatory analysis manifest must freeze B01-B04")
    flat_targets = [item for batch in batches.values() for item in batch]
    if len(flat_targets) != 20 or len(set(flat_targets)) != 20:
        raise ValueError("Confirmatory analysis manifest must freeze 20 unique targets")
    profile_batches = balanced_profile_batches(profiles, flat_targets)
    expected_assignment = {
        row["profile_id"]: (
            row["true_country_code"], row["displayed_country_label"]
        )
        for row in assignment["assignments"]
    }
    if len(expected_assignment) != 504:
        raise ValueError("Label assignment must contain 504 unique profiles")
    for profile in profiles:
        target_ids = [row["item_id"] for row in profile["targets"]]
        if target_ids != flat_targets:
            raise ValueError(
                f"Profile target order differs from frozen batches: {profile['profile_id']}"
            )
        assigned = expected_assignment.get(profile["profile_id"])
        if assigned != (
            profile["true_country_code"],
            profile["balanced_random_country_label"],
        ):
            raise ValueError(
                f"Profile/assignment mismatch: {profile['profile_id']}"
            )
    donors = {row["profile_id"]: row for row in donor_assignment["assignments"]}
    profile_by_id = {row["profile_id"]: row for row in profiles}
    if len(donors) != 504:
        raise ValueError("Donor assignment must contain 504 unique profiles")
    for profile in profiles:
        frozen = donors.get(profile["profile_id"])
        donor = profile_by_id.get(profile.get("donor_profile_id"))
        if frozen is None or donor is None:
            raise ValueError(f"Missing reciprocal donor: {profile['profile_id']}")
        if frozen.get("pair_id") != profile.get("pair_id") or frozen.get(
            "donor_profile_id"
        ) != donor["profile_id"]:
            raise ValueError(f"Profile/donor manifest mismatch: {profile['profile_id']}")
        if (
            donor.get("donor_profile_id") != profile["profile_id"]
            or donor.get("pair_id") != profile["pair_id"]
            or donor["true_country_code"] != profile["true_country_code"]
            or donor["balanced_random_country_label"]
            != profile["balanced_random_country_label"]
            or donor["evidence"] != profile["shuffled_evidence"]
        ):
            raise ValueError(f"Invalid reciprocal donor pair: {profile['profile_id']}")
    task_units = [
        {
            "profile_id": profile["profile_id"],
            "condition": condition,
            "evidence_variant": evidence,
            "batch_id": batch_id,
        }
        for profile in sorted(profiles, key=lambda row: row["profile_id"])
        for condition, evidence in CELLS
        for batch_id in batches
    ]
    expected_units = 504 * len(CELLS) * len(batches)
    if len(task_units) != expected_units or len(
        {
            (
                row["profile_id"], row["condition"],
                row["evidence_variant"], row["batch_id"],
            )
            for row in task_units
        }
    ) != expected_units:
        raise ValueError("Confirmatory task units are incomplete or duplicated")
    return {
        "schema_version": "survey-task-manifest-v2",
        "version": "confirmatory-core-task-v2-cached",
        "status": (
            "frozen offline task definition; execution blocked until the aggregate "
            "Gate C status closes and respondent-data processing is acknowledged"
        ),
        "suite_version": "confirmatory-core-suite-v2-cached",
        "experiment_id": "confirmatory_core_v2_raw",
        "task_mode": "standard",
        "profile_ids": sorted(row["profile_id"] for row in profiles),
        "cells": [list(cell) for cell in CELLS],
        "logical_cells": [list(cell) for cell in LOGICAL_CELLS],
        "target_batches": list(batches),
        "profile_target_batches": profile_batches,
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
            "donor_assignment_file": relative(DONOR_ASSIGNMENT_PATH),
            "donor_assignment_sha256": sha256(DONOR_ASSIGNMENT_PATH),
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
            "method": "all 504 profiles and four correct-evidence API cells/batch",
            "profile_count": 504,
            "profiles_by_true_country": {country: 84 for country in COUNTRIES},
            "random_label_assignment_version": assignment["version"],
            "random_label_assignment_seed": assignment["seed"],
            "profiles_per_true_country_and_displayed_label": 14,
            "task_units_per_provider": expected_units,
            "reciprocal_pairs": 252,
            "pairs_per_true_country_and_displayed_label": 7,
        },
        "expected": {
            "providers": len(OFFICIAL_PROVIDER_MODELS),
            "calls_per_provider": expected_units,
            "total_calls": expected_units * len(OFFICIAL_PROVIDER_MODELS),
        },
        "immutable_output_path": relative(RESULT_PATH),
        "immutable_checkpoint_path": relative(CHECKPOINT_PATH),
        "analysis_contract": {
            "primary_stratum": "high_country_value",
            "primary_hypotheses": {
                "H1": "beta_O > 0 and beta_O - beta_R > 0",
                "H2": "verified-country Brier utility > 0",
                "multiplicity": "Holm across H1 and H2",
            },
            "sesoi_alternatives": {
                "beta_O": 0.05,
                "beta_O_minus_beta_R": 0.05,
                "verified_utility": 0.02,
            },
            "weighting": (
                "survey weights normalized within true country; equal country, "
                "target item and official-provider macro aggregation"
            ),
            "execution_gate_status_file": relative(GATE_STATUS_PATH),
            "execution_gate_json_path": "gate.full_gate_c_pass",
            "execution_gate_required_value": True,
            "data_processing_acknowledgement_required": True,
            "result_probability_rule": (
                "reject sum error above 0.01; otherwise normalize a local analysis "
                "copy to sum exactly one and retain the raw response"
            ),
            "cached_logical_panel": {
                "raw_api_cells": [list(cell) for cell in CELLS],
                "derived_cells": [
                    [condition, "shuffled"] for condition, _ in CELLS
                ],
                "logical_cells_total": 8,
                "materialization_rule": (
                    "reuse the reciprocal donor's correct-evidence result only "
                    "when reconstructed target-shuffled and donor-correct rendered "
                    "prompt bytes have identical SHA-256"
                ),
                "derived_output_path": (
                    "api_pilot/restricted_outputs/confirmatory_core_v2_logical.jsonl"
                ),
            },
        },
        "runtime_policy": {
            "version": "complete-through-report-v1",
            "scientific_gate_action": "report_only",
            "failed_call_action": "defer_then_retry",
            "protocol_amendment_file": relative(RUNTIME_AMENDMENT_PATH),
            "protocol_amendment_sha256": sha256(RUNTIME_AMENDMENT_PATH),
        },
        "prospective_amendment": {
            "version": "confirmatory-cached-core-amendment-v2",
            "frozen_at": "2026-07-25T16:40:00+08:00",
            "confirmatory_outcomes_observed": False,
            "change": (
                "replace four redundant shuffled-evidence API cells with "
                "fail-closed reciprocal-donor cache materialization"
            ),
            "logical_estimands_unchanged": True,
            "raw_calls_reduced_from": 48_384,
            "raw_calls_reduced_to": 32_256,
        },
    }


def main() -> int:
    manifest = build_manifest()
    OUTPUT_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": relative(OUTPUT_PATH),
                "sha256": sha256(OUTPUT_PATH),
                "profiles": len(manifest["profile_ids"]),
                "calls_per_provider": manifest["expected"]["calls_per_provider"],
                "total_calls": manifest["expected"]["total_calls"],
                "api_requests_made": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
