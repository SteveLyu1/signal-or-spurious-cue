#!/usr/bin/env python3
"""Build the frozen post-review checklist-wording correction supplement."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILES = ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl"
SOURCE_ANALYSIS = ROOT / "dataset/analysis/confirmatory_manifest_v2.json"
SOURCE_TASK = ROOT / "dataset/analysis/confirmatory_core_v2_raw_five_model_manifest_v1.json"
SOURCE_PROMPT = ROOT / "api_pilot/prompts/survey-confirmatory-v4.json"

PROMPT_OUT = ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json"
PROFILES_OUT = ROOT / "api_pilot/restricted_outputs/checklist_correction_profiles_6x12.jsonl"
MANIFEST_OUT = ROOT / "dataset/analysis/checklist_correction_supplement_manifest_v1.json"
AMENDMENT_OUT = ROOT / "dataset/analysis/checklist_conversion_amendment_v1.json"
REVIEW_OUT = ROOT / "api_pilot/human_review/checklist_correction_review.csv"
RESULT_OUT = ROOT / "api_pilot/restricted_outputs/checklist_correction_supplement_v1.jsonl"
CHECKPOINT_OUT = ROOT / "api_pilot/restricted_outputs/checklist_correction_supplement_v1.checkpoint.sqlite3"

COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")
CELLS = (("M0", "correct"), ("MO", "correct"), ("MR", "correct"), ("MV", "correct"))
PROVIDERS = {
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.6-flash",
    "mistral": "mistral-medium-3-5",
    "qwen": "qwen3.7-max-2026-06-08",
    "openai": "gpt-5.4",
}
SELECTION_SEED = "aaai-checklist-correction-6x12-v1"

CORRECTIONS = {
    "A124_09": {
        "family": "neighbour checklist",
        "question": "Would you prefer not to have homosexuals as neighbours?",
    },
    "A124_08": {
        "family": "neighbour checklist",
        "question": "Would you prefer not to have drug addicts as neighbours?",
    },
    "A124_03": {
        "family": "neighbour checklist",
        "question": "Would you prefer not to have heavy drinkers as neighbours?",
    },
    "A124_06": {
        "family": "neighbour checklist",
        "question": "Would you prefer not to have immigrants or foreign workers as neighbours?",
    },
    "A039": {
        "family": "child-quality checklist",
        "question": (
            "Would you select determination and perseverance as one of up to five "
            "especially important qualities for children to learn at home?"
        ),
    },
    "A030": {
        "family": "child-quality checklist",
        "question": (
            "Would you select hard work as one of up to five especially important "
            "qualities for children to learn at home?"
        ),
    },
    "A029": {
        "family": "child-quality checklist",
        "question": (
            "Would you select independence as one of up to five especially important "
            "qualities for children to learn at home?"
        ),
    },
    "A042": {
        "family": "child-quality checklist",
        "question": (
            "Would you select obedience as one of up to five especially important "
            "qualities for children to learn at home?"
        ),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_profiles() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in SOURCE_PROFILES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def selected_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one reciprocal pair per true-country/displayed-label cell."""
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(row["true_country_code"], row["balanced_random_country_label"])][
            row["pair_id"]
        ].append(row)
    chosen: list[dict[str, Any]] = []
    for country in COUNTRIES:
        country_cells = sorted(key for key in grouped if key[0] == country)
        if len(country_cells) != 6:
            raise ValueError(f"Expected six displayed-label cells in {country}")
        for cell in country_cells:
            valid_pairs = [pair_id for pair_id, pair in grouped[cell].items() if len(pair) == 2]
            if not valid_pairs:
                raise ValueError(f"No reciprocal pair available for {cell}")
            pair_id = min(
                valid_pairs,
                key=lambda value: hashlib.sha256(
                    f"{SELECTION_SEED}\x1f{country}\x1f{cell[1]}\x1f{value}".encode()
                ).digest(),
            )
            chosen.extend(grouped[cell][pair_id])
    if Counter(row["true_country_code"] for row in chosen) != Counter(
        {country: 12 for country in COUNTRIES}
    ):
        raise ValueError("Supplement selection is not balanced 6 x 12")
    if len({row["profile_id"] for row in chosen}) != 72:
        raise ValueError("Supplement selection contains duplicate profile IDs")
    return sorted(chosen, key=lambda row: row["profile_id"])


def corrected_profile(source: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(source)
    targets = {target["item_id"]: target for target in row["targets"]}
    for item_id in CORRECTIONS:
        options = row["target_options"][item_id]
        if [option["answer_code"] for option in options] != ["0", "1"]:
            raise ValueError(f"Unexpected checklist codes for {item_id}")
        options[0]["answer"] = "No"
        options[1]["answer"] = "Yes"
        targets[item_id]["answer"] = "Yes" if targets[item_id]["answer_code"] == "1" else "No"
    return row


def build_prompt() -> dict[str, Any]:
    prompt = read_json(SOURCE_PROMPT)
    prompt["version"] = "survey-checklist-correction-v1"
    prompt["status"] = "post-review correction supplement; original confirmatory prompt retained"
    prompt["frozen_at"] = datetime.now().astimezone().isoformat()
    prompt["amendment"] = {
        "type": "post-review checklist response-format correction",
        "source_prompt": relative(SOURCE_PROMPT),
        "source_prompt_sha256": sha256(SOURCE_PROMPT),
        "affected_items": sorted(CORRECTIONS),
        "reason": (
            "Blind review found that checklist indicators coded as 0=Not mentioned and "
            "1=Mentioned had been rendered as standalone questions. The correction "
            "preserves the binary codes while displaying 0=No and 1=Yes."
        ),
        "confirmatory_outcomes_observed_before_change": True,
        "inference_status": "corrective supplementary analysis, not preregistered confirmation",
    }
    for item_id, correction in CORRECTIONS.items():
        prompt["items"][item_id] = {
            "question": correction["question"],
            "scale": "0 = No; 1 = Yes.",
        }
    return prompt


def write_review(prompt: dict[str, Any]) -> None:
    REVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "review_unit", "family", "question", "scale",
                "question_scale_match", "ambiguous_or_mistranslated",
                "acceptable", "notes",
            ),
        )
        writer.writeheader()
        for item_id in sorted(CORRECTIONS):
            writer.writerow(
                {
                    "review_unit": item_id,
                    "family": CORRECTIONS[item_id]["family"],
                    "question": prompt["items"][item_id]["question"],
                    "scale": prompt["items"][item_id]["scale"],
                    "question_scale_match": "",
                    "ambiguous_or_mistranslated": "",
                    "acceptable": "",
                    "notes": "",
                }
            )


def main() -> int:
    source_task = read_json(SOURCE_TASK)
    prompt = build_prompt()
    write_json(PROMPT_OUT, prompt)

    selected = [corrected_profile(row) for row in selected_profiles(load_profiles())]
    PROFILES_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_OUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )

    profile_ids = [row["profile_id"] for row in selected]
    batch_ids = list(source_task["target_batches"])
    profile_batches = {
        profile_id: source_task["profile_target_batches"][profile_id]
        for profile_id in profile_ids
    }
    task_units = [
        {
            "profile_id": profile_id,
            "condition": condition,
            "evidence_variant": evidence,
            "batch_id": batch_id,
        }
        for profile_id in profile_ids
        for condition, evidence in CELLS
        for batch_id in batch_ids
    ]
    expected_per_provider = 72 * len(CELLS) * len(batch_ids)
    if len(task_units) != expected_per_provider:
        raise ValueError("Supplement task count is inconsistent")

    amendment = {
        "schema_version": "checklist-conversion-amendment-v1",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "trigger": "independent blind question/scale review",
        "affected_items": CORRECTIONS,
        "source_confirmatory_prompt": relative(SOURCE_PROMPT),
        "source_confirmatory_prompt_sha256": sha256(SOURCE_PROMPT),
        "source_confirmatory_results_retained": True,
        "analysis_actions": [
            "exclude all eight affected items from a no-new-API sensitivity analysis",
            "rerun corrected wording on a country- and displayed-label-balanced 6x12 subset",
            "preserve the original five-item batch composition and order",
            "report correction effects as post-review supplementary evidence",
        ],
        "selection_seed": SELECTION_SEED,
        "selection_uses_model_outputs": False,
        "planned_api_calls": expected_per_provider * len(PROVIDERS),
    }
    write_json(AMENDMENT_OUT, amendment)

    manifest = {
        "schema_version": "survey-task-manifest-v2",
        "version": "checklist-correction-supplement-v1",
        "status": "frozen post-review corrective supplement",
        "suite_version": "checklist-correction-suite-v1",
        "experiment_id": "checklist_correction_supplement_v1",
        "task_mode": "standard",
        "profile_ids": profile_ids,
        "cells": [list(cell) for cell in CELLS],
        "target_batches": batch_ids,
        "profile_target_batches": profile_batches,
        "task_units": task_units,
        "source": {
            "profile_file": relative(PROFILES_OUT),
            "profile_file_sha256": sha256(PROFILES_OUT),
            "profile_count": 72,
            "profiles_per_country": 12,
            "analysis_manifest_file": relative(SOURCE_ANALYSIS),
            "analysis_manifest_sha256": sha256(SOURCE_ANALYSIS),
            "label_assignment_file": source_task["source"]["label_assignment_file"],
            "label_assignment_sha256": source_task["source"]["label_assignment_sha256"],
            "donor_assignment_file": source_task["source"]["donor_assignment_file"],
            "donor_assignment_sha256": source_task["source"]["donor_assignment_sha256"],
            "amendment_file": relative(AMENDMENT_OUT),
            "amendment_sha256": sha256(AMENDMENT_OUT),
        },
        "prompt": {
            "file": relative(PROMPT_OUT),
            "version": prompt["version"],
            "sha256": sha256(PROMPT_OUT),
        },
        "provider_models": PROVIDERS,
        "runner_configuration": {
            "target_batch_size": 5,
            "reverse_target_options": False,
        },
        "selection": {
            "method": "one reciprocal pair per true-country/displayed-label cell",
            "seed": SELECTION_SEED,
            "profiles": 72,
            "profiles_by_true_country": {country: 12 for country in COUNTRIES},
            "profiles_per_true_country_and_displayed_label": 2,
            "uses_model_outputs": False,
        },
        "expected": {
            "providers": len(PROVIDERS),
            "calls_per_provider": expected_per_provider,
            "total_calls": expected_per_provider * len(PROVIDERS),
        },
        "immutable_output_path": relative(RESULT_OUT),
        "immutable_checkpoint_path": relative(CHECKPOINT_OUT),
        "analysis_contract": {
            "status": "post-review corrective supplement",
            "affected_items": sorted(CORRECTIONS),
            "original_confirmatory_decisions_unchanged": True,
            "scientific_gate_action": "report_only",
        },
        "runtime_policy": {
            "scientific_gate_action": "report_only",
            "failed_call_action": "defer_then_retry",
        },
    }
    write_json(MANIFEST_OUT, manifest)
    write_review(prompt)
    print(
        json.dumps(
            {
                "profiles": len(selected),
                "affected_items": len(CORRECTIONS),
                "calls_per_provider": expected_per_provider,
                "total_calls": expected_per_provider * len(PROVIDERS),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
