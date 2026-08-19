#!/usr/bin/env python3
"""Freeze an independent corrected-wording replication outside the original 504 records."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset" / "tools"))
import build_confirmatory_manifest as cm  # noqa: E402
import build_pilot_gate as gate  # noqa: E402


PANEL = ROOT / "dataset/derived/country_panel_6.csv"
METADATA = ROOT / "dataset/metadata/variable_metadata.json"
SOURCE_ANALYSIS = ROOT / "dataset/analysis/confirmatory_manifest_v2.json"
SOURCE_PROMPT = ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json"
PROFILES_OUT = ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl"
ANALYSIS_OUT = ROOT / "dataset/analysis/independent_correction_analysis_manifest_v2.json"
MANIFEST_OUT = ROOT / "dataset/analysis/independent_correction_replication_manifest_v2.json"
RESULT_OUT = ROOT / "api_pilot/restricted_outputs/independent_correction_replication_v2.jsonl"
CHECKPOINT_OUT = RESULT_OUT.with_suffix(".checkpoint.sqlite3")

HIGH_ITEMS = ("A124_09", "E025", "A124_08", "A124_03", "A030", "E027", "A029")
FILLER_ITEMS = ("D060", "D078", "D061")
TARGET_BATCHES = {
    "B01": ["A124_09", "E025", "A124_08", "A124_03", "D060"],
    "B02": ["A030", "E027", "A029", "D078", "D061"],
}
CELLS = (("M0", "correct"), ("MO", "correct"), ("MR", "correct"), ("MV", "correct"))
PROVIDERS = {
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.6-flash",
    "mistral": "mistral-medium-3-5",
    "qwen": "qwen3.7-max-2026-06-08",
    "openai": "gpt-5.4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checkpoint_result_count(path: Path) -> int:
    """Return committed calls without creating or modifying the checkpoint."""
    if not path.exists():
        return 0
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='results'"
        ).fetchone()
        return int(connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]) if table else 0
    finally:
        connection.close()


def validate_frozen_outputs() -> dict[str, int]:
    """Fail closed once calls exist; a rerun must never rewrite frozen inputs."""
    for path in (PROFILES_OUT, ANALYSIS_OUT, MANIFEST_OUT):
        if not path.exists():
            raise FileNotFoundError(f"Frozen artifact missing after calls began: {path}")
    profiles = [
        json.loads(line)
        for line in PROFILES_OUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
    if len(profiles) != 72 or len(manifest.get("task_units", [])) != 576:
        raise ValueError("Frozen v2 artifacts do not retain the 72-profile/576-unit contract")
    if manifest.get("expected", {}).get("total_calls") != 2880:
        raise ValueError("Frozen v2 manifest no longer specifies 2,880 calls")
    return {"profiles": 72, "calls_per_provider": 576, "total_calls": 2880}


def load_confirmatory_rows() -> list[dict[str, str]]:
    with PANEL.open(encoding="utf-8-sig", newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if gate.respondent_split(row["uniqid"]) == "confirmatory"
        ]


def eligible(rows: list[dict[str, str]], targets: tuple[str, ...]) -> list[dict[str, str]]:
    required = gate.EVIDENCE_ITEMS + targets
    return [row for row in rows if all(row[item] not in gate.MISSING_VALUES for item in required)]


def apply_checklist_corrections(profile: dict[str, Any]) -> dict[str, Any]:
    """Align binary profile labels with the corrected Yes/No prompt wording."""
    corrected = {"A124_09", "A124_08", "A124_03", "A030", "A029"}
    targets = {row["item_id"]: row for row in profile["targets"]}
    for item_id in corrected:
        options = profile["target_options"][item_id]
        if [row["answer_code"] for row in options] != ["0", "1"]:
            raise ValueError(f"Unexpected checklist coding for {item_id}")
        options[0]["answer"] = "No"
        options[1]["answer"] = "Yes"
        targets[item_id]["answer"] = (
            "Yes" if targets[item_id]["answer_code"] == "1" else "No"
        )
    return profile


def main() -> int:
    committed = checkpoint_result_count(CHECKPOINT_OUT)
    if committed:
        summary = validate_frozen_outputs()
        summary["frozen_checkpoint_rows"] = committed
        summary["action"] = "validated_without_rewrite"
        print(json.dumps(summary))
        return 0

    source = json.loads(SOURCE_ANALYSIS.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))["variables"]
    rows = load_confirmatory_rows()
    full_targets = tuple(item for batch in source["target_batches"].values() for item in batch)
    original_ids: set[str] = set()
    remaining: list[dict[str, str]] = []
    availability: dict[str, dict[str, int]] = {}
    targets = HIGH_ITEMS + FILLER_ITEMS
    for country in gate.COUNTRIES:
        country_rows = [row for row in rows if row["cntry_AN"] == country]
        original = sorted(
            eligible(country_rows, full_targets),
            key=lambda row: cm.stable_unit("respondent-order", row["uniqid"]),
        )[:84]
        original_ids.update(row["uniqid"] for row in original)
        candidates = sorted(
            (row for row in eligible(country_rows, targets) if row["uniqid"] not in original_ids),
            key=lambda row: cm.stable_unit("respondent-order", row["uniqid"]),
        )
        if len(candidates) < 12:
            raise ValueError(f"{country} has only {len(candidates)} independent complete rows")
        remaining.extend(candidates)
        availability[country] = {
            "eligible_for_replication": len(candidates),
            "selected": 12,
        }

    profiles, _ = cm.build_profiles(
        remaining,
        metadata,
        list(targets),
        profiles_per_country=12,
        profile_prefix="R",
        label_assignment="balanced_seeded",
        shuffled_pairing="reciprocal_within_country_label",
    )
    profiles = [apply_checklist_corrections(row) for row in profiles]
    if any(row["profile_id"].startswith("C-") for row in profiles):
        raise ValueError("Replication profile namespace overlaps the original experiment")
    if Counter(row["true_country_code"] for row in profiles) != Counter({c: 12 for c in gate.COUNTRIES}):
        raise ValueError("Replication is not balanced 6 x 12")

    PROFILES_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_OUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in profiles),
        encoding="utf-8",
    )
    frozen_at = datetime.now().astimezone().isoformat()
    analysis = {
        "status": "post-review independent replication v2 frozen before valid replication outputs",
        "frozen_at": frozen_at,
        "analysis_role": "independent corrected-wording replication; not original confirmation",
        "target_strata": {"high_country_value": list(HIGH_ITEMS), "filler": list(FILLER_ITEMS)},
        "target_batches": TARGET_BATCHES,
        "profiles": {"count": 72, "per_country": 12, "availability_after_excluding_original_504": availability},
        "source_confirmatory_manifest": rel(SOURCE_ANALYSIS),
        "source_confirmatory_manifest_sha256": sha256(SOURCE_ANALYSIS),
    }
    write_json(ANALYSIS_OUT, analysis)

    profile_ids = [row["profile_id"] for row in profiles]
    units = [
        {"profile_id": pid, "condition": condition, "evidence_variant": evidence, "batch_id": batch}
        for pid in profile_ids for condition, evidence in CELLS for batch in TARGET_BATCHES
    ]
    per_provider = len(units)
    manifest = {
        "schema_version": "survey-task-manifest-v2",
        "version": "independent-correction-replication-v2",
        "status": "frozen post-review independent replication",
        "suite_version": "independent-correction-suite-v1",
        "experiment_id": "independent_correction_replication_v2",
        "task_mode": "standard",
        "profile_ids": profile_ids,
        "cells": [list(cell) for cell in CELLS],
        "target_batches": list(TARGET_BATCHES),
        "profile_target_batches": {pid: TARGET_BATCHES for pid in profile_ids},
        "task_units": units,
        "source": {
            "profile_file": rel(PROFILES_OUT),
            "profile_file_sha256": sha256(PROFILES_OUT),
            "analysis_manifest_file": rel(ANALYSIS_OUT),
            "analysis_manifest_sha256": sha256(ANALYSIS_OUT),
        },
        "prompt": {"file": rel(SOURCE_PROMPT), "version": "survey-checklist-correction-v1", "sha256": sha256(SOURCE_PROMPT)},
        "provider_models": PROVIDERS,
        "runner_configuration": {"target_batch_size": 5, "reverse_target_options": False},
        "selection": {
            "method": "first 12 by frozen respondent hash after excluding all original 504 source records",
            "uses_model_outputs": False,
            "profiles": 72,
            "profiles_by_true_country": {country: 12 for country in gate.COUNTRIES},
            "profiles_per_true_country_and_displayed_label": 2,
        },
        "expected": {"providers": 5, "calls_per_provider": per_provider, "total_calls": per_provider * 5},
        "immutable_output_path": rel(RESULT_OUT),
        "immutable_checkpoint_path": rel(CHECKPOINT_OUT),
        "analysis_contract": {
            "primary_items": list(HIGH_ITEMS),
            "filler_items_excluded_from_inference": list(FILLER_ITEMS),
            "scientific_gate_action": "report_only",
            "invalidated_predecessor": "independent-correction-replication-v1; profile option labels were inconsistent with corrected Yes/No wording",
        },
        "runtime_policy": {"scientific_gate_action": "report_only", "failed_call_action": "defer_then_retry"},
    }
    write_json(MANIFEST_OUT, manifest)
    print(json.dumps({"profiles": 72, "calls_per_provider": per_provider, "total_calls": per_provider * 5}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
