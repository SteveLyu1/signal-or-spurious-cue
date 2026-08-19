#!/usr/bin/env python3
"""Create a restricted, hash-bound source-record exclusion audit for replication v2."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset/tools"))
import build_confirmatory_manifest as cm  # noqa: E402
import build_pilot_gate as gate  # noqa: E402

PANEL = ROOT / "dataset/derived/country_panel_6.csv"
METADATA = ROOT / "dataset/metadata/variable_metadata.json"
SOURCE_ANALYSIS = ROOT / "dataset/analysis/confirmatory_manifest_v2.json"
TASK = ROOT / "dataset/analysis/independent_correction_replication_manifest_v2.json"
ANALYSIS = ROOT / "dataset/analysis/independent_correction_analysis_manifest_v2.json"
PROMPT = ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json"
PROFILES = ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl"
INVALIDATION = ROOT / "api_pilot/restricted_outputs/invalidated/independent_correction_v1_20260726-213804/INVALIDATION.json"
AUDIT_OUT = ROOT / "api_pilot/restricted_outputs/independent_correction_selection_audit_v2.json"
CONTRACT_OUT = ROOT / "dataset/analysis/independent_correction_provenance_amendment_v1.json"
COMMITMENT_SALT = "aaai-independent-correction-source-commitment-v1"
REPLICATION_TARGETS = (
    "A124_09", "E025", "A124_08", "A124_03", "A030", "E027", "A029",
    "D060", "D078", "D061",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commitment(value: str) -> str:
    return hashlib.sha256(f"{COMMITMENT_SALT}\x1f{value}".encode()).hexdigest()


def set_commitment(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(commitment(value) for value in values)).encode()).hexdigest()


def canonical_commitment(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    previous_contract = (
        json.loads(CONTRACT_OUT.read_text(encoding="utf-8"))
        if CONTRACT_OUT.exists()
        else {}
    )
    source = json.loads(SOURCE_ANALYSIS.read_text(encoding="utf-8"))
    full_targets = tuple(item for batch in source["target_batches"].values() for item in batch)
    with PANEL.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if gate.respondent_split(row["uniqid"]) == "confirmatory"
        ]
    original_ids: set[str] = set()
    replication_ids: set[str] = set()
    per_country: dict[str, dict[str, int]] = {}
    for country in gate.COUNTRIES:
        country_rows = [row for row in rows if row["cntry_AN"] == country]
        original = sorted(
            (row for row in country_rows if all(
                row[item] not in gate.MISSING_VALUES
                for item in gate.EVIDENCE_ITEMS + full_targets
            )),
            key=lambda row: cm.stable_unit("respondent-order", row["uniqid"]),
        )[:84]
        country_original = {row["uniqid"] for row in original}
        candidates = sorted(
            (row for row in country_rows if row["uniqid"] not in country_original and all(
                row[item] not in gate.MISSING_VALUES
                for item in gate.EVIDENCE_ITEMS + REPLICATION_TARGETS
            )),
            key=lambda row: cm.stable_unit("respondent-order", row["uniqid"]),
        )
        selected = {row["uniqid"] for row in candidates[:12]}
        if len(country_original) != 84 or len(selected) != 12:
            raise ValueError(f"Unexpected source selection count for {country}")
        original_ids.update(country_original)
        replication_ids.update(selected)
        per_country[country] = {
            "original": len(country_original),
            "eligible_after_exclusion": len(candidates),
            "selected_replication": len(selected),
        }
    overlap = original_ids.intersection(replication_ids)
    if overlap:
        raise ValueError("Replication source records overlap the original 504")
    profile_rows = [json.loads(line) for line in PROFILES.read_text(encoding="utf-8").splitlines() if line]
    if Counter(row["true_country_code"] for row in profile_rows) != Counter({c: 12 for c in gate.COUNTRIES}):
        raise ValueError("Restricted replication profiles are not balanced 6 x 12")
    audit = {
        "status": "verified_zero_source_record_overlap",
        "generated_at": datetime.now().astimezone().isoformat(),
        "restricted": True,
        "selection_algorithm": "reconstruct original full-20 complete first-84 by frozen respondent hash; exclude; select first-12 complete for replication ten-item frame",
        "panel_sha256": sha256(PANEL),
        "metadata_sha256": sha256(METADATA),
        "original_source_record_count": len(original_ids),
        "replication_source_record_count": len(replication_ids),
        "intersection_count": len(overlap),
        "original_source_set_commitment": set_commitment(original_ids),
        "replication_source_set_commitment": set_commitment(replication_ids),
        "salt_identifier": COMMITMENT_SALT,
        "per_country": per_country,
        "restricted_profile_sha256": sha256(PROFILES),
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    task_payload = json.loads(TASK.read_text(encoding="utf-8"))
    contract = {
        "status": "post-freeze provenance amendment; no scientific task, prompt, sample, or outcome changed",
        "created_at": datetime.now().astimezone().isoformat(),
        "serialization_history": {
            "superseded_task_manifest_sha256": previous_contract.get("task_manifest_sha256"),
            "reason": "A validation dry-run rewrote generated timestamps before the builder became freeze-aware.",
            "scientific_task_set_unchanged": True,
            "current_task_units_canonical_sha256": canonical_commitment(task_payload["task_units"]),
        },
        "task_manifest": TASK.relative_to(ROOT).as_posix(),
        "task_manifest_sha256": sha256(TASK),
        "analysis_manifest": ANALYSIS.relative_to(ROOT).as_posix(),
        "analysis_manifest_sha256": sha256(ANALYSIS),
        "prompt_bundle": PROMPT.relative_to(ROOT).as_posix(),
        "prompt_bundle_sha256": sha256(PROMPT),
        "restricted_profiles": PROFILES.relative_to(ROOT).as_posix(),
        "restricted_profiles_sha256": sha256(PROFILES),
        "selection_audit": AUDIT_OUT.relative_to(ROOT).as_posix(),
        "selection_audit_sha256": sha256(AUDIT_OUT),
        "invalidated_v1_record": INVALIDATION.relative_to(ROOT).as_posix(),
        "invalidated_v1_record_sha256": sha256(INVALIDATION),
    }
    CONTRACT_OUT.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"original": len(original_ids), "replication": len(replication_ids), "intersection": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
