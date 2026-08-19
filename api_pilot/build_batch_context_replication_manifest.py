#!/usr/bin/env python3
"""Freeze a 2.38% one-target-in-new-context confirmatory replication."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "dataset/analysis/confirmatory_core_task_manifest_v2.json"
PROFILES = ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl"
OUTPUT = ROOT / "dataset/analysis/confirmatory_batch_context_replication_v1.json"
RESULT = ROOT / "api_pilot/restricted_outputs/confirmatory_batch_context_replication_v1.jsonl"
COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")


def digest(*parts: str) -> bytes:
    return hashlib.sha256("\x1f".join(parts).encode()).digest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    core = json.loads(CORE.read_text(encoding="utf-8"))
    profiles = [
        json.loads(line) for line in PROFILES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    for country in COUNTRIES:
        candidates = sorted(
            (row for row in profiles if row["true_country_code"] == country),
            key=lambda row: digest("batch-context-v1", row["profile_id"]),
        )
        selected.extend(candidates[:8])
    if Counter(row["true_country_code"] for row in selected) != Counter(
        {country: 8 for country in COUNTRIES}
    ):
        raise ValueError("Replication selection is not country-balanced")

    mappings = {}
    anchor_by_profile = {}
    for index, profile in enumerate(sorted(selected, key=lambda row: row["profile_id"])):
        original = core["profile_target_batches"][profile["profile_id"]]
        flat = [item for batch in core["target_batches"] for item in original[batch]]
        anchor = flat[index % len(flat)]
        original_batch = next(batch for batch, items in original.items() if anchor in items)
        original_partners = set(original[original_batch]) - {anchor}
        alternatives = [
            item for item in flat if item != anchor and item not in original_partners
        ]
        alternatives.sort(key=lambda item: digest(profile["profile_id"], anchor, item))
        new_first = [anchor, *alternatives[:4]]
        remainder = [item for item in flat if item not in new_first]
        mappings[profile["profile_id"]] = {
            "B01": new_first,
            "B02": remainder[:5],
            "B03": remainder[5:10],
            "B04": remainder[10:15],
        }
        anchor_by_profile[profile["profile_id"]] = {
            "item_id": anchor,
            "original_batch_id": original_batch,
            "replication_batch_id": "B01",
        }

    cells = core["cells"]
    units = [
        {
            "profile_id": profile["profile_id"],
            "condition": condition,
            "evidence_variant": evidence,
            "batch_id": "B01",
        }
        for profile in sorted(selected, key=lambda row: row["profile_id"])
        for condition, evidence in cells
    ]
    manifest = {
        **core,
        "version": "confirmatory-batch-context-replication-v1",
        "status": "prospectively frozen report-only batch-context replication",
        "experiment_id": "confirmatory_batch_context_replication_v1",
        "profile_ids": sorted(row["profile_id"] for row in selected),
        "target_batches": ["B01"],
        "task_units": units,
        "profile_target_batches": mappings,
        "selection": {
            "method": "stable SHA-256 selection of eight profiles per country",
            "profile_count": 48,
            "profiles_by_true_country": {country: 8 for country in COUNTRIES},
            "anchor_by_profile": anchor_by_profile,
            "role": "report-only context sensitivity; no runtime gate",
        },
        "expected": {
            "providers": 4,
            "calls_per_provider": len(units),
            "total_calls": 4 * len(units),
        },
        "immutable_output_path": RESULT.relative_to(ROOT).as_posix(),
        "immutable_checkpoint_path": RESULT.with_suffix(
            ".checkpoint.sqlite3"
        ).relative_to(ROOT).as_posix(),
        "analysis_contract": {
            **core["analysis_contract"],
            "role": "report-only one-target-in-new-five-item-context replication",
            "anchor_by_profile": anchor_by_profile,
        },
    }
    manifest["source"] = {**core["source"], "core_manifest_file": CORE.relative_to(ROOT).as_posix(), "core_manifest_sha256": sha256(CORE)}
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": OUTPUT.relative_to(ROOT).as_posix(), "calls": manifest["expected"]["total_calls"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
