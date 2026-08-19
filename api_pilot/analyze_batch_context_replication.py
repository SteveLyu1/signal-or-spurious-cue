#!/usr/bin/env python3
"""Measure report-only forecast drift under a changed five-item batch context."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_raw_five_model.jsonl"
REPLICATION = ROOT / "api_pilot/restricted_outputs/confirmatory_batch_context_replication_v1_five_model.jsonl"
MANIFEST = ROOT / "dataset/analysis/confirmatory_batch_context_replication_v1_five_model_manifest_v1.json"
OUTPUT = ROOT / "api_pilot/outputs/confirmatory_batch_context_analysis.json"
OFFICIAL_PROVIDERS = {"openai", "deepseek", "gemini", "mistral", "qwen"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["call_id"] in latest:
                raise ValueError(f"Duplicate immutable call_id: {row['call_id']}")
            latest[row["call_id"]] = row
    return list(latest.values())


def expected_result_experiment(manifest: dict[str, Any], provider: str) -> str:
    amendment = manifest.get("roster_amendment") or {}
    if provider == "openai":
        return str(
            amendment.get("openai_source_experiment_id")
            or manifest["experiment_id"]
        )
    return str(amendment.get("source_experiment_id") or manifest["experiment_id"])


def item_probabilities(row: dict[str, Any], item_id: str) -> dict[str, float]:
    codes = [str(value) for value in row["target_option_codes"][item_id]]
    values = [float(value) for value in row["parsed_forecasts"][item_id]]
    total = sum(values)
    if (
        len(codes) != len(values)
        or not values
        or any(value < 0 for value in values)
        or abs(total - 1.0) > 0.011
        or total <= 0
    ):
        raise ValueError(f"Invalid probability vector: {row['call_id']}/{item_id}")
    return dict(zip(codes, (value / total for value in values)))


def total_variation(first: dict[str, float], second: dict[str, float]) -> float:
    if set(first) != set(second):
        raise ValueError("Paired forecasts use different answer option codes")
    return 0.5 * sum(abs(first[code] - second[code]) for code in first)


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean_tv": statistics.fmean(values),
        "median_tv": statistics.median(values),
        "p90_tv": sorted(values)[max(0, int(0.9 * len(values)) - 1)],
    }


def main() -> int:
    manifest = load_json(MANIFEST)
    anchors = manifest["selection"]["anchor_by_profile"]
    core_rows = latest_rows(CORE)
    replication_rows = latest_rows(REPLICATION)
    if len(replication_rows) != manifest["expected"]["total_calls"]:
        raise ValueError("Five-model replication panel is incomplete")
    providers = {str(row.get("provider")) for row in replication_rows}
    if providers != OFFICIAL_PROVIDERS:
        raise ValueError(
            f"Five-model replication roster mismatch: {sorted(providers)}"
        )
    provider_counts = {
        provider: sum(row.get("provider") == provider for row in replication_rows)
        for provider in sorted(providers)
    }
    expected_per_provider = int(manifest["expected"]["calls_per_provider"])
    if any(value != expected_per_provider for value in provider_counts.values()):
        raise ValueError(f"Replication provider count mismatch: {provider_counts}")

    core_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in core_rows:
        if row.get("success") and row.get("evidence_variant") == "correct":
            key = (row["provider"], row["profile_id"], row["condition"], row["batch_id"])
            core_index[key] = row

    pairs: list[dict[str, Any]] = []
    for repeat in replication_rows:
        if not repeat.get("success") or repeat.get("evidence_variant") != "correct":
            raise ValueError("Replication panel contains an incomplete or unexpected row")
        if repeat.get("terminal_failure"):
            raise ValueError("Replication panel contains a terminal failure")
        if repeat.get("experiment") != expected_result_experiment(
            manifest, str(repeat.get("provider"))
        ):
            raise ValueError(
                f"Replication experiment mismatch: {repeat['call_id']}"
            )
        anchor = anchors[repeat["profile_id"]]
        item_id = anchor["item_id"]
        key = (
            repeat["provider"], repeat["profile_id"], repeat["condition"],
            anchor["original_batch_id"],
        )
        original = core_index.get(key)
        if original is None or item_id not in original["target_ids"] or item_id not in repeat["target_ids"]:
            raise ValueError(f"Missing frozen context pair: {key}/{item_id}")
        tv = total_variation(
            item_probabilities(original, item_id), item_probabilities(repeat, item_id)
        )
        pairs.append({
            "provider": repeat["provider"], "profile_id": repeat["profile_id"],
            "condition": repeat["condition"], "item_id": item_id, "tv": tv,
        })

    expected_pairs = len(anchors) * len(manifest["cells"]) * len(providers)
    if len(pairs) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} context pairs, observed {len(pairs)}")
    by_provider: dict[str, list[float]] = defaultdict(list)
    by_condition: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        by_provider[pair["provider"]].append(pair["tv"])
        by_condition[pair["condition"]].append(pair["tv"])
    report = {
        "status": "complete",
        "complete": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "role": "report-only batch-context sensitivity analysis",
        "interpretation": "Lower total variation means greater forecast stability when the anchor item is placed with different companion questions.",
        "audit": {
            "complete": True,
            "paired_forecasts": len(pairs),
            "expected": expected_pairs,
            "providers": sorted(providers),
            "calls_by_provider": provider_counts,
            "raw_results_unchanged": True,
            "probability_vectors_locally_normalized_for_tv": True,
        },
        "overall": summary([pair["tv"] for pair in pairs]),
        "by_provider": {key: summary(value) for key, value in sorted(by_provider.items())},
        "by_condition": {key: summary(value) for key, value in sorted(by_condition.items())},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
