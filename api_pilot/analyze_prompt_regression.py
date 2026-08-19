#!/usr/bin/env python3
"""Summarise a one-batch prompt regression without exposing row-level data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_survey_pilot import (
    bootstrap_direction_beta,
    bootstrap_profiles,
    brier,
    total_variation,
)


def latest_successes(path: Path) -> tuple[list[dict], Counter[str]]:
    attempts: Counter[str] = Counter()
    successes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        attempts[row["call_id"]] += 1
        if row["success"]:
            successes[row["call_id"]] = row
    return list(successes.values()), attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=Path("dataset/analysis/pilot_gate_report.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results, attempts = latest_successes(args.results)
    profiles = {
        profile["profile_id"]: profile
        for profile in (
            json.loads(line)
            for line in args.profiles.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    anchors = {row["variable"]: row for row in gate["results"]}
    by_cell = {
        (row["profile_id"], row["condition"], row["evidence_variant"]): row
        for row in results
    }
    item_rows = []
    for profile_id, profile in profiles.items():
        base_row = by_cell[(profile_id, "M0", "correct")]
        shuffled_row = by_cell[(profile_id, "M0", "shuffled")]
        opaque_row = by_cell[(profile_id, "MO", "correct")]
        random_row = by_cell[(profile_id, "MR", "correct")]
        random_shuffled_row = by_cell[(profile_id, "MR", "shuffled")]
        verified_row = by_cell[(profile_id, "MV", "correct")]
        target_truth = {target["item_id"]: target for target in profile["targets"]}
        for item_id in base_row["target_ids"]:
            option_codes = [
                option["answer_code"] for option in profile["target_options"][item_id]
            ]
            truth_index = option_codes.index(target_truth[item_id]["answer_code"])
            base = base_row["parsed_forecasts"][item_id]
            shuffled = shuffled_row["parsed_forecasts"][item_id]
            opaque = opaque_row["parsed_forecasts"][item_id]
            random = random_row["parsed_forecasts"][item_id]
            random_shuffled = random_shuffled_row["parsed_forecasts"][item_id]
            verified = verified_row["parsed_forecasts"][item_id]
            displayed = random_row["displayed_country_code"]
            distributions = anchors[item_id]["reference_distributions"]
            direction = np.asarray(distributions[displayed]) - np.mean(
                [np.asarray(distributions[country]) for country in distributions], axis=0
            )
            opaque_delta = np.asarray(opaque) - np.asarray(base)
            random_delta = np.asarray(random) - np.asarray(base)
            item_rows.append(
                {
                    "true_country_code": profile["true_country_code"],
                    "profile_id": profile_id,
                    "item_id": item_id,
                    "evidence_gain_m0": brier(shuffled, truth_index) - brier(base, truth_index),
                    "evidence_gain_mr": brier(random_shuffled, truth_index) - brier(random, truth_index),
                    "override": (
                        brier(shuffled, truth_index) - brier(base, truth_index)
                        - brier(random_shuffled, truth_index) + brier(random, truth_index)
                    ),
                    "mr_brier_delta": brier(random, truth_index) - brier(base, truth_index),
                    "mv_brier_delta": brier(verified, truth_index) - brier(base, truth_index),
                    "mo_tv": total_variation(base, opaque),
                    "mr_tv": total_variation(base, random),
                    "mo_direction_projection": float(np.dot(opaque_delta, direction)),
                    "mr_direction_projection": float(np.dot(random_delta, direction)),
                    "mo_minus_mr_direction_projection": float(
                        np.dot(opaque_delta - random_delta, direction)
                    ),
                    "direction_norm_sq": float(np.dot(direction, direction)),
                }
            )

    metrics = {}
    for key in (
        "evidence_gain_m0",
        "evidence_gain_mr",
        "override",
        "mr_brier_delta",
        "mv_brier_delta",
        "mo_tv",
        "mr_tv",
    ):
        estimate, low, high = bootstrap_profiles(item_rows, key, "v2:" + key)
        metrics[key] = {"mean": estimate, "ci_low": low, "ci_high": high}
    for name, key in (
        ("mo_direction_beta", "mo_direction_projection"),
        ("mr_direction_beta", "mr_direction_projection"),
        ("mo_minus_mr_direction_beta", "mo_minus_mr_direction_projection"),
    ):
        estimate, low, high = bootstrap_direction_beta(item_rows, key, "v2:" + name)
        metrics[name] = {"mean": estimate, "ci_low": low, "ci_high": high}

    summary = {
        "scope": "independent prompt regression on development-pilot respondents; not confirmatory",
        "profiles": len(profiles),
        "items": sorted({row["item_id"] for row in item_rows}),
        "planned_cells": len(attempts),
        "successful_cells": len(results),
        "first_attempt_success_rate": sum(count == 1 for count in attempts.values()) / len(attempts),
        "usage": {
            "input_tokens": sum(row.get("input_tokens") or 0 for row in results),
            "output_tokens": sum(row.get("output_tokens") or 0 for row in results),
            "mean_latency_ms": float(np.mean([row["latency_ms"] for row in results])),
            "p95_latency_ms": float(np.quantile([row["latency_ms"] for row in results], 0.95)),
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
