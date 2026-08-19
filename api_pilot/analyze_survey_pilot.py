#!/usr/bin/env python3
"""Aggregate restricted survey pilot outputs without exposing row-level data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")
CONDITIONS = ("M0", "MO", "MR", "MV")
HIGH_ITEMS = {"A124_09", "E025", "E027", "A030"}
CONTROL_ITEMS = {"A124_06", "A004"}
DEFAULT_CONFIRMATORY_MANIFEST = Path("dataset/analysis/confirmatory_manifest_v2.json")
N_BOOTSTRAP = 10_000
N_ASSIGNMENT_PERMUTATIONS = 100_000


def prompt_repeat_source_id(row: dict[str, Any]) -> str | None:
    """Resolve a prompt-identical repeat to its source call, if this is one."""
    call_id = str(row.get("call_id") or "")
    explicit = row.get("source_call_id")
    suffix_source = None
    if "/REPEAT-" in call_id:
        suffix_source, separator, repeat_label = call_id.rpartition("/REPEAT-")
        if not separator or not suffix_source or not repeat_label:
            raise ValueError(f"Malformed repeat call_id: {call_id}")
    marked_repeat = row.get("experiment") == "repeat" or explicit is not None or suffix_source is not None
    if not marked_repeat:
        return None
    if explicit is None and suffix_source is None:
        raise ValueError(f"Repeat row has no source_call_id or repeat suffix: {call_id}")
    if explicit is not None and suffix_source is not None and explicit != suffix_source:
        raise ValueError(
            f"Repeat source_call_id conflicts with call_id suffix: {call_id}"
        )
    source_id = str(explicit or suffix_source)
    if source_id == call_id:
        raise ValueError(f"Repeat row refers to itself: {call_id}")
    return source_id


def index_result_batches(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str, str | None], dict[str, Any]]:
    """Index each inference cell without collapsing distinct target batches."""
    indexed = {}
    for row in rows:
        # Balanced prompt-identical repeats retain the source condition and batch.
        # They are audit observations, not replacements for the source forecast.
        if prompt_repeat_source_id(row) is not None:
            continue
        key = (
            row["provider"],
            row["profile_id"],
            row["condition"],
            row["evidence_variant"],
            row.get("batch_id"),
        )
        if key in indexed:
            raise ValueError(f"Duplicate successful result cell: {key}")
        indexed[key] = row
    return indexed


def prompt_repeat_item_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair successful prompt-identical repeats and calculate item-level TV."""
    by_call_id = {row["call_id"]: row for row in rows}
    if len(by_call_id) != len(rows):
        raise ValueError("Successful results contain duplicate call IDs")
    item_rows = []
    identity_fields = (
        "provider",
        "model_requested",
        "prompt_version",
        "profile_id",
        "true_country_code",
        "condition",
        "evidence_variant",
        "batch_id",
        "displayed_country_code",
        "prompt_sha256",
    )
    for repeated in rows:
        source_call_id = prompt_repeat_source_id(repeated)
        if source_call_id is None:
            continue
        source = by_call_id.get(source_call_id)
        if source is None:
            raise ValueError(
                f"Successful repeat {repeated['call_id']} has no successful source {source_call_id}"
            )
        if prompt_repeat_source_id(source) is not None:
            raise ValueError(
                f"Repeat {repeated['call_id']} points to another repeat {source_call_id}"
            )
        for field in identity_fields:
            if source.get(field) != repeated.get(field):
                raise ValueError(
                    f"Repeat/source mismatch for {field}: {repeated['call_id']}"
                )
        source_targets = list(source.get("target_ids") or [])
        repeated_targets = list(repeated.get("target_ids") or [])
        if source_targets != repeated_targets or not source_targets:
            raise ValueError(
                f"Repeat/source target IDs do not match: {repeated['call_id']}"
            )
        if source.get("target_option_codes") != repeated.get("target_option_codes"):
            raise ValueError(
                f"Repeat/source option codes do not match: {repeated['call_id']}"
            )
        source_forecasts = source.get("parsed_forecasts") or {}
        repeated_forecasts = repeated.get("parsed_forecasts") or {}
        if set(source_forecasts) != set(source_targets) or set(repeated_forecasts) != set(source_targets):
            raise ValueError(
                f"Repeat/source forecasts do not cover their targets: {repeated['call_id']}"
            )
        for item_id in source_targets:
            source_values = source_forecasts[item_id]
            repeated_values = repeated_forecasts[item_id]
            if len(source_values) != len(repeated_values):
                raise ValueError(
                    f"Repeat/source probability dimensions differ for {item_id}: "
                    f"{repeated['call_id']}"
                )
            item_rows.append(
                {
                    "repeat_call_id": repeated["call_id"],
                    "source_call_id": source_call_id,
                    "provider": repeated["provider"],
                    "profile_id": repeated["profile_id"],
                    "condition": repeated["condition"],
                    "evidence_variant": repeated["evidence_variant"],
                    "batch_id": repeated.get("batch_id"),
                    "item_id": item_id,
                    "tv": total_variation(source_values, repeated_values),
                }
            )
    return item_rows


def summarize_prompt_repeat_noise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate prompt-identical repeat noise overall, by provider, and by cell."""
    observations = prompt_repeat_item_rows(rows)

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        values = [row["tv"] for row in group]
        return {
            "pair_count": len({row["repeat_call_id"] for row in group}),
            "item_comparison_count": len(group),
            "mean_item_tv": float(np.mean(values)) if values else None,
            "median_item_tv": float(np.median(values)) if values else None,
            "max_item_tv": float(np.max(values)) if values else None,
        }

    providers = sorted({row["provider"] for row in observations})
    cells = sorted(
        {(row["condition"], row["evidence_variant"]) for row in observations}
    )
    return {
        "available": bool(observations),
        "definition": "TV between exact prompt-identical source and repeat forecasts, averaged over items",
        "overall": aggregate(observations),
        "by_provider": {
            provider: aggregate(
                [row for row in observations if row["provider"] == provider]
            )
            for provider in providers
        },
        "by_cell": {
            f"{condition}/{evidence_variant}": aggregate(
                [
                    row
                    for row in observations
                    if row["condition"] == condition
                    and row["evidence_variant"] == evidence_variant
                ]
            )
            for condition, evidence_variant in cells
        },
        "by_provider_cell": {
            provider: {
                f"{condition}/{evidence_variant}": aggregate(
                    [
                        row
                        for row in observations
                        if row["provider"] == provider
                        and row["condition"] == condition
                        and row["evidence_variant"] == evidence_variant
                    ]
                )
                for condition, evidence_variant in cells
                if any(
                    row["provider"] == provider
                    and row["condition"] == condition
                    and row["evidence_variant"] == evidence_variant
                    for row in observations
                )
            }
            for provider in providers
        },
    }


def result_for_item(
    indexed: dict[tuple[str, str, str, str, str | None], dict[str, Any]],
    provider: str,
    profile_id: str,
    condition: str,
    evidence_variant: str,
    item_id: str,
    item_batches: dict[str, str],
) -> dict[str, Any]:
    """Return the one result row containing an item, using its frozen batch when known."""
    prefix = (provider, profile_id, condition, evidence_variant)
    expected_batch = item_batches.get(item_id)
    if expected_batch is not None:
        expected = indexed.get((*prefix, expected_batch))
        if expected is None:
            raise ValueError(
                f"Missing result for {prefix}, manifest batch {expected_batch}, item {item_id}"
            )
        contained = set(expected.get("target_ids") or (expected.get("parsed_forecasts") or {}))
        if item_id not in contained:
            raise ValueError(
                f"Manifest assigns {item_id} to {expected_batch}, but the result row does not contain it"
            )
        return expected

    candidates = []
    for key, row in indexed.items():
        if key[:4] != prefix:
            continue
        contained = set(row.get("target_ids") or (row.get("parsed_forecasts") or {}))
        if item_id in contained:
            candidates.append(row)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one result for {prefix} and item {item_id}; found {len(candidates)}"
        )
    return candidates[0]


def analysis_design(
    item_ids: set[str], manifest_path: Path | None
) -> tuple[dict[str, str], dict[str, str], tuple[str, ...], str]:
    """Load frozen confirmatory strata/batches, or use the legacy six-item pilot design."""
    selected_manifest = manifest_path
    if selected_manifest is None and DEFAULT_CONFIRMATORY_MANIFEST.exists():
        candidate = json.loads(DEFAULT_CONFIRMATORY_MANIFEST.read_text(encoding="utf-8"))
        candidate_items = {
            item_id
            for stratum_items in candidate["target_strata"].values()
            for item_id in stratum_items
        }
        if item_ids == candidate_items:
            selected_manifest = DEFAULT_CONFIRMATORY_MANIFEST

    if selected_manifest is not None:
        manifest = json.loads(selected_manifest.read_text(encoding="utf-8"))
        item_strata = {}
        for stratum, stratum_items in manifest["target_strata"].items():
            for item_id in stratum_items:
                if item_id in item_strata:
                    raise ValueError(f"Manifest assigns {item_id} to multiple strata")
                item_strata[item_id] = stratum
        item_batches = {}
        for batch_id, batch_items in manifest["target_batches"].items():
            for item_id in batch_items:
                if item_id in item_batches:
                    raise ValueError(f"Manifest assigns {item_id} to multiple batches")
                item_batches[item_id] = batch_id
        missing = item_ids.difference(item_strata).union(item_ids.difference(item_batches))
        if missing:
            raise ValueError(f"Manifest does not cover profile targets: {sorted(missing)}")
        strata = tuple(manifest["target_strata"])
        return item_strata, item_batches, strata, str(selected_manifest)

    legacy_items = HIGH_ITEMS | CONTROL_ITEMS
    missing = item_ids.difference(legacy_items)
    if missing:
        raise ValueError(
            "Targets are not the legacy six-item pilot and no matching manifest was selected: "
            f"{sorted(missing)}"
        )
    item_strata = {
        item_id: "high" if item_id in HIGH_ITEMS else "control"
        for item_id in item_ids
    }
    return item_strata, {}, ("high", "control"), "legacy-six-item-pilot"


def latest_successes(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    attempts = Counter(row["call_id"] for row in rows)
    successes = {}
    for row in rows:
        if row["success"]:
            successes[row["call_id"]] = row
    return list(successes.values()), attempts


def brier(probabilities: list[float], truth_index: int) -> float:
    truth = np.zeros(len(probabilities))
    truth[truth_index] = 1
    return float(np.sum((np.asarray(probabilities) - truth) ** 2))


def total_variation(left: list[float], right: list[float]) -> float:
    return float(np.sum(np.abs(np.asarray(left) - np.asarray(right))) / 2)


def argmax_accuracy(probabilities: list[float], truth_index: int) -> float:
    """Give fractional credit when multiple displayed options tie for top."""
    values = np.asarray(probabilities, dtype=float)
    winners = np.flatnonzero(np.isclose(values, np.max(values)))
    return float(1 / len(winners) if truth_index in winners else 0.0)


def ordinal_crps(probabilities: list[float], truth_index: int) -> float:
    """Ranked probability score for an ordered categorical response."""
    values = np.asarray(probabilities, dtype=float)
    forecast_cdf = np.cumsum(values)[:-1]
    truth_cdf = (np.arange(len(values) - 1) >= truth_index).astype(float)
    return float(np.sum((forecast_cdf - truth_cdf) ** 2))


def bootstrap_profiles(
    rows: list[dict[str, Any]], value_key: str, seed: str
) -> tuple[float, float, float]:
    by_country: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_country[row["true_country_code"]][row["profile_id"]].append(row)
    country_profile_values = {
        country: (
            np.asarray(
                [np.mean([row[value_key] for row in values]) for values in profiles.values()],
                dtype=float,
            ),
            np.asarray(
                [float(values[0].get("analysis_weight", 1.0)) for values in profiles.values()],
                dtype=float,
            ),
        )
        for country, profiles in by_country.items()
    }
    observed = float(
        np.mean(
            [np.average(values, weights=weights) for values, weights in country_profile_values.values()]
        )
    )
    rng_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    samples = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        samples[index] = np.mean(
            [
                np.average(values[indexes], weights=weights[indexes])
                for values, weights in country_profile_values.values()
                for indexes in [rng.integers(0, len(values), len(values))]
            ]
        )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return observed, float(lower), float(upper)


def direction_beta(rows: list[dict[str, Any]], projection_key: str) -> float:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    item_betas = []
    for item_rows in by_item.values():
        country_weight_totals = defaultdict(float)
        for row in item_rows:
            country_weight_totals[row["true_country_code"]] += float(
                row.get("analysis_weight", 1.0)
            )
        normalized_weights = [
            float(row.get("analysis_weight", 1.0))
            / country_weight_totals[row["true_country_code"]]
            for row in item_rows
        ]
        denominator = sum(
            weight * row["direction_norm_sq"]
            for weight, row in zip(normalized_weights, item_rows)
        )
        if denominator > 0:
            item_betas.append(
                sum(
                    weight * row[projection_key]
                    for weight, row in zip(normalized_weights, item_rows)
                )
                / denominator
            )
    if not item_betas:
        raise ValueError("No non-zero country directions available")
    return float(np.mean(item_betas))


def bootstrap_direction_beta(
    rows: list[dict[str, Any]], projection_key: str, seed: str
) -> tuple[float, float, float]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["true_country_code"]][row["profile_id"]].append(row)
    observed = direction_beta(rows, projection_key)
    rng_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    samples = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sampled_rows = []
        for country in COUNTRIES:
            profile_rows = list(grouped[country].values())
            sampled_indexes = rng.integers(0, len(profile_rows), len(profile_rows))
            for sampled_index in sampled_indexes:
                sampled_rows.extend(profile_rows[sampled_index])
        samples[index] = direction_beta(sampled_rows, projection_key)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return observed, float(lower), float(upper)


def bootstrap_model_macro(
    rows: list[dict[str, Any]], value_key: str, seed: str
) -> tuple[float, float, float]:
    """Equal-model/country macro mean with shared profile resampling."""
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        grouped[row["provider"]][row["true_country_code"]][row["profile_id"]].append(row)
    providers = sorted(grouped)
    profile_ids = {
        country: sorted(
            set.intersection(
                *(set(grouped[provider][country]) for provider in providers)
            )
        )
        for country in COUNTRIES
    }
    if any(not ids for ids in profile_ids.values()):
        raise ValueError("Macro bootstrap lacks shared profiles for a country")

    values = {
        country: np.asarray(
            [
                [
                    np.mean(
                        [row[value_key] for row in grouped[provider][country][profile_id]]
                    )
                    for profile_id in profile_ids[country]
                ]
                for provider in providers
            ],
            dtype=float,
        )
        for country in COUNTRIES
    }
    weights = {
        country: np.asarray(
            [
                float(grouped[providers[0]][country][profile_id][0].get("analysis_weight", 1.0))
                for profile_id in profile_ids[country]
            ],
            dtype=float,
        )
        for country in COUNTRIES
    }
    observed = float(
        np.mean(
            [
                np.average(matrix, weights=weights[country], axis=1).mean()
                for country, matrix in values.items()
            ]
        )
    )
    rng_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    samples = np.zeros(N_BOOTSTRAP, dtype=float)
    for country in COUNTRIES:
        matrix = values[country]
        indexes = rng.integers(0, matrix.shape[1], (N_BOOTSTRAP, matrix.shape[1]))
        sampled_values = np.take(matrix, indexes, axis=1).transpose(1, 0, 2)
        sampled_weights = weights[country][indexes]
        samples += (
            np.sum(sampled_values * sampled_weights[:, None, :], axis=2)
            / np.sum(sampled_weights, axis=1)[:, None]
        ).mean(axis=1)
    samples /= len(COUNTRIES)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return observed, float(lower), float(upper)


def bootstrap_model_macro_direction(
    rows: list[dict[str, Any]], projection_key: str, seed: str
) -> tuple[float, float, float]:
    """Equal-model macro beta with the same resampled profiles for every model."""
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        grouped[row["provider"]][row["true_country_code"]][row["profile_id"]].append(row)
    providers = sorted(grouped)
    profile_ids = {
        country: sorted(
            set.intersection(
                *(set(grouped[provider][country]) for provider in providers)
            )
        )
        for country in COUNTRIES
    }

    items = sorted({row["item_id"] for row in rows})
    item_index = {item_id: index for index, item_id in enumerate(items)}
    projection = {}
    norm = {}
    weights = {}
    for country in COUNTRIES:
        shape = (len(providers), len(items), len(profile_ids[country]))
        projection[country] = np.zeros(shape, dtype=float)
        norm[country] = np.zeros(shape, dtype=float)
        weights[country] = np.zeros(len(profile_ids[country]), dtype=float)
        profile_index = {value: index for index, value in enumerate(profile_ids[country])}
        for provider_index, provider in enumerate(providers):
            for profile_id in profile_ids[country]:
                weights[country][profile_index[profile_id]] = float(
                    grouped[provider][country][profile_id][0].get("analysis_weight", 1.0)
                )
                for row in grouped[provider][country][profile_id]:
                    coordinates = (
                        provider_index,
                        item_index[row["item_id"]],
                        profile_index[profile_id],
                    )
                    projection[country][coordinates] += row[projection_key]
                    norm[country][coordinates] += row["direction_norm_sq"]

    observed_projection = sum(
        np.sum(matrix * (weights[country] / weights[country].sum())[None, None, :], axis=2)
        for country, matrix in projection.items()
    )
    observed_norm = sum(
        np.sum(matrix * (weights[country] / weights[country].sum())[None, None, :], axis=2)
        for country, matrix in norm.items()
    )
    observed = float(np.mean(observed_projection / observed_norm))
    rng_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    sampled_projection = np.zeros((N_BOOTSTRAP, len(providers), len(items)))
    sampled_norm = np.zeros_like(sampled_projection)
    for country in COUNTRIES:
        matrix = projection[country]
        indexes = rng.integers(0, matrix.shape[2], (N_BOOTSTRAP, matrix.shape[2]))
        sampled_weights = weights[country][indexes]
        sampled_weights /= sampled_weights.sum(axis=1, keepdims=True)
        sampled_projection += np.sum(
            np.take(matrix, indexes, axis=2).transpose(2, 0, 1, 3)
            * sampled_weights[:, None, None, :],
            axis=3,
        )
        sampled_norm += np.sum(
            np.take(norm[country], indexes, axis=2).transpose(2, 0, 1, 3)
            * sampled_weights[:, None, None, :],
            axis=3,
        )
    samples = np.mean(sampled_projection / sampled_norm, axis=(1, 2))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return observed, float(lower), float(upper)


def main() -> int:
    global N_BOOTSTRAP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("api_pilot/restricted_outputs/survey_pilot.jsonl"))
    parser.add_argument("--profiles", type=Path, default=Path("dataset/derived/pilot_profiles_6x12.jsonl"))
    parser.add_argument("--gate", type=Path, default=Path("dataset/analysis/pilot_gate_report.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Frozen item-stratum and target-batch manifest. The confirmatory manifest is "
            "auto-selected only when its complete target set matches the profiles."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("api_pilot/outputs/survey_pilot_summary.json"))
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=N_BOOTSTRAP,
        help="Profile-cluster bootstrap replicates (default: 10000).",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be at least 100")
    N_BOOTSTRAP = args.bootstrap_replicates

    results, attempts = latest_successes(args.results)
    profiles = {
        profile["profile_id"]: profile
        for profile in (
            json.loads(line) for line in args.profiles.read_text(encoding="utf-8").splitlines() if line
        )
    }
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    gate_by_item = {row["variable"]: row for row in gate["results"]}
    profile_target_sets = {
        frozenset(target["item_id"] for target in profile["targets"])
        for profile in profiles.values()
    }
    if len(profile_target_sets) != 1:
        raise ValueError("All profiles must contain the same frozen target set")
    item_ids = set(next(iter(profile_target_sets)))
    item_strata, item_batches, strata, design_source = analysis_design(
        item_ids, args.manifest
    )
    missing_anchors = item_ids.difference(gate_by_item)
    if missing_anchors:
        raise ValueError(f"Gate report does not cover profile targets: {sorted(missing_anchors)}")
    by_cell = index_result_batches(results)
    providers = sorted({row["provider"] for row in results})

    item_rows = []
    for provider in providers:
        profile_ids = sorted({row["profile_id"] for row in results if row["provider"] == provider})
        for profile_id in profile_ids:
            profile = profiles[profile_id]
            target_by_id = {target["item_id"]: target for target in profile["targets"]}
            for item_id, target in target_by_id.items():
                option_codes = [item["answer_code"] for item in profile["target_options"][item_id]]
                truth_index = option_codes.index(target["answer_code"])
                rows_by_condition = {
                    (condition, evidence_variant): result_for_item(
                        by_cell,
                        provider,
                        profile_id,
                        condition,
                        evidence_variant,
                        item_id,
                        item_batches,
                    )
                    for condition, evidence_variant in (
                        ("M0", "correct"),
                        ("M0", "shuffled"),
                        ("MO", "correct"),
                        ("MR", "correct"),
                        ("MR", "shuffled"),
                        ("MV", "correct"),
                    )
                }
                base = rows_by_condition[("M0", "correct")]["parsed_forecasts"][item_id]
                shuffled_base = rows_by_condition[("M0", "shuffled")]["parsed_forecasts"][item_id]
                random_correct = rows_by_condition[("MR", "correct")]["parsed_forecasts"][item_id]
                random_shuffled = rows_by_condition[("MR", "shuffled")]["parsed_forecasts"][item_id]
                displayed_code = rows_by_condition[("MR", "correct")]["displayed_country_code"]
                direction = np.asarray(gate_by_item[item_id]["reference_distributions"][displayed_code])
                pooled = np.mean(
                    [np.asarray(gate_by_item[item_id]["reference_distributions"][country]) for country in COUNTRIES],
                    axis=0,
                )
                random_delta = np.asarray(random_correct) - np.asarray(base)
                opaque = rows_by_condition[("MO", "correct")]["parsed_forecasts"][item_id]
                opaque_delta = np.asarray(opaque) - np.asarray(base)
                direction_vector = direction - pooled
                row = {
                    "provider": provider,
                    "profile_id": profile_id,
                    "true_country_code": profile["true_country_code"],
                    "analysis_weight": float(profile.get("analysis_weight", 1.0)),
                    "item_id": item_id,
                    "batch_id": item_batches.get(item_id),
                    "stratum": item_strata[item_id],
                    "evidence_gain_m0": brier(shuffled_base, truth_index) - brier(base, truth_index),
                    "evidence_gain_mr": brier(random_shuffled, truth_index) - brier(random_correct, truth_index),
                    "random_regret": brier(random_correct, truth_index) - brier(base, truth_index),
                    "random_tv": total_variation(base, random_correct),
                    "m0_accuracy": argmax_accuracy(base, truth_index),
                    "mo_direction_projection": float(
                        np.dot(opaque_delta, direction_vector)
                    ),
                    "mr_direction_projection": float(
                        np.dot(random_delta, direction_vector)
                    ),
                    "mo_minus_mr_direction_projection": float(
                        np.dot(opaque_delta - random_delta, direction_vector)
                    ),
                    "direction_norm_sq": float(
                        np.dot(direction_vector, direction_vector)
                    ),
                }
                for condition in ("MO", "MR", "MV"):
                    forecast = rows_by_condition[(condition, "correct")]["parsed_forecasts"][item_id]
                    row[f"{condition.lower()}_brier_delta"] = brier(forecast, truth_index) - brier(base, truth_index)
                    row[f"{condition.lower()}_tv"] = total_variation(base, forecast)
                    row[f"{condition.lower()}_accuracy_delta"] = (
                        argmax_accuracy(forecast, truth_index)
                        - argmax_accuracy(base, truth_index)
                    )
                if len(option_codes) > 2:
                    row["m0_ordinal_crps"] = ordinal_crps(base, truth_index)
                    for condition in ("MO", "MR", "MV"):
                        forecast = rows_by_condition[(condition, "correct")]["parsed_forecasts"][item_id]
                        row[f"{condition.lower()}_ordinal_crps_delta"] = (
                            ordinal_crps(forecast, truth_index)
                            - ordinal_crps(base, truth_index)
                        )
                item_rows.append(row)

    metrics = {}
    repeat_rows = []
    for provider in providers:
        profile_ids = sorted(
            {
                row["profile_id"]
                for row in results
                if row["provider"] == provider and row["condition"] == "M0_REPEAT"
            }
        )
        for profile_id in profile_ids:
            profile = profiles[profile_id]
            target_by_id = {target["item_id"]: target for target in profile["targets"]}
            for item_id, target in target_by_id.items():
                option_codes = [item["answer_code"] for item in profile["target_options"][item_id]]
                truth_index = option_codes.index(target["answer_code"])
                baseline = result_for_item(
                    by_cell, provider, profile_id, "M0", "correct", item_id, item_batches
                )["parsed_forecasts"][item_id]
                repeated = result_for_item(
                    by_cell,
                    provider,
                    profile_id,
                    "M0_REPEAT",
                    "correct",
                    item_id,
                    item_batches,
                )["parsed_forecasts"][item_id]
                random = result_for_item(
                    by_cell, provider, profile_id, "MR", "correct", item_id, item_batches
                )["parsed_forecasts"][item_id]
                repeat_rows.append(
                    {
                        "provider": provider,
                        "profile_id": profile_id,
                        "true_country_code": profile["true_country_code"],
                        "item_id": item_id,
                        "batch_id": item_batches.get(item_id),
                        "stratum": item_strata[item_id],
                        "repeat_tv": total_variation(baseline, repeated),
                        "repeat_brier_delta": brier(repeated, truth_index) - brier(baseline, truth_index),
                        "mr_tv_same_profiles": total_variation(baseline, random),
                        "mr_excess_over_repeat_tv": total_variation(baseline, random) - total_variation(baseline, repeated),
                    }
                )
    metric_keys = (
        "evidence_gain_m0",
        "evidence_gain_mr",
        "random_regret",
        "random_tv",
        "mo_brier_delta",
        "mo_tv",
        "mr_brier_delta",
        "mr_tv",
        "mv_brier_delta",
        "mv_tv",
        "m0_accuracy",
        "mo_accuracy_delta",
        "mr_accuracy_delta",
        "mv_accuracy_delta",
    )
    for provider in providers:
        metrics[provider] = {}
        for stratum in ("all", *strata):
            subset = [
                row for row in item_rows
                if row["provider"] == provider and (stratum == "all" or row["stratum"] == stratum)
            ]
            metrics[provider][stratum] = {}
            for metric in metric_keys:
                estimate, low, high = bootstrap_profiles(subset, metric, f"{provider}:{stratum}:{metric}")
                metrics[provider][stratum][metric] = {
                    "mean": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            ordinal_subset = [row for row in subset if "m0_ordinal_crps" in row]
            for metric in (
                "m0_ordinal_crps",
                "mo_ordinal_crps_delta",
                "mr_ordinal_crps_delta",
                "mv_ordinal_crps_delta",
            ):
                estimate, low, high = bootstrap_profiles(
                    ordinal_subset, metric, f"{provider}:{stratum}:{metric}"
                )
                metrics[provider][stratum][metric] = {
                    "mean": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            override_values = []
            for row in subset:
                copy = dict(row)
                copy["override"] = row["evidence_gain_m0"] - row["evidence_gain_mr"]
                override_values.append(copy)
            estimate, low, high = bootstrap_profiles(override_values, "override", f"{provider}:{stratum}:override")
            metrics[provider][stratum]["override"] = {"mean": estimate, "ci_low": low, "ci_high": high}
            for condition, projection_key in (
                ("mo", "mo_direction_projection"),
                ("mr", "mr_direction_projection"),
                ("mo_minus_mr", "mo_minus_mr_direction_projection"),
            ):
                estimate, low, high = bootstrap_direction_beta(
                    subset,
                    projection_key,
                    f"{provider}:{stratum}:{condition}_direction_beta",
                )
                metrics[provider][stratum][f"{condition}_direction_beta"] = {
                    "mean": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
        provider_repeats = [row for row in repeat_rows if row["provider"] == provider]
        if provider_repeats:
            metrics[provider]["repeat_control"] = {}
            for metric in (
                "repeat_tv",
                "repeat_brier_delta",
                "mr_tv_same_profiles",
                "mr_excess_over_repeat_tv",
            ):
                estimate, low, high = bootstrap_profiles(
                    provider_repeats, metric, f"{provider}:repeat:{metric}"
                )
                metrics[provider]["repeat_control"][metric] = {
                    "mean": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }

    macro_average = {}
    for stratum in ("all", *strata):
        subset = [
            row for row in item_rows
            if stratum == "all" or row["stratum"] == stratum
        ]
        macro_average[stratum] = {}
        for metric in metric_keys:
            estimate, low, high = bootstrap_model_macro(
                subset, metric, f"macro:{stratum}:{metric}"
            )
            macro_average[stratum][metric] = {
                "mean": estimate, "ci_low": low, "ci_high": high
            }
        override_rows = []
        for row in subset:
            copy = dict(row)
            copy["override"] = row["evidence_gain_m0"] - row["evidence_gain_mr"]
            override_rows.append(copy)
        estimate, low, high = bootstrap_model_macro(
            override_rows, "override", f"macro:{stratum}:override"
        )
        macro_average[stratum]["override"] = {
            "mean": estimate, "ci_low": low, "ci_high": high
        }
        for condition, projection_key in (
            ("mo", "mo_direction_projection"),
            ("mr", "mr_direction_projection"),
            ("mo_minus_mr", "mo_minus_mr_direction_projection"),
        ):
            estimate, low, high = bootstrap_model_macro_direction(
                subset, projection_key, f"macro:{stratum}:{condition}_direction_beta"
            )
            macro_average[stratum][f"{condition}_direction_beta"] = {
                "mean": estimate, "ci_low": low, "ci_high": high
            }

    first_rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line]
    initial_n = len({row["call_id"] for row in first_rows[:144]})
    initial_success = sum(row["success"] for row in first_rows[:144])
    prompt_repeat_noise = summarize_prompt_repeat_noise(results)
    summary = {
        "scope": "engineering and measurement pilot; not confirmatory paper results",
        "profiles": len({row["profile_id"] for row in results}),
        "providers": providers,
        "analysis_design": {
            "source": design_source,
            "strata": list(strata),
            "target_batches": {
                batch_id: sorted(
                    item_id for item_id, assigned_batch in item_batches.items()
                    if assigned_batch == batch_id
                )
                for batch_id in sorted(set(item_batches.values()))
            },
        },
        "unique_successful_cells": len(results),
        "initial_batch": {"calls": initial_n, "successes": initial_success, "success_rate": initial_success / initial_n},
        "after_transport_requeue": {"planned_cells": len(attempts), "successful_cells": len(results), "completion_rate": len(results) / len(attempts)},
        "prompt_identical_repeat_noise": prompt_repeat_noise,
        "usage": {
            provider: {
                "input_tokens": sum(row.get("input_tokens") or 0 for row in results if row["provider"] == provider),
                "output_tokens": sum(row.get("output_tokens") or 0 for row in results if row["provider"] == provider),
                "reasoning_tokens": sum(row.get("reasoning_tokens") or 0 for row in results if row["provider"] == provider),
                "mean_latency_ms": float(np.mean([row["latency_ms"] for row in results if row["provider"] == provider])),
                "p95_latency_ms": float(np.quantile([row["latency_ms"] for row in results if row["provider"] == provider], 0.95)),
            }
            for provider in providers
        },
        "metrics": metrics,
        "macro_average": macro_average,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
