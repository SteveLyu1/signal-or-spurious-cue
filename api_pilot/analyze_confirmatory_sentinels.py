#!/usr/bin/env python3
"""Fail-closed offline analysis of formal runtime sentinel repeats.

No estimate is emitted unless the 32,256-call cached formal core and all three
192-call sentinel waves pass exact frozen-result audits.  Prompt-identical
forecasts are aligned by answer code before total variation is calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot import build_confirmatory_sentinel_manifests as builder
from api_pilot.run_confirmatory_strict_inference import (
    audit_results as audit_core_results,
    latest_rows,
    normalize_forecasts,
)
from api_pilot.run_confirmatory_strict_inference import expected_call_map
from api_pilot.survey_pilot_runner import load_task_manifest


DEFAULT_REPORT = ROOT / "api_pilot/outputs/confirmatory_sentinel_analysis.json"
BOOTSTRAP_SEED = 2026072501


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (
        ordered[upper] - ordered[lower]
    )


def probability_map(row: dict[str, Any], item_id: str) -> dict[str, float]:
    codes = row["target_option_codes"][item_id]
    probabilities = row["parsed_forecasts"][item_id]
    if len(codes) != len(probabilities):
        raise ValueError(f"Answer-code/probability mismatch: {item_id}")
    mapped = {str(code): float(value) for code, value in zip(codes, probabilities)}
    if len(mapped) != len(codes):
        raise ValueError(f"Duplicate answer code: {item_id}")
    return mapped


def total_variation(first: dict[str, float], second: dict[str, float]) -> float:
    if set(first) != set(second):
        raise ValueError("Paired sentinel forecasts use different answer-code sets")
    return 0.5 * sum(abs(first[code] - second[code]) for code in first)


def expected_result_experiment(
    manifest: dict[str, Any], provider: str
) -> str:
    """Return the immutable source experiment retained by a merged panel row."""
    amendment = manifest.get("roster_amendment") or {}
    if provider == "openai":
        return str(
            amendment.get("openai_source_experiment_id")
            or manifest["experiment_id"]
        )
    return str(amendment.get("source_experiment_id") or manifest["experiment_id"])


def audit_sentinel_results(
    latest: dict[str, dict[str, Any]],
    expected_core_specs: dict[str, dict[str, Any]],
    audited_core_rows: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {
        source_id + "/ROBUST-REPEAT-01": (source_id, spec)
        for source_id, spec in expected_core_specs.items()
    }
    observed_ids = set(latest)
    expected_ids = set(expected)
    missing = expected_ids - observed_ids
    extra = observed_ids - expected_ids
    incomplete = [
        call_id for call_id in expected_ids & observed_ids
        if not latest[call_id].get("success")
        or latest[call_id].get("terminal_failure")
    ]
    if missing or extra or incomplete:
        return [], {
            "complete": False,
            "planned": len(expected),
            "observed_unique": len(observed_ids),
            "missing": len(missing),
            "unexpected": len(extra),
            "incomplete": len(incomplete),
            "analysis_ready": 0,
        }
    rows: list[dict[str, Any]] = []
    normalized_items = 0
    by_provider: Counter[str] = Counter()
    for call_id in sorted(expected):
        source_id, spec = expected[call_id]
        if source_id not in audited_core_rows:
            raise ValueError(f"Sentinel source call absent from audited core: {source_id}")
        repeat = latest[call_id]
        core = audited_core_rows[source_id]
        for field in (
            "provider", "model_requested", "model_returned", "prompt_version",
            "profile_id", "true_country_code", "condition", "evidence_variant",
            "batch_id", "target_ids", "target_option_codes",
            "displayed_country_code", "prompt_sha256",
        ):
            if repeat.get(field) != core.get(field):
                raise ValueError(f"Sentinel/core mismatch for {field}: {call_id}")
        if repeat.get("source_call_id") != source_id:
            raise ValueError(f"Sentinel source_call_id mismatch: {call_id}")
        if repeat.get("experiment") != expected_result_experiment(
            manifest, str(repeat.get("provider"))
        ):
            raise ValueError(f"Sentinel experiment mismatch: {call_id}")
        normalized, changed = normalize_forecasts(repeat, spec)
        normalized_items += changed
        rows.append(normalized)
        by_provider[repeat["provider"]] += 1
    if len(rows) != manifest["expected"]["total_calls"]:
        raise ValueError("Sentinel audit count differs from frozen manifest")
    return rows, {
        "complete": True,
        "planned": len(expected),
        "observed_unique": len(observed_ids),
        "missing": 0,
        "unexpected": 0,
        "incomplete": 0,
        "analysis_ready": len(rows),
        "by_provider": dict(sorted(by_provider.items())),
        "locally_normalized_item_vectors": normalized_items,
        "raw_results_unchanged": True,
    }


def source_specs_for_manifest(
    manifest: dict[str, Any],
    core_expected: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Subset the already reconstructed core calls to exact sentinel units."""
    units = {
        (
            unit["profile_id"], unit["condition"],
            unit["evidence_variant"], unit["batch_id"],
        )
        for unit in manifest["task_units"]
    }
    selected = {
        call_id: spec
        for call_id, spec in core_expected.items()
        if (
            spec["profile"]["profile_id"], spec["condition"],
            spec["evidence_variant"], spec["batch_id"],
        ) in units
    }
    if len(selected) != manifest["expected"]["total_calls"]:
        raise ValueError("Sentinel units do not map exactly to cached core calls")
    return selected


def pair_rows(
    wave: str,
    sentinel_rows: Iterable[dict[str, Any]],
    core_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for repeat in sentinel_rows:
        core = core_rows[repeat["source_call_id"]]
        for item_id in repeat["target_ids"]:
            first = probability_map(core, item_id)
            second = probability_map(repeat, item_id)
            pairs.append(
                {
                    "wave": wave,
                    "provider": repeat["provider"],
                    "profile_id": repeat["profile_id"],
                    "condition": repeat["condition"],
                    "evidence_variant": repeat["evidence_variant"],
                    "cell": (
                        f"{repeat['condition']}/{repeat['evidence_variant']}"
                    ),
                    "batch_id": repeat["batch_id"],
                    "item_id": item_id,
                    "tv": total_variation(first, second),
                }
            )
    return pairs


def cluster_summary(
    rows: list[dict[str, Any]], replicates: int, salt: str
) -> dict[str, Any]:
    by_profile: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_profile[row["profile_id"]].append(float(row["tv"]))
    profile_means = {
        profile_id: statistics.fmean(values)
        for profile_id, values in by_profile.items()
    }
    if not profile_means:
        raise ValueError("Cannot summarise an empty sentinel subset")
    point = statistics.fmean(float(row["tv"]) for row in rows)
    seed = BOOTSTRAP_SEED + int.from_bytes(
        hashlib.sha256(salt.encode()).digest()[:4], "big"
    )
    rng = random.Random(seed)
    profile_ids = sorted(profile_means)
    draws = []
    for _ in range(replicates):
        sampled = [rng.choice(profile_ids) for _ in profile_ids]
        draws.append(
            statistics.fmean(profile_means[profile_id] for profile_id in sampled)
        )
    values = [float(row["tv"]) for row in rows]
    return {
        "mean_tv": point,
        "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "median_tv": quantile(values, 0.5),
        "p90_tv": quantile(values, 0.9),
        "maximum_tv": max(values),
        "profiles": len(profile_ids),
        "paired_item_forecasts": len(rows),
        "bootstrap_replicates": replicates,
        "cluster_unit": "profile_id",
    }


def grouped_summaries(
    rows: list[dict[str, Any]], fields: tuple[str, ...], replicates: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    return {
        "/".join(key): cluster_summary(
            subset, replicates, "|".join((*fields, *key))
        )
        for key, subset in sorted(grouped.items())
    }


def wave_contrasts(
    rows: list[dict[str, Any]], replicates: int
) -> dict[str, Any]:
    """Estimate after-before and mid-before drift contrasts by provider.

    Sentinel profiles are intentionally disjoint across waves, so the bootstrap
    resamples profile clusters independently within each wave and reports a
    difference of wave means rather than pretending observations are paired.
    """
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(row["provider"], row["wave"])][row["profile_id"]].append(
            float(row["tv"])
        )

    def contrast(provider: str, first: str, second: str) -> dict[str, Any]:
        first_values = {
            profile: statistics.fmean(values)
            for profile, values in grouped[(provider, first)].items()
        }
        second_values = {
            profile: statistics.fmean(values)
            for profile, values in grouped[(provider, second)].items()
        }
        if not first_values or not second_values:
            raise ValueError("Runtime-wave contrast has an empty provider/wave cell")
        point = statistics.fmean(second_values.values()) - statistics.fmean(
            first_values.values()
        )
        rng = random.Random(
            BOOTSTRAP_SEED
            + int.from_bytes(
                hashlib.sha256(
                    f"{provider}|{first}|{second}".encode()
                ).digest()[:4],
                "big",
            )
        )
        first_ids = sorted(first_values)
        second_ids = sorted(second_values)
        draws = []
        for _ in range(replicates):
            first_draw = statistics.fmean(
                first_values[rng.choice(first_ids)] for _ in first_ids
            )
            second_draw = statistics.fmean(
                second_values[rng.choice(second_ids)] for _ in second_ids
            )
            draws.append(second_draw - first_draw)
        return {
            "estimate": point,
            "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)],
            "profiles_first": len(first_ids),
            "profiles_second": len(second_ids),
            "bootstrap_replicates": replicates,
            "independent_profile_clusters_by_wave": True,
        }

    providers = sorted({row["provider"] for row in rows})
    return {
        provider: {
            "mid_minus_before": contrast(provider, "before", "mid"),
            "after_minus_before": contrast(provider, "before", "after"),
        }
        for provider in providers
    }


def pending_result(
    suite_path: Path,
    reason: str,
    core_audit: dict[str, Any],
    wave_audits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "confirmatory-sentinel-analysis-v1",
        "status": "pending_inference",
        "complete": False,
        "estimates_generated": False,
        "reason": reason,
        "suite_file": suite_path.relative_to(ROOT).as_posix(),
        "suite_sha256": sha256(suite_path),
        "core_audit": core_audit,
        "wave_audits": wave_audits,
    }


def analyse(
    suite_path: Path = builder.SUITE_PATH,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    suite = load_json(suite_path)
    core_manifest_path = ROOT / suite["core_manifest_file"]
    if sha256(core_manifest_path) != suite["core_manifest_sha256"]:
        raise ValueError("Formal core manifest hash differs from sentinel freeze")
    core_manifest = load_json(core_manifest_path)
    profiles = [json.loads(line) for line in (ROOT / core_manifest["source"]["profile_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    analysis_manifest = load_json(ROOT / core_manifest["source"]["analysis_manifest_file"])
    # Core v2 carries prospective-amendment and logical-cell audit fields that
    # are deliberately outside the runner's execution schema.  The suite hash
    # above binds the exact file, so analysis reads it without dropping fields.
    core_expected = expected_call_map(
        profiles, analysis_manifest, core_manifest, ROOT / core_manifest["prompt"]["file"]
    )
    core_output = ROOT / core_manifest["immutable_output_path"]
    core_rows, core_audit = audit_core_results(
        latest_rows(core_output), core_expected
    )
    wave_audits: dict[str, Any] = {}
    if not core_audit["complete"]:
        return pending_result(
            suite_path,
            "formal core is absent or incomplete; no sentinel estimate was generated",
            core_audit,
            wave_audits,
        )
    core_by_call = {row["call_id"]: row for row in core_rows}
    audited_waves: dict[str, list[dict[str, Any]]] = {}
    for wave in builder.WAVES:
        entry = suite["waves"][wave]
        manifest_path = ROOT / entry["manifest"]
        if sha256(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"Sentinel manifest hash differs from suite: {wave}")
        manifest = load_task_manifest(manifest_path)
        if manifest is None:
            raise ValueError(f"Sentinel manifest absent: {wave}")
        expected_specs = source_specs_for_manifest(manifest, core_expected)
        output_path = ROOT / manifest["immutable_output_path"]
        rows, audit = audit_sentinel_results(
            latest_rows(output_path), expected_specs, core_by_call, manifest
        )
        wave_audits[wave] = audit
        if not audit["complete"]:
            return pending_result(
                suite_path,
                f"sentinel wave {wave} is absent or incomplete; no estimate was generated",
                core_audit,
                wave_audits,
            )
        audited_waves[wave] = rows
    pairs = [
        pair
        for wave in builder.WAVES
        for pair in pair_rows(wave, audited_waves[wave], core_by_call)
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "confirmatory-sentinel-analysis-v1",
        "status": "complete",
        "complete": True,
        "estimates_generated": True,
        "role": "runtime drift diagnostic; not a primary hypothesis test",
        "suite_file": suite_path.relative_to(ROOT).as_posix(),
        "suite_sha256": sha256(suite_path),
        "core_audit": core_audit,
        "wave_audits": wave_audits,
        "paired_item_forecasts": len(pairs),
        "metrics": {
            "overall": cluster_summary(pairs, bootstrap_replicates, "overall"),
            "by_wave": grouped_summaries(
                pairs, ("wave",), bootstrap_replicates
            ),
            "by_provider": grouped_summaries(
                pairs, ("provider",), bootstrap_replicates
            ),
            "by_cell": grouped_summaries(
                pairs, ("cell",), bootstrap_replicates
            ),
            "by_wave_provider_cell": grouped_summaries(
                pairs, ("wave", "provider", "cell"), bootstrap_replicates
            ),
            "wave_contrasts_by_provider": wave_contrasts(
                pairs, bootstrap_replicates
            ),
        },
        "interpretation": (
            "TV quantifies prompt-identical forecast drift. Confidence intervals "
            "resample formal profile clusters. These diagnostics do not alter or "
            "replace the preregistered confirmatory hypotheses."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=builder.SUITE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyse(args.suite.resolve(), args.bootstrap_replicates)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"status={result['status']} complete={result['complete']} "
        f"estimates_generated={result['estimates_generated']} api_requests_made=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
