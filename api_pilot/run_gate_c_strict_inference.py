#!/usr/bin/env python3
"""Run design-aligned Gate C inference from immutable LLM outputs.

The output is aggregate-only. Respondent-derived profiles, reference rows, and
item-level model forecasts remain in their restricted local files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.tools import build_pilot_gate as gate

from api_pilot.analyze_survey_pilot import brier
from api_pilot.survey_pilot_runner import (
    CELLS,
    build_prompt,
    load_prompt_bundle,
    make_call_id,
    sha256_text,
)
from api_pilot.strict_inference import (
    ReferenceAnchorResampler,
    run_strict_confirmatory_inference,
)


COUNTRIES = tuple(gate.COUNTRIES)
EXPECTED_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.6-flash",
    "mistral": "mistral-medium-3-5",
    "openai": "gpt-5.4",
    "qwen": "qwen3.7-max-2026-06-08",
}
EXPECTED_PROMPT_VERSION = "survey-confirmatory-v3"
EXPECTED_BATCHES = ("B01", "B02")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def successful_results(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("success"):
            latest[row["call_id"]] = row
    if not latest:
        raise ValueError("No successful LLM results")
    return list(latest.values())


def audit_complete_results(
    results: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    prompt_bundle_path: Path,
    *,
    expected_models: dict[str, str] = EXPECTED_MODELS,
) -> dict[str, Any]:
    """Fail closed unless the immutable Gate C result panel matches its design."""
    bundle = load_prompt_bundle(prompt_bundle_path)
    if bundle["version"] != EXPECTED_PROMPT_VERSION:
        raise ValueError(f"Unexpected prompt version: {bundle['version']}")
    profile_by_id = {profile["profile_id"]: profile for profile in profiles}
    if len(profile_by_id) != len(profiles):
        raise ValueError("Duplicate profile IDs in frozen profile file")
    expected_keys: set[tuple[str, str, str, str, str]] = set()
    expected_prompts: dict[tuple[str, str, str, str], tuple[str, str, list[str]]] = {}
    for provider, model in expected_models.items():
        for profile in profiles:
            for batch_index, start in enumerate(range(0, len(profile["targets"]), 5), 1):
                batch_id = f"B{batch_index:02d}"
                if batch_id not in EXPECTED_BATCHES:
                    raise ValueError("Frozen Gate C profile has an unexpected target batch")
                batch_targets = profile["targets"][start : start + 5]
                batch_profile = dict(profile)
                batch_profile["targets"] = batch_targets
                batch_profile["target_options"] = {
                    target["item_id"]: profile["target_options"][target["item_id"]]
                    for target in batch_targets
                }
                for condition, evidence in CELLS:
                    prompt, targets = build_prompt(
                        batch_profile, condition, evidence, bundle
                    )
                    digest = sha256_text(bundle["system_prompt"] + "\n" + prompt)
                    call_id = make_call_id(
                        provider,
                        model,
                        profile["profile_id"],
                        condition,
                        evidence,
                        bundle["version"],
                        batch_id,
                        digest,
                    )
                    key = (provider, profile["profile_id"], batch_id, condition, evidence)
                    expected_keys.add(key)
                    expected_prompts[(profile["profile_id"], batch_id, condition, evidence)] = (
                        call_id.split("/", 2)[-1],
                        digest,
                        [target["item_id"] for target in targets],
                    )

    seen: set[tuple[str, str, str, str, str]] = set()
    repairs = 0
    raw_valid_by_provider: Counter[str] = Counter()
    for row in results:
        provider = str(row.get("provider") or "")
        if provider not in expected_models:
            raise ValueError(f"Unexpected provider in results: {provider}")
        if not row.get("success") or row.get("terminal_failure"):
            raise ValueError(f"Gate C row is not analysis-ready: {row.get('call_id')}")
        if row.get("source_call_id") or row.get("experiment") != "core":
            raise ValueError(f"Unexpected repeat/experiment row: {row.get('call_id')}")
        if (
            row.get("model_requested") != expected_models[provider]
            or row.get("model_returned") != expected_models[provider]
        ):
            raise ValueError(f"Model provenance mismatch: {row.get('call_id')}")
        if row.get("prompt_version") != EXPECTED_PROMPT_VERSION:
            raise ValueError(f"Prompt-version mismatch: {row.get('call_id')}")
        profile_id = str(row.get("profile_id") or "")
        profile = profile_by_id.get(profile_id)
        if profile is None:
            raise ValueError(f"Unknown profile in results: {profile_id}")
        key = (
            provider,
            profile_id,
            str(row.get("batch_id") or ""),
            str(row.get("condition") or ""),
            str(row.get("evidence_variant") or ""),
        )
        if key not in expected_keys or key in seen:
            raise ValueError(f"Unexpected or duplicate result cell: {key}")
        seen.add(key)
        _, digest, target_ids = expected_prompts[key[1:]]
        if row.get("prompt_sha256") != digest or row.get("target_ids") != target_ids:
            raise ValueError(f"Prompt/target digest mismatch: {row.get('call_id')}")
        if not str(row.get("call_id") or "").endswith("/" + digest[:12]):
            raise ValueError(f"Call ID does not bind the prompt digest: {row.get('call_id')}")
        if row.get("true_country_code") != profile["true_country_code"]:
            raise ValueError(f"True-country mismatch: {row.get('call_id')}")
        condition = row["condition"]
        displayed = row.get("displayed_country_code")
        random_code = next(
            code
            for code, label in gate.COUNTRY_LABELS.items()
            if label == profile["balanced_random_country_label"]
        )
        expected_displayed = (
            profile["true_country_code"]
            if condition == "MV"
            else random_code
            if condition in {"MO", "MR"}
            else None
        )
        if displayed != expected_displayed:
            raise ValueError(f"Metadata semantics mismatch: {row.get('call_id')}")
        forecasts = row.get("parsed_forecasts") or {}
        if set(forecasts) != set(target_ids):
            raise ValueError(f"Forecast coverage mismatch: {row.get('call_id')}")
        for item_id, values in forecasts.items():
            array = np.asarray(values, dtype=float)
            option_codes = row.get("target_option_codes", {}).get(item_id) or []
            if (
                array.ndim != 1
                or len(array) != len(option_codes)
                or not np.all(np.isfinite(array))
                or np.any(array < 0)
                or np.any(array > 1)
                or not np.isclose(array.sum(), 1.0, atol=1e-8)
            ):
                raise ValueError(f"Invalid probability vector: {row.get('call_id')}/{item_id}")
        repair = row.get("postprocess_repair")
        if repair:
            repairs += 1
            original_sums = repair.get("original_sums") or {}
            if (
                provider != "mistral"
                or repair.get("method") != "divide_each_probability_by_item_sum"
                or set(original_sums) != set(target_ids)
                or not repair.get("preserves_original_checkpoint")
                or any(
                    item_id != "E039" and not np.isclose(total, 1.0)
                    for item_id, total in original_sums.items()
                )
                or abs(float(original_sums.get("E039", 1.0)) - 1.0) > 0.05 + 1e-12
            ):
                raise ValueError(f"Unapproved postprocess repair: {row.get('call_id')}")
        else:
            raw_valid_by_provider[provider] += 1
    missing = expected_keys.difference(seen)
    extra = seen.difference(expected_keys)
    if missing or extra or len(results) != len(expected_keys):
        raise ValueError(
            f"Incomplete Gate C panel: expected={len(expected_keys)}, observed={len(results)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if repairs != 12:
        raise ValueError(f"Unexpected number of frozen normalization repairs: {repairs}")
    return {
        "complete": True,
        "planned": len(expected_keys),
        "merged": len(results),
        "missing": 0,
        "incomplete": 0,
        "prompt_hashes_recomputed": len(results),
        "model_and_prompt_roster_verified": True,
        "metadata_semantics_verified": True,
        "raw_valid_by_provider": dict(sorted(raw_valid_by_provider.items())),
        "postprocess_repairs": repairs,
    }


def item_result_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("experiment") == "repeat" or (
            row.get("source_call_id") and not row.get("derived_cached")
        ):
            continue
        forecasts = row.get("parsed_forecasts") or {}
        if set(forecasts) != set(row.get("target_ids") or []):
            raise ValueError(f"Forecast coverage mismatch: {row.get('call_id')}")
        for item_id in row["target_ids"]:
            key = (
                row["provider"],
                row["profile_id"],
                row["condition"],
                row["evidence_variant"],
                item_id,
            )
            if key in indexed:
                raise ValueError(f"Duplicate item result: {key}")
            indexed[key] = row
    return indexed


def build_inference_rows(
    results: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    items: list[str],
    *,
    require_balanced_assignment: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index = item_result_index(results)
    providers = sorted({row["provider"] for row in results})
    profile_by_id = {profile["profile_id"]: profile for profile in profiles}
    direction_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    checked_cells = 0
    for provider in providers:
        for profile_id in sorted(profile_by_id):
            profile = profile_by_id[profile_id]
            targets = {target["item_id"]: target for target in profile["targets"]}
            for item_id in items:
                if item_id not in targets:
                    raise ValueError(f"Profile {profile_id} lacks frozen item {item_id}")
                rows = {
                    condition: index.get(
                        (provider, profile_id, condition, "correct", item_id)
                    )
                    for condition in ("M0", "MO", "MR", "MV")
                }
                missing = [condition for condition, row in rows.items() if row is None]
                if missing:
                    raise ValueError(
                        f"Missing cells for {provider}/{profile_id}/{item_id}: {missing}"
                    )
                checked_cells += len(rows)
                canonical_codes = [
                    option["answer_code"]
                    for option in profile["target_options"][item_id]
                ]
                for condition, row in rows.items():
                    if row["target_option_codes"][item_id] != canonical_codes:
                        raise ValueError(
                            f"Option-code mismatch for {provider}/{profile_id}/"
                            f"{item_id}/{condition}"
                        )
                    if row["true_country_code"] != profile["true_country_code"]:
                        raise ValueError("Result/profile true-country mismatch")
                displayed_mo = rows["MO"].get("displayed_country_code")
                displayed_mr = rows["MR"].get("displayed_country_code")
                if displayed_mo != displayed_mr or displayed_mo not in COUNTRIES:
                    raise ValueError(
                        f"MO/MR displayed-label mismatch for {provider}/{profile_id}"
                    )
                expected_displayed = next(
                    code
                    for code, label in gate.COUNTRY_LABELS.items()
                    if label == profile["balanced_random_country_label"]
                )
                if displayed_mo != expected_displayed:
                    raise ValueError(
                        f"Manifest/result displayed-label mismatch for {profile_id}"
                    )
                base = np.asarray(rows["M0"]["parsed_forecasts"][item_id], dtype=float)
                opaque = np.asarray(rows["MO"]["parsed_forecasts"][item_id], dtype=float)
                random = np.asarray(rows["MR"]["parsed_forecasts"][item_id], dtype=float)
                verified = np.asarray(rows["MV"]["parsed_forecasts"][item_id], dtype=float)
                for name, values in (
                    ("M0", base), ("MO", opaque), ("MR", random), ("MV", verified)
                ):
                    if len(values) != len(canonical_codes) or not np.isclose(
                        values.sum(), 1.0, atol=0.01
                    ):
                        raise ValueError(
                            f"Invalid forecast for {provider}/{profile_id}/{item_id}/{name}"
                        )
                weight = float(profile["analysis_weight"])
                common = {
                    "provider": provider,
                    "profile_id": profile_id,
                    "country": profile["true_country_code"],
                    "item_id": item_id,
                    "weight": weight,
                }
                direction_rows.append(
                    {
                        **common,
                        "displayed_label": displayed_mo,
                        "mo_delta": (opaque - base).tolist(),
                        "mr_delta": (random - base).tolist(),
                    }
                )
                truth_index = canonical_codes.index(targets[item_id]["answer_code"])
                utility_rows.append(
                    {
                        **common,
                        "value": brier(base.tolist(), truth_index)
                        - brier(verified.tolist(), truth_index),
                    }
                )
    expected = len(providers) * len(profiles) * len(items)
    if len(direction_rows) != expected or len(utility_rows) != expected:
        raise ValueError("Strict inference rows are incomplete")
    assignment = {
        country: Counter(
            row["displayed_label"]
            for row in direction_rows
            if row["provider"] == providers[0] and row["country"] == country
            and row["item_id"] == items[0]
        )
        for country in COUNTRIES
    }
    expected_per_label = len(profiles) // len(COUNTRIES) // len(COUNTRIES)
    if require_balanced_assignment and any(
        counts != Counter({country: expected_per_label for country in COUNTRIES})
        for counts in assignment.values()
    ):
        raise ValueError(f"Displayed assignment is not balanced: {assignment}")
    audit = {
        "providers": providers,
        "profiles": len(profiles),
        "items": items,
        "direction_rows": len(direction_rows),
        "utility_rows": len(utility_rows),
        "source_cells_checked": checked_cells,
        "balanced_labels_by_true_country": {
            country: dict(counts) for country, counts in assignment.items()
        },
    }
    return direction_rows, utility_rows, audit


def reference_rows_and_anchors(
    panel_path: Path,
    metadata_path: Path,
    items: list[str],
) -> tuple[ReferenceAnchorResampler, dict[str, Any], dict[str, int]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["variables"]
    codes = {item: gate.positive_codes(metadata, item) for item in items}
    code_index = {
        item: {code: index for index, code in enumerate(values)}
        for item, values in codes.items()
    }
    rows = []
    counts: Counter[str] = Counter()
    with panel_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"uniqid", "cntry_AN", "gwght", *items}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference panel lacks columns: {sorted(missing)}")
        for source in reader:
            if gate.respondent_split(source["uniqid"]) != "reference":
                continue
            country = source["cntry_AN"]
            if country not in COUNTRIES:
                continue
            counts[country] += 1
            weight = gate.survey_weight(source)
            for item in items:
                answer_index = code_index[item].get(source[item], -1)
                rows.append(
                    {
                        "country": country,
                        "profile_id": source["uniqid"],
                        "item_id": item,
                        "answer_index": answer_index,
                        "n_options": len(codes[item]),
                        "weight": weight,
                    }
                )
    resampler = ReferenceAnchorResampler(
        rows, expected_countries=COUNTRIES, expected_items=items
    )
    return resampler, resampler.estimate(), dict(counts)


def verify_anchor_match(
    anchors: dict[str, Any], gate_report_path: Path, items: list[str]
) -> float:
    report = json.loads(gate_report_path.read_text(encoding="utf-8"))
    expected = {row["variable"]: row["reference_distributions"] for row in report["results"]}
    maximum = 0.0
    for item in items:
        for country in COUNTRIES:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            np.asarray(anchors[item][country])
                            - np.asarray(expected[item][country])
                        )
                    )
                ),
            )
    if maximum > 1e-12:
        raise ValueError(f"Rebuilt anchors differ from frozen Gate report: {maximum}")
    return maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "api_pilot/restricted_outputs/gate_c_v3_complete.jsonl",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=ROOT / "dataset/derived/gate_c_profiles_6x12.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dataset/analysis/gate_c_manifest_v1.json",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=ROOT / "dataset/analysis/pilot_gate_report.json",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=ROOT / "dataset/derived/country_panel_6.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "dataset/metadata/variable_metadata.json",
    )
    parser.add_argument(
        "--prompt-bundle",
        type=Path,
        default=ROOT / "api_pilot/prompts/survey-confirmatory-v3.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "api_pilot/outputs/gate_c_v3_strict_inference.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument(
        "--providers",
        nargs="+",
        help=(
            "Optional frozen provider subset. Rows from all other providers are "
            "excluded before constructing the complete paired panel."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = list(manifest["target_strata"]["high_country_value"])
    profiles = load_jsonl(args.profiles)
    if len(profiles) != manifest["profiles"]["count"]:
        raise ValueError("Profile count differs from Gate manifest")
    if sha256(args.profiles) != manifest["profiles"]["restricted_file_sha256"]:
        raise ValueError("Profile hash differs from Gate manifest")
    results = successful_results(args.results)
    source_audit = audit_complete_results(results, profiles, args.prompt_bundle)
    all_providers = sorted({row["provider"] for row in results})
    if args.providers:
        requested_providers = list(dict.fromkeys(args.providers))
        unknown = sorted(set(requested_providers).difference(all_providers))
        if unknown:
            raise ValueError(f"Requested providers are absent: {unknown}")
        results = [
            row for row in results if row["provider"] in requested_providers
        ]
    else:
        requested_providers = all_providers
    direction_rows, utility_rows, row_audit = build_inference_rows(
        results, profiles, items
    )
    resampler, anchors, reference_counts = reference_rows_and_anchors(
        args.panel, args.metadata, items
    )
    anchor_error = verify_anchor_match(anchors, args.gate_report, items)
    inference = run_strict_confirmatory_inference(
        direction_rows,
        utility_rows,
        anchors,
        anchor_bootstrap=resampler,
        n_bootstrap=args.bootstrap_replicates,
        n_permutations=args.permutations,
        expected_providers=row_audit["providers"],
        expected_countries=COUNTRIES,
        expected_items=items,
    )
    models = {
        provider: sorted(
            {row["model_requested"] for row in results if row["provider"] == provider}
        )
        for provider in row_audit["providers"]
    }
    if any(len(values) != 1 for values in models.values()):
        raise ValueError(f"Provider has mixed model IDs: {models}")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Gate C measurement pilot; not confirmatory paper results",
        "status": "strict design-aligned aggregate inference",
        "inputs": {
            "results_sha256": sha256(args.results),
            "profiles_sha256": sha256(args.profiles),
            "manifest_sha256": sha256(args.manifest),
            "gate_report_sha256": sha256(args.gate_report),
            "panel_sha256": sha256(args.panel),
            "metadata_sha256": sha256(args.metadata),
            "prompt_bundle_sha256": sha256(args.prompt_bundle),
        },
        "models": {provider: values[0] for provider, values in models.items()},
        "provider_selection": {
            "included": row_audit["providers"],
            "excluded": sorted(set(all_providers).difference(row_audit["providers"])),
        },
        "model_configuration_note": (
            "The frozen five-model roster includes GPT-5.4 with its Responses API "
            "request settings and returned model identifier recorded."
        ),
        "estimand": {
            "stratum": "high_country_value",
            "items": items,
            "weighting": (
                "survey weights normalized within true survey country; equal country, "
                "item, and model macro aggregation"
            ),
            "direction_ci": (
                "profile-cluster bootstrap with reference respondents resampled within "
                "country and anchors rebuilt on every replicate"
            ),
            "direction_test": (
                "weak-null H1 uses a paired profile/reference cluster bootstrap; a "
                "separate unstudentized balanced-label permutation tests the stronger "
                "Fisher sharp null and is reported only as a diagnostic"
            ),
            "h1": (
                "primary IUT max[cluster-bootstrap p(betaO<=0), "
                "cluster-bootstrap p(betaO-betaR<=0)]"
            ),
            "sesoi_note": (
                "betaO-betaR=0.05 and verified Brier utility=0.02 are design "
                "alternatives and interpretation thresholds, not primary nulls"
            ),
            "h2": (
                "paired profile bootstrap p for verified-country Brier utility > 0"
            ),
            "multiplicity": "Holm across H1 and H2",
        },
        "replicates": {
            "profile_reference_bootstrap": args.bootstrap_replicates,
            "assignment_permutations": args.permutations,
        },
        "reference": {
            "profile_counts_by_country": reference_counts,
            "rebuilt_anchor_max_abs_error_vs_gate_report": anchor_error,
            "anchors_reestimated_in_bootstrap": True,
        },
        "row_audit": row_audit,
        "source_audit": source_audit,
        "inference": inference.to_dict(),
        "primary_hypothesis_tests": {
            "H1": max(
                inference.direction_bootstrap.beta_o.p_greater,
                inference.direction_bootstrap.beta_diff.p_greater,
            ),
            "H2": inference.utility_h2.p_greater,
            "note": (
                "These are the preregistered one-sided tests against zero. H1 is "
                "the intersection-union maximum; Holm adjustment is stored in "
                "inference.holm_adjusted_p."
            ),
        },
        "sesoi_exceedance_diagnostics": {
            "H1": inference.raw_sesoi_exceedance_p["H1"],
            "H2": inference.raw_sesoi_exceedance_p["H2"],
            "note": (
                "Secondary tests against the design alternatives betaO=0, "
                "betaO-betaR=0.05, and UtilityV=0.02. They describe scientific "
                "magnitude and do not replace the primary zero-effect hypotheses."
            ),
        },
        "interpretation_guardrail": (
            "This file supersedes the old unweighted Gate C intervals and normal-"
            "approximation pass. It does not by itself close robustness, data-"
            "processing-permission or confirmatory-design gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "beta_o": inference.assignment_permutation.beta_o,
                "beta_r": inference.assignment_permutation.beta_r,
                "beta_diff": inference.assignment_permutation.beta_diff,
                "h1_primary_zero_effect_p": inference.raw_hypothesis_p["H1"],
                "h1_sesoi_exceedance_p": inference.raw_sesoi_exceedance_p["H1"],
                "h1_fisher_sharp_null_p": inference.assignment_permutation.h1_iut_p,
                "utility": inference.utility_h2.estimate,
                "h2_primary_zero_effect_p": inference.raw_hypothesis_p["H2"],
                "h2_sesoi_exceedance_p": inference.raw_sesoi_exceedance_p["H2"],
                "holm": inference.holm_adjusted_p,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
