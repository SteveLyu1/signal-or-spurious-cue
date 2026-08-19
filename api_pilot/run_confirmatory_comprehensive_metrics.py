#!/usr/bin/env python3
"""Fail-closed comprehensive secondary analysis for the confirmatory core.

This entry point has no API client. It audits the immutable 32,256-call raw
panel and reconstructs the 64,512-row eight-cell logical panel before analysis. An
absent or incomplete result panel produces only a pending-status artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot.comprehensive_gate_metrics import comprehensive_bootstrap
from api_pilot.run_confirmatory_strict_inference import (
    OFFICIAL_MODELS,
    audit_results,
    expected_call_map,
    latest_rows,
    write_json,
)
from api_pilot.materialize_confirmatory_cached_panel import materialize_logical_rows
from api_pilot.run_gate_c_comprehensive_metrics import METRICS, build_rows
from api_pilot.run_gate_c_strict_inference import (
    COUNTRIES,
    item_result_index,
    load_jsonl,
    reference_rows_and_anchors,
    sha256,
    verify_anchor_match,
)
from api_pilot.survey_pilot_runner import load_task_manifest


STRATUM_ORDER = (
    "high_country_value",
    "representative",
    "equivalence_controls",
    "all_targets",
)
REQUIRED_CELLS = (
    ("M0", "correct"),
    ("MO", "correct"),
    ("MR", "correct"),
    ("MV", "correct"),
    ("M0", "shuffled"),
    ("MO", "shuffled"),
    ("MR", "shuffled"),
    ("MV", "shuffled"),
)


def confirmatory_strata(analysis_manifest: dict[str, Any]) -> dict[str, list[str]]:
    source = analysis_manifest.get("target_strata")
    if not isinstance(source, dict):
        raise ValueError("Analysis manifest lacks target_strata")
    expected_sizes = {
        "high_country_value": 7,
        "representative": 9,
        "equivalence_controls": 4,
    }
    strata: dict[str, list[str]] = {}
    for name, expected_size in expected_sizes.items():
        values = source.get(name)
        if not isinstance(values, list) or len(values) != expected_size:
            raise ValueError(
                f"Confirmatory stratum {name} must contain {expected_size} items"
            )
        if len(set(values)) != len(values) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"Confirmatory stratum {name} is invalid")
        strata[name] = list(values)
    all_targets = [
        item
        for name in (
            "high_country_value",
            "representative",
            "equivalence_controls",
        )
        for item in strata[name]
    ]
    if len(all_targets) != 20 or len(set(all_targets)) != 20:
        raise ValueError("Confirmatory strata must define exactly 20 unique targets")
    frozen_batches = analysis_manifest.get("target_batches")
    if not isinstance(frozen_batches, dict):
        raise ValueError("Analysis manifest lacks target_batches")
    batched = [item for values in frozen_batches.values() for item in values]
    if len(batched) != 20 or set(batched) != set(all_targets):
        raise ValueError("Frozen target batches do not match the 20 target strata")
    strata["all_targets"] = all_targets
    return strata


def audit_formal_item_panel(
    results: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    items: list[str],
) -> dict[str, Any]:
    """Verify option alignment and all six analysis cells before scoring."""
    index = item_result_index(results)
    providers = tuple(sorted({row["provider"] for row in results}))
    if providers != tuple(sorted(OFFICIAL_MODELS)):
        raise ValueError(f"Unexpected official-provider roster: {providers}")
    checked = 0
    for provider in providers:
        for profile in profiles:
            targets = {target["item_id"]: target for target in profile["targets"]}
            for item in items:
                if item not in targets:
                    raise ValueError(
                        f"Profile {profile['profile_id']} lacks target {item}"
                    )
                canonical_codes = [
                    option["answer_code"]
                    for option in profile["target_options"][item]
                ]
                if targets[item]["answer_code"] not in canonical_codes:
                    raise ValueError(
                        f"Gold answer is outside the option set: "
                        f"{profile['profile_id']}/{item}"
                    )
                for condition, evidence in REQUIRED_CELLS:
                    key = (
                        provider,
                        profile["profile_id"],
                        condition,
                        evidence,
                        item,
                    )
                    row = index.get(key)
                    if row is None:
                        raise ValueError(f"Missing formal item cell: {key}")
                    if row.get("true_country_code") != profile["true_country_code"]:
                        raise ValueError(f"True-country mismatch: {key}")
                    if row.get("target_option_codes", {}).get(item) != canonical_codes:
                        raise ValueError(f"Option-code alignment mismatch: {key}")
                    probabilities = np.asarray(
                        row["parsed_forecasts"][item], dtype=float
                    )
                    if (
                        probabilities.ndim != 1
                        or len(probabilities) != len(canonical_codes)
                        or not np.all(np.isfinite(probabilities))
                        or np.any(probabilities < 0)
                        or np.any(probabilities > 1)
                        or not np.isclose(probabilities.sum(), 1.0, atol=1e-12)
                    ):
                        raise ValueError(f"Invalid analysis probability vector: {key}")
                    checked += 1
    expected = len(providers) * len(profiles) * len(items) * len(REQUIRED_CELLS)
    if checked != expected:
        raise ValueError("Formal item-panel audit did not cover every required cell")
    return {
        "complete": True,
        "providers": list(providers),
        "profiles": len(profiles),
        "target_items": len(items),
        "required_cells_per_provider_profile_item": len(REQUIRED_CELLS),
        "item_cells_checked": checked,
        "option_codes_aligned_to_frozen_profiles": True,
        "probability_vectors_analysis_ready": True,
    }


def pending_output(
    args: argparse.Namespace,
    source_audit: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "504-profile confirmatory core comprehensive secondary metrics",
        "status": "pending_inference",
        "estimates_generated": False,
        "hypothesis_decisions_generated": False,
        "reason": reason,
        "source_audit": source_audit,
        "inputs": {
            name: sha256(path) if path.exists() else None
            for name, path in {
                "results": args.results,
                "profiles": args.profiles,
                "analysis_manifest": args.analysis_manifest,
                "task_manifest": args.task_manifest,
                "prompt_bundle": args.prompt_bundle,
            }.items()
        },
    }


def render_markdown(output: dict[str, Any]) -> str:
    if output["status"] == "pending_inference":
        audit = output["source_audit"]
        return "\n".join(
            [
                "# Confirmatory Core Comprehensive Metrics",
                "",
                "> Status: pending inference. No estimate has been generated.",
                "",
                output["reason"],
                "",
                "| Planned calls | Observed unique | Missing | Unexpected | Incomplete |",
                "|---:|---:|---:|---:|---:|",
                f"| {audit['planned']} | {audit['observed_unique']} | "
                f"{audit['missing']} | {audit['unexpected']} | {audit['incomplete']} |",
                "",
            ]
        )

    def fmt(result: dict[str, Any]) -> str:
        return (
            f"{result['estimate']:.4f} "
            f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
        )

    lines = [
        "# Confirmatory Core Comprehensive Metrics",
        "",
        "> Status: complete secondary analysis of the frozen confirmatory core.",
        "",
        "Estimates use equal-country, equal-item and equal-model "
        "aggregation. Confidence intervals use a profile-cluster bootstrap within "
        "true survey country and rebuild the reference anchors in every replicate.",
        "",
        "| Stratum | betaO | betaR | betaO-betaR | RegretO | RegretR | UtilityV | G0 | GO | GR | GV | Override |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    strata = output["analysis"]["provider_sets"]["formal_five"]["strata"]
    for name in STRATUM_ORDER:
        row = strata[name]
        direction = row["direction"]
        prediction = row["prediction"]
        lines.append(
            f"| {name} | {fmt(direction['beta_o']['macro'])} | "
            f"{fmt(direction['beta_r']['macro'])} | "
            f"{fmt(direction['beta_o_minus_beta_r']['macro'])} | "
            f"{fmt(prediction['regret_o']['macro'])} | "
            f"{fmt(prediction['regret_r']['macro'])} | "
            f"{fmt(prediction['utility_v']['macro'])} | "
            f"{fmt(prediction['g0']['macro'])} | "
            f"{fmt(prediction['g_o']['macro'])} | "
            f"{fmt(prediction['g_r']['macro'])} | "
            f"{fmt(prediction['g_v']['macro'])} | "
            f"{fmt(prediction['override']['macro'])} |"
        )
    lines += [
        "",
        "## High minus equivalence interaction",
        "",
        "| Metric | Estimate [95% CI] |",
        "|---|---:|",
    ]
    interaction = output["analysis"]["provider_sets"]["formal_five"][
        "interactions"
    ]["high_minus_equivalence"]
    for metric, result in interaction["direction"].items():
        lines.append(f"| {metric} | {fmt(result['macro'])} |")
    for metric, result in interaction["prediction"].items():
        lines.append(f"| {metric} | {fmt(result['macro'])} |")
    lines += [
        "",
        "These are descriptive secondary estimates. Confirmatory H1 and H2 "
        "decisions remain the responsibility of run_confirmatory_strict_inference.py.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_logical.jsonl",
    )
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_raw.jsonl",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl",
    )
    parser.add_argument(
        "--analysis-manifest",
        type=Path,
        default=ROOT / "dataset/analysis/confirmatory_manifest_v2.json",
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=ROOT
        / "dataset/analysis/confirmatory_core_task_manifest_v2.json",
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
        default=ROOT / "api_pilot/prompts/survey-confirmatory-v4.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "api_pilot/outputs/confirmatory_core_v2_comprehensive_metrics.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "api_pilot/reports/diagnostics/CONFIRMATORY_CORE_V2_COMPREHENSIVE_METRICS.md",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026072523)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_manifest = json.loads(
        args.analysis_manifest.read_text(encoding="utf-8")
    )
    task_manifest = load_task_manifest(args.task_manifest)
    if task_manifest is None:
        raise ValueError("Frozen v2 task manifest is required")
    if task_manifest.get("schema_version") != "survey-task-manifest-v2":
        raise ValueError("Confirmatory analysis requires a v2 task manifest")
    expected_raw = 8_064 * len(OFFICIAL_MODELS)
    if task_manifest.get("expected", {}).get("total_calls") != expected_raw:
        raise ValueError("Confirmatory core count does not match five-model roster")
    if task_manifest.get("provider_models") != OFFICIAL_MODELS:
        raise ValueError("Frozen task manifest does not match the official roster")

    profiles = load_jsonl(args.profiles)
    if (
        len(profiles) != 504
        or sha256(args.profiles)
        != task_manifest["source"]["profile_file_sha256"]
    ):
        raise ValueError("Profile count/hash differs from the frozen task manifest")
    for label, path, expected_hash in (
        (
            "analysis manifest",
            args.analysis_manifest,
            task_manifest["source"]["analysis_manifest_sha256"],
        ),
        ("prompt bundle", args.prompt_bundle, task_manifest["prompt"]["sha256"]),
    ):
        if sha256(path) != expected_hash:
            raise ValueError(f"{label} hash differs from the frozen task manifest")

    strata = confirmatory_strata(analysis_manifest)
    expected = expected_call_map(
        profiles, analysis_manifest, task_manifest, args.prompt_bundle
    )
    raw_rows, source_audit = audit_results(latest_rows(args.raw_results), expected)
    if not source_audit["complete"]:
        output = pending_output(
            args,
            source_audit,
            reason=(
                f"The immutable {expected_raw:,}-call raw core panel is absent, incomplete, or "
                "contains unexpected calls. Secondary estimates are withheld."
            ),
        )
        write_json(args.output, output)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(output), encoding="utf-8")
        print(json.dumps({"status": output["status"], **source_audit}))
        return 0

    expected_logical, cache_audit = materialize_logical_rows(
        raw_rows, profiles, analysis_manifest, task_manifest, args.prompt_bundle
    )
    if not args.results.exists():
        pending = dict(source_audit)
        pending.update({"complete": False, "missing": len(expected_logical)})
        output = pending_output(
            args, pending,
            reason="The audited eight-cell cached logical panel has not been materialized.",
        )
        write_json(args.output, output)
        args.report.write_text(render_markdown(output), encoding="utf-8")
        return 0
    observed_logical = load_jsonl(args.results)
    canonical = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True)
    if [canonical(row) for row in observed_logical] != [
        canonical(row) for row in expected_logical
    ]:
        raise ValueError("Logical panel differs from fail-closed cache reconstruction")
    rows = observed_logical

    item_panel_audit = audit_formal_item_panel(
        rows, profiles, strata["all_targets"]
    )
    direction_rows, scalar_rows, row_audit = build_rows(rows, profiles, strata)
    resampler, anchors, reference_counts = reference_rows_and_anchors(
        args.panel, args.metadata, strata["all_targets"]
    )
    anchor_error = verify_anchor_match(
        anchors, args.gate_report, strata["all_targets"]
    )
    official_providers = tuple(sorted(OFFICIAL_MODELS))
    analysis = comprehensive_bootstrap(
        direction_rows,
        scalar_rows,
        anchors,
        anchor_bootstrap=resampler,
        provider_sets={"formal_five": official_providers},
        expected_providers=official_providers,
        expected_countries=COUNTRIES,
        items_by_stratum=strata,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
        pair_ids_by_profile={
            profile["profile_id"]: profile["pair_id"] for profile in profiles
        },
        pair_cluster_metrics=(
            "g0", "g_o", "g_r", "g_v", "override", "override_o",
            "verified_evidence_interaction",
        ),
    )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "504-profile confirmatory core comprehensive secondary metrics",
        "status": "confirmatory_secondary_metrics_complete",
        "estimates_generated": True,
        "hypothesis_decisions_generated": False,
        "source_audit": source_audit,
        "cache_audit": cache_audit,
        "item_panel_audit": item_panel_audit,
        "row_audit": row_audit,
        "models": OFFICIAL_MODELS,
        "strata": strata,
        "design": {
            "profiles": 504,
            "countries": list(COUNTRIES),
            "target_items": 20,
            "api_batches_per_profile": 4,
            "weighting": (
                "survey weights normalized within true country; equal country, "
                "item and five-model macro aggregation"
            ),
            "metric_definitions": {
                "beta_o": "direction of MO(correct)-M0(correct)",
                "beta_r": "direction of MR(correct)-M0(correct)",
                "beta_o_minus_beta_r": "beta_O-beta_R",
                "regret_o": "Brier(MO correct)-Brier(M0 correct)",
                "regret_r": "Brier(MR correct)-Brier(M0 correct)",
                "utility_v": "Brier(M0 correct)-Brier(MV correct)",
                "g0": "Brier(M0 shuffled)-Brier(M0 correct)",
                "g_o": "Brier(MO shuffled)-Brier(MO correct)",
                "g_r": "Brier(MR shuffled)-Brier(MR correct)",
                "g_v": "Brier(MV shuffled)-Brier(MV correct)",
                "override": "G0-GR",
                "override_o": "G0-GO",
                "verified_evidence_interaction": "G0-GV",
            },
            "interaction": "high_country_value minus equivalence_controls",
        },
        "reference": {
            "profile_counts_by_country": reference_counts,
            "rebuilt_anchor_max_abs_error_vs_gate_report": anchor_error,
            "anchors_reestimated_in_bootstrap": True,
        },
        "inputs": {
            name: sha256(path)
            for name, path in {
                "logical_results": args.results,
                "raw_results": args.raw_results,
                "profiles": args.profiles,
                "analysis_manifest": args.analysis_manifest,
                "task_manifest": args.task_manifest,
                "gate_report": args.gate_report,
                "panel": args.panel,
                "metadata": args.metadata,
                "prompt_bundle": args.prompt_bundle,
            }.items()
        },
        "analysis": analysis,
        "interpretation_guardrail": (
            "These are descriptive secondary estimates for the frozen model, "
            "country, item and prompt roster. H1/H2 decisions are generated only "
            "by the separate strict confirmatory analysis."
        ),
    }
    if tuple(output["analysis"]["provider_sets"]) != ("formal_five",):
        raise ValueError("Comprehensive output contains an unapproved provider set")
    if set(METRICS) != {
        "regret_o", "regret_r", "utility_v", "g0", "g_o", "g_r", "g_v",
        "override", "override_o", "verified_evidence_interaction",
    }:
        raise ValueError("Comprehensive metric contract has changed")
    write_json(args.output, output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(output), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "profiles": 504,
                "targets": 20,
                "bootstrap": args.bootstrap_replicates,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
