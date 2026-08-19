#!/usr/bin/env python3
"""Export paper-ready tables from the authoritative five-model result files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "api_pilot/outputs/final_tables"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def interval(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimate": value.get("estimate"),
        "ci_low": value.get("ci_low"),
        "ci_high": value.get("ci_high"),
    }


def main() -> int:
    source_paths = {
        "strict": ROOT / "api_pilot/outputs/confirmatory_core_v2_strict_inference.json",
        "comprehensive": ROOT / "api_pilot/outputs/confirmatory_core_v2_comprehensive_metrics.json",
        "sentinel": ROOT / "api_pilot/outputs/confirmatory_sentinel_analysis.json",
        "batch": ROOT / "api_pilot/outputs/confirmatory_batch_context_analysis.json",
        "robustness": ROOT / "api_pilot/outputs/five_model_robustness_gate.json",
        "contract": ROOT / "api_pilot/outputs/experiment_completion_contract.json",
        "correction": ROOT / "api_pilot/outputs/checklist_correction_analysis.json",
    }
    independent_completion_path = ROOT / "api_pilot/outputs/independent_correction_replication_v2_completion.json"
    if independent_completion_path.exists():
        independent_completion = load_json(independent_completion_path)
        if independent_completion.get("complete") is True:
            source_paths.update({
                "independent_replication": ROOT / "api_pilot/outputs/independent_correction_replication_v2_analysis.json",
                "independent_completion": independent_completion_path,
            })
    clean_consistency_path = ROOT / "api_pilot/outputs/clean_replacement_primary_consistency.json"
    if clean_consistency_path.exists():
        source_paths["clean_replacement_consistency"] = clean_consistency_path
    loaded = {name: load_json(path) for name, path in source_paths.items()}
    strict = loaded["strict"]
    comprehensive = loaded["comprehensive"]
    sentinel = loaded["sentinel"]
    batch = loaded["batch"]
    robustness = loaded["robustness"]
    contract = loaded["contract"]
    correction = loaded["correction"]
    if not contract.get("machine_complete") or contract.get("failures"):
        raise ValueError("Final tables require a complete five-model machine contract")
    if strict.get("status") != "strict_confirmatory_inference_complete":
        raise ValueError("Strict confirmatory inference is incomplete")
    if comprehensive.get("status") != "confirmatory_secondary_metrics_complete":
        raise ValueError("Comprehensive metrics are incomplete")

    direction = strict["inference"]["direction_bootstrap"]
    utility = strict["inference"]["utility_h2"]
    holm = strict["inference"]["holm_adjusted_p"]
    permutation = strict["inference"]["assignment_permutation"]
    main_rows = [
        {
            "hypothesis": "H1a",
            "metric": "beta_o",
            **interval(direction["beta_o"]),
            "holm_adjusted_p": "",
            "decision": "supported_component",
            "analysis_role": "confirmatory_component",
        },
        {
            "hypothesis": "H1b",
            "metric": "beta_o_minus_beta_r",
            **interval(direction["beta_diff"]),
            "holm_adjusted_p": holm["H1"],
            "preregistered_assignment_p": permutation["h1_iut_p"],
            "decision": "not_supported",
            "analysis_role": "initial_protocol_superseded_by_corrected_wording",
        },
        {
            "hypothesis": "H1",
            "metric": "provenance_sensitive_attenuation_IUT",
            "estimate": direction["beta_diff"]["estimate"],
            "ci_low": direction["beta_diff"]["ci_low"],
            "ci_high": direction["beta_diff"]["ci_high"],
            "holm_adjusted_p": holm["H1"],
            "preregistered_assignment_p": permutation["h1_iut_p"],
            "decision": "not_supported",
            "analysis_role": "initial_protocol_superseded_by_corrected_wording",
        },
        {
            "hypothesis": "H2",
            "metric": "verified_country_brier_utility",
            **interval(utility),
            "holm_adjusted_p": holm["H2"],
            "preregistered_assignment_p": "",
            "decision": "supported_in_initial_version",
            "analysis_role": "initial_protocol_superseded_by_corrected_wording",
        },
    ]
    write_csv(
        OUTPUT_DIR / "table_1_main_hypotheses.csv",
        main_rows,
        [
            "hypothesis", "metric", "estimate", "ci_low", "ci_high",
            "holm_adjusted_p", "preregistered_assignment_p", "decision", "analysis_role",
        ],
    )
    # Keep the historical top-level filename useful without duplicating a
    # separate computation path. Its schema now mirrors the authoritative
    # confirmatory hypothesis export.
    write_csv(
        ROOT / "api_pilot/outputs/final_experiment_results.csv",
        main_rows,
        [
            "hypothesis", "metric", "estimate", "ci_low", "ci_high",
            "holm_adjusted_p", "preregistered_assignment_p", "decision", "analysis_role",
        ],
    )

    formal = comprehensive["analysis"]["provider_sets"]["formal_five"]
    strata_rows: list[dict[str, Any]] = []
    for stratum, content in formal["strata"].items():
        for metric in ("beta_o", "beta_r", "beta_o_minus_beta_r"):
            strata_rows.append({
                "stratum": stratum,
                "metric": metric,
                **interval(content["direction"][metric]["macro"]),
                "analysis_role": "descriptive_secondary",
            })
        for metric in (
            "regret_o", "regret_r", "utility_v", "g0", "override",
            "m0_accuracy", "mo_accuracy_delta", "mr_accuracy_delta",
            "mv_accuracy_delta", "mo_tv", "mr_tv", "mv_tv",
        ):
            strata_rows.append({
                "stratum": stratum,
                "metric": metric,
                **interval(content["prediction"][metric]["macro"]),
                "analysis_role": "descriptive_secondary",
            })
    write_csv(
        OUTPUT_DIR / "table_2_stratum_metrics.csv",
        strata_rows,
        ["stratum", "metric", "estimate", "ci_low", "ci_high", "analysis_role"],
    )

    high = formal["strata"]["high_country_value"]
    provider_rows: list[dict[str, Any]] = []
    for metric in ("beta_o", "beta_r", "beta_o_minus_beta_r"):
        for provider, value in high["direction"][metric]["by_provider"].items():
            provider_rows.append({
                "provider": provider,
                "model": comprehensive["models"][provider],
                "stratum": "high_country_value",
                "metric": metric,
                **interval(value),
                "analysis_role": "descriptive_secondary",
            })
    for metric in ("regret_o", "regret_r", "utility_v", "g0", "override"):
        for provider, value in high["prediction"][metric]["by_provider"].items():
            provider_rows.append({
                "provider": provider,
                "model": comprehensive["models"][provider],
                "stratum": "high_country_value",
                "metric": metric,
                **interval(value),
                "analysis_role": "descriptive_secondary",
            })
    write_csv(
        OUTPUT_DIR / "table_3_model_metrics.csv",
        provider_rows,
        [
            "provider", "model", "stratum", "metric", "estimate", "ci_low",
            "ci_high", "analysis_role",
        ],
    )

    robustness_rows = []
    for name, value in robustness["experiments"].items():
        report = load_json(ROOT / value["report"])
        decision = report["gate_decision"]
        for check, result in decision["checks"].items():
            robustness_rows.append({
                "diagnostic": name,
                "check": check,
                "estimate": result.get("estimate"),
                "ci95_upper": result.get("ci95_upper"),
                "direction_agreement_rate": result.get("direction_agreement_rate"),
                "direction_agreement_ci95_lower": result.get(
                    "direction_agreement_ci95_lower"
                ),
                "passes_prespecified_threshold": result["passes"],
                "pipeline_role": "report_only",
            })
    robustness_rows.extend([
        {
            "diagnostic": "runtime_sentinel",
            "check": "prompt_identical_forecast_tv",
            "estimate": sentinel["metrics"]["overall"]["mean_tv"],
            "ci95_upper": sentinel["metrics"]["overall"]["ci95"][1],
            "direction_agreement_rate": "",
            "direction_agreement_ci95_lower": "",
            "passes_prespecified_threshold": "diagnostic_only",
            "pipeline_role": "report_only",
        },
        {
            "diagnostic": "batch_context_replication",
            "check": "changed_companion_questions_forecast_tv",
            "estimate": batch["overall"]["mean_tv"],
            "ci95_upper": "",
            "direction_agreement_rate": "",
            "direction_agreement_ci95_lower": "",
            "passes_prespecified_threshold": "diagnostic_only",
            "pipeline_role": "report_only",
        },
    ])
    write_csv(
        OUTPUT_DIR / "table_4_robustness_diagnostics.csv",
        robustness_rows,
        [
            "diagnostic", "check", "estimate", "ci95_upper",
            "direction_agreement_rate", "direction_agreement_ci95_lower",
            "passes_prespecified_threshold", "pipeline_role",
        ],
    )

    correction_rows: list[dict[str, Any]] = []
    for scope, versions in correction["exact_subset_comparison"].items():
        for version, content in versions.items():
            inference = content["inference"]
            direction_values = inference["direction_bootstrap"]
            for metric, source in (
                ("beta_o", direction_values["beta_o"]),
                ("beta_r", direction_values["beta_r"]),
                ("beta_o_minus_beta_r", direction_values["beta_diff"]),
                ("utility_v", inference["utility_h2"]),
            ):
                correction_rows.append({
                    "scope": scope,
                    "analysis": version,
                    "metric": metric,
                    **interval(source),
                    "analysis_role": "post_review_matched_subset_sensitivity",
                })
    exclusion = correction["exclusion_sensitivity"]["remaining_high_country_value"]
    inference = exclusion["inference"]
    for metric, source in (
        ("beta_o", inference["direction_bootstrap"]["beta_o"]),
        ("beta_r", inference["direction_bootstrap"]["beta_r"]),
        ("beta_o_minus_beta_r", inference["direction_bootstrap"]["beta_diff"]),
        ("utility_v", inference["utility_h2"]),
    ):
        correction_rows.append({
            "scope": "remaining_high_items_E025_E027",
            "analysis": "exclude_all_eight_affected_items",
            "metric": metric,
            **interval(source),
            "analysis_role": "post_review_exclusion_sensitivity",
        })
    write_csv(
        OUTPUT_DIR / "table_5_checklist_correction_sensitivity.csv",
        correction_rows,
        ["scope", "analysis", "metric", "estimate", "ci_low", "ci_high", "analysis_role"],
    )

    independent = loaded.get("independent_replication")
    if independent:
        inference = independent["result"]["inference"]
        direction_values = inference["direction_bootstrap"]
        utility_value = inference["utility_h2"]
        raw = inference["raw_hypothesis_p"]
        holm = inference["holm_adjusted_p"]
        assignment = inference["assignment_permutation"]
        h1_supported = bool(
            holm["H1"] < 0.05
            and direction_values["beta_o"]["estimate"] > 0
            and direction_values["beta_diff"]["estimate"] > 0
        )
        h2_supported = bool(holm["H2"] < 0.05 and utility_value["estimate"] > 0)
        independent_rows = [
            {
                "hypothesis": "H1a",
                "metric": "beta_o",
                **interval(direction_values["beta_o"]),
                "raw_p": direction_values["beta_o"]["p_greater"],
                "holm_adjusted_p": "",
                "assignment_permutation_p": assignment["p_beta_o_greater"],
                "decision": "component",
                "analysis_role": "primary_sample_disjoint_corrected_protocol",
            },
            {
                "hypothesis": "H1b",
                "metric": "beta_o_minus_beta_r",
                **interval(direction_values["beta_diff"]),
                "raw_p": direction_values["beta_diff"]["p_greater"],
                "holm_adjusted_p": "",
                "assignment_permutation_p": assignment["p_beta_diff_greater"],
                "decision": "component",
                "analysis_role": "primary_sample_disjoint_corrected_protocol",
            },
            {
                "hypothesis": "H1",
                "metric": "provenance_sensitive_attenuation_IUT",
                **interval(direction_values["beta_diff"]),
                "raw_p": raw["H1"],
                "holm_adjusted_p": holm["H1"],
                "assignment_permutation_p": assignment["h1_iut_p"],
                "decision": "supported" if h1_supported else "not_supported",
                "analysis_role": "primary_sample_disjoint_corrected_protocol",
            },
            {
                "hypothesis": "H2",
                "metric": "verified_country_brier_utility",
                **interval(utility_value),
                "raw_p": raw["H2"],
                "holm_adjusted_p": holm["H2"],
                "assignment_permutation_p": "",
                "decision": "supported" if h2_supported else "not_supported",
                "analysis_role": "primary_sample_disjoint_corrected_protocol",
            },
        ]
        write_csv(
            OUTPUT_DIR / "table_6_independent_correction_replication.csv",
            independent_rows,
            [
                "hypothesis", "metric", "estimate", "ci_low", "ci_high",
                "raw_p", "holm_adjusted_p", "assignment_permutation_p",
                "decision", "analysis_role",
            ],
        )

    clean_result = loaded.get("clean_replacement_consistency")
    if clean_result:
        if clean_result.get("status") != "complete":
            raise ValueError("Clean-replacement consistency analysis is incomplete")
        clean_rows = [
            {
                "metric": metric,
                "estimate": values["estimate"],
                "ci_low": values["ci_low"],
                "ci_high": values["ci_high"],
                "bootstrap_p_greater_zero": values["bootstrap_p_greater_zero"],
                "analysis_role": clean_result["analysis_role"],
            }
            for metric, values in clean_result["estimates"].items()
        ]
        write_csv(
            OUTPUT_DIR / "table_17_clean_replacement_consistency.csv",
            clean_rows,
            [
                "metric", "estimate", "ci_low", "ci_high",
                "bootstrap_p_greater_zero", "analysis_role",
            ],
        )

    tables = sorted(OUTPUT_DIR.glob("table_*.csv"))
    manifest = {
        "status": "complete",
        "source_contract_status": contract["status"],
        "machine_complete": True,
        "human_review_complete": bool(contract["human_review"]["passes"]),
        "tables": [path.name for path in tables],
        "table_sha256": {path.name: sha256(path) for path in tables},
        "source_sha256": {
            name: sha256(path) for name, path in source_paths.items()
        },
        "guardrail": (
            "The initial wording version is retained for version history and is superseded for H1/H2 "
            "interpretation. Table 5 is a matched-record wording sensitivity analysis. Table 6, when "
            "present, is the primary sample-disjoint corrected-protocol H1/H2 analysis. Its corrected "
            "wording, sample rule, endpoints and analysis were fixed before its valid outputs. Table 17 "
            "is a key-secondary clean-replacement consistency analysis: two unaffected targets use 504 "
            "records and five corrected targets use a balanced 72-record subset with equal target weight."
        ),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
