#!/usr/bin/env python3
"""Run the conservative 504-profile Gate C sensitivity analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot.confirmatory_power import (
    simulate_confirmatory_sensitivity,
    simulate_prospective_macro_design_power,
)
from api_pilot.run_gate_c_strict_inference import (
    COUNTRIES,
    build_inference_rows,
    load_jsonl,
    reference_rows_and_anchors,
    successful_results,
    verify_anchor_match,
)


FIXED_MODEL_PROVIDERS = ("deepseek", "gemini", "mistral", "openai", "qwen")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


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
        "--output-json",
        type=Path,
        default=ROOT / "api_pilot/outputs/confirmatory_504_sensitivity.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "api_pilot/reports/diagnostics/CONFIRMATORY_504_SENSITIVITY.md",
    )
    parser.add_argument(
        "--design-power-json",
        type=Path,
        default=ROOT / "api_pilot/outputs/confirmatory_504_design_power.json",
    )
    parser.add_argument(
        "--design-power-markdown",
        type=Path,
        default=ROOT / "api_pilot/reports/diagnostics/CONFIRMATORY_504_DESIGN_POWER.md",
    )
    parser.add_argument("--calibration-replicates", type=int, default=10_000)
    parser.add_argument("--sensitivity-replicates", type=int, default=10_000)
    parser.add_argument(
        "--supersede-json",
        type=Path,
        default=ROOT / "api_pilot/outputs/confirmatory_504_power.json",
    )
    parser.add_argument(
        "--supersede-markdown",
        type=Path,
        default=ROOT / "api_pilot/reports/archive/CONFIRMATORY_504_POWER.md",
    )
    return parser.parse_args()


def render_design_power_markdown(output: dict[str, Any]) -> str:
    design = output["design"]
    power = output["prospective_power"]
    macro = power["macro_power"]
    null = power["null_calibration"]
    lines = [
        "# 504-profile prospective design-power approximation",
        "",
        f"Generated: `{output['generated_at']}`",
        "",
        "This offline calculation addresses the actual co-primary equal-provider "
        "macro hypotheses for the four official systems. Provider-level results are "
        "variance diagnostics and do not determine the Gate. No API calls were made.",
        "",
        "## Confirmatory decision represented",
        "",
        "- H1 is an intersection-union test: `beta_O > 0` and "
        "`beta_O - beta_R > 0` must both hold.",
        "- H2 tests verified-country Brier utility `> 0`.",
        f"- Holm controls the two family p-values at alpha "
        f"`{design['familywise_alpha']:.3f}`.",
        "- The design Gate requires H1 and H2 to survive Holm in the same simulated "
        f"experiment with probability at least `{design['power_target']:.2f}`.",
        "- SESOI alternatives are beta_O = 0.05, beta_O-beta_R = 0.05, and "
        "verified utility = 0.02. They are planning alternatives, not non-zero nulls.",
        "",
        "## Design and assumptions",
        "",
        f"- Sample: {design['profiles_total']} profiles, "
        f"{design['profiles_per_country']} in each of six true-country strata.",
        f"- Balanced displayed labels: {design['planned_profiles_per_country_label']} "
        "profiles for each of six labels within each true country.",
        "- Joint profile resampling preserves shared profiles, item effects, the "
        "four-system covariance, survey weights, and the covariance among all three "
        "test statistics. Reference records are resampled and anchors rebuilt.",
        "- The centred true-country x original-label pilot error distribution is "
        "assumed exchangeable with the confirmatory run at the larger cell size.",
        "- Exact assignment-test power is not identifiable because each pilot profile "
        "was queried under one displayed label. Labels and outputs are never relabelled, "
        "and no missing potential outcomes are manufactured.",
        "",
        "## Primary macro result",
        "",
        "| Quantity | Estimated probability |",
        "|---|---:|",
        f"| H1 survives Holm | {macro['h1_holm_power']:.3f} |",
        f"| H2 survives Holm | {macro['h2_holm_power']:.3f} |",
        f"| H1 and H2 both survive Holm | {macro['joint_holm_power']:.3f} |",
        f"| Joint-power Monte Carlo 95% Wilson interval | "
        f"[{macro['joint_holm_wilson_low']:.4f}, {macro['joint_holm_wilson_high']:.4f}] |",
        f"| Monte Carlo SE for joint power | {macro['joint_holm_monte_carlo_se']:.4f} |",
        f"| Null probability of any Holm rejection | {null['any_holm_rejection']:.3f} |",
        "",
        f"Prospective macro design Gate: **{'PASS' if output['gate_decision']['passes'] else 'FAIL'}**.",
        "",
        "## Provider variance diagnostics",
        "",
        "| Provider | H1 Holm | H2 Holm | Joint Holm |",
        "|---|---:|---:|---:|",
    ]
    for provider, values in power["provider_diagnostics"].items():
        lines.append(
            f"| {provider} | {values['h1_holm_power']:.3f} | "
            f"{values['h2_holm_power']:.3f} | {values['joint_holm_power']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These rows show where secondary provider estimates may be imprecise. "
            "They neither veto nor replace the equal-provider macro Gate.",
            "",
            "## Prospective stress tests",
            "",
            "These assumption-based scenarios reuse the same joint bootstrap draws "
            "and frozen SESOI alternatives. They are non-gating design diagnostics, "
            "not observed results.",
            "",
            "### Variance inflation",
            "",
            "A variance multiplier of `v` scales centred errors by `sqrt(v)`.",
            "",
            "| Error-variance multiplier | Error-SD multiplier | H1 Holm | H2 Holm | Joint Holm | Reaches 0.80 |",
            "|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for scenario in power["variance_inflation_sensitivity"].values():
        values = scenario["macro_power"]
        lines.append(
            f"| {scenario['variance_multiplier']:.1f} | "
            f"{scenario['standard_deviation_multiplier']:.3f} | "
            f"{values['h1_holm_power']:.3f} | {values['h2_holm_power']:.3f} | "
            f"{values['joint_holm_power']:.3f} | "
            f"{'yes' if scenario['reaches_power_target'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "### Single-provider dropout",
            "",
            "Each row recomputes the equal-provider macro over the three retained "
            "official systems, preserving their covariance in every draw.",
            "",
            "| Excluded provider | Retained providers | H1 Holm | H2 Holm | Joint Holm | Reaches 0.80 |",
            "|---|---|---:|---:|---:|:---:|",
        ]
    )
    for excluded, scenario in power["single_provider_dropout_sensitivity"].items():
        values = scenario["macro_power"]
        lines.append(
            f"| {excluded} | {', '.join(scenario['providers'])} | "
            f"{values['h1_holm_power']:.3f} | {values['h2_holm_power']:.3f} | "
            f"{values['joint_holm_power']:.3f} | "
            f"{'yes' if scenario['reaches_power_target'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Failure of a stress scenario does not retrospectively change the "
            "frozen four-provider design Gate. It identifies a condition under which "
            "the design would need re-powering before confirmatory inference.",
            "",
            "## Interpretation boundary",
            "",
            "This is an assumption-based covariance/bootstrap design-power "
            "approximation. It supports a prospective sample-size decision if the "
            "pilot error structure transfers. It is not exact randomization power, "
            "a confirmatory result, or evidence that the SESOI effects will occur.",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown(output: dict[str, Any]) -> str:
    design = output["design"]
    lines = [
        "# 504-profile confirmatory sensitivity audit",
        "",
        f"Generated: `{output['generated_at']}`",
        "",
        "This is an offline sensitivity analysis based on the completed Gate C "
        "measurement pilot. It is not planned permutation-test power, is not a "
        "confirmatory result, and made no API calls.",
        "",
        "## Design",
        "",
        f"- Planned sample: {design['profiles_total']} profiles, "
        f"{design['profiles_per_country']} per country.",
        f"- Resampling: each observed true-country x displayed-label cell expands "
        f"from {design['pilot_profiles_per_country_label']} to "
        f"{design['planned_profiles_per_country_label']} profiles with replacement. "
        "Original labels and model responses are retained.",
        f"- H1 thresholds: `beta_O > {design['beta_o_null']:.3f}` and "
        f"`beta_O - beta_R > {design['beta_diff_null']:.3f}`; H1 uses an "
        "intersection-union decision.",
        f"- H2 threshold: verified Brier utility > {design['h2_null']:.3f}.",
        f"- One-sided alpha: {design['alpha']:.3f}, used conservatively for each "
        "co-primary family.",
        f"- Simulation: {design['calibration_replicates']:,} independent calibration "
        f"draws and {design['sensitivity_replicates']:,} sensitivity draws.",
        f"- Seeds: calibration `{design['calibration_seed']}`, sensitivity "
        f"`{design['sensitivity_seed']}`.",
        "- Estimation: survey weights within country, then equal country, item, "
        "and provider weighting. Reference respondents are resampled and anchors "
        "rebuilt in every draw.",
        "",
        "## Results",
        "",
        "| Provider | H1 sensitivity | H2 sensitivity | Joint H1 and H2 | Min family | >=80% both |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for provider, values in output["sensitivity"]["results"].items():
        lines.append(
            f"| {provider} | {values['h1_iut_power']:.3f} | "
            f"{values['h2_verified_utility']['estimated_power']:.3f} | "
            f"{values['joint_co_primary_power']:.3f} | "
            f"{values['minimum_co_primary_power']:.3f} | "
            f"{'yes' if values['reaches_80_percent'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "This point-effect sensitivity is non-gating. The prospective design "
            "decision uses the pre-specified equal-provider macro because provider "
            "effects are secondary fixed-case results. The joint column reports both "
            "co-primary families passing in the same draw.",
            "",
            "## Limitations",
            "",
            "- Each country-label cell contains only two Gate C profiles. Resampling "
            "to fourteen per cell cannot reveal heterogeneity absent from the pilot.",
            "- Future effects are assumed to equal Gate C point estimates. The "
            "calculation transfers centred empirical sampling error and does not "
            "posit a generative model for future LLM forecasts.",
            "- Deltas under unobserved alternative displayed labels are unavailable. "
            "This analysis therefore preserves original labels and is not power for "
            "the planned assignment-permutation test.",
            "- The five fixed model channels, including GPT-5.4, define this "
            "sensitivity analysis.",
            "- A sensitivity result above 80% supports the planned sample only under "
            "the pilot-effect and transferred-error assumptions. It does not close "
            "permission or robustness gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = list(manifest["target_strata"]["high_country_value"])
    profiles = load_jsonl(args.profiles)
    results = [
        row
        for row in successful_results(args.results)
        if row["provider"] in FIXED_MODEL_PROVIDERS
    ]
    observed = tuple(sorted({row["provider"] for row in results}))
    if set(observed) != set(FIXED_MODEL_PROVIDERS):
        raise ValueError(
            f"Fixed-model roster mismatch: observed={observed}, "
            f"expected={FIXED_MODEL_PROVIDERS}"
        )
    direction, utility, audit = build_inference_rows(results, profiles, items)
    resampler, anchors, reference_counts = reference_rows_and_anchors(
        args.panel, args.metadata, items
    )
    anchor_error = verify_anchor_match(anchors, args.gate_report, items)
    simulation = simulate_confirmatory_sensitivity(
        direction,
        utility,
        anchors,
        anchor_bootstrap=resampler,
        profiles_per_country=84,
        calibration_replicates=args.calibration_replicates,
        sensitivity_replicates=args.sensitivity_replicates,
        alpha=0.025,
        beta_o_null=0.0,
        beta_diff_null=0.05,
        h2_null=0.02,
        expected_providers=observed,
        expected_countries=COUNTRIES,
        expected_items=items,
    )
    design_power = simulate_prospective_macro_design_power(
        direction,
        utility,
        anchors,
        anchor_bootstrap=resampler,
        profiles_per_country=84,
        calibration_replicates=args.calibration_replicates,
        power_replicates=args.sensitivity_replicates,
        familywise_alpha=0.05,
        power_target=0.80,
        beta_o_alternative=0.05,
        beta_diff_alternative=0.05,
        h2_alternative=0.02,
        expected_providers=observed,
        expected_countries=COUNTRIES,
        expected_items=items,
    )
    models = {
        provider: sorted(
            {row["model_requested"] for row in results if row["provider"] == provider}
        )
        for provider in observed
    }
    if any(len(values) != 1 for values in models.values()):
        raise ValueError(f"Provider has mixed model IDs: {models}")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "offline stratified-bootstrap sensitivity; not permutation-test power "
            "and not confirmatory results"
        ),
        "inputs": {
            "results_sha256": sha256(args.results),
            "profiles_sha256": sha256(args.profiles),
            "manifest_sha256": sha256(args.manifest),
            "gate_report_sha256": sha256(args.gate_report),
            "panel_sha256": sha256(args.panel),
            "metadata_sha256": sha256(args.metadata),
        },
        "code": {
            "runner_sha256": sha256(Path(__file__)),
            "power_module_sha256": sha256(ROOT / "api_pilot/confirmatory_power.py"),
            "strict_inference_sha256": sha256(ROOT / "api_pilot/strict_inference.py"),
        },
        "models": {provider: values[0] for provider, values in models.items()},
        "model_scope": "Five fixed model providers, including GPT-5.4",
        "source_audit": {
            **audit,
            "reference_profiles_by_country": reference_counts,
            "rebuilt_anchor_max_abs_error": anchor_error,
        },
        "design": {
            "profiles_total": simulation.profiles_total,
            "profiles_per_country": simulation.profiles_per_country,
            "pilot_profiles_per_country_label": simulation.pilot_profiles_per_country_label,
            "planned_profiles_per_country_label": simulation.planned_profiles_per_country_label,
            "calibration_replicates": simulation.calibration_replicates,
            "sensitivity_replicates": simulation.sensitivity_replicates,
            "calibration_seed": simulation.calibration_seed,
            "sensitivity_seed": simulation.sensitivity_seed,
            "alpha": simulation.alpha,
            "beta_o_null": simulation.beta_o_null,
            "beta_diff_null": simulation.beta_diff_null,
            "h2_null": simulation.h2_null,
            "decision_rule": "non-gating pilot-point-effect sensitivity diagnostic",
            "method": (
                "with-replacement profile resampling within true-country x original-"
                "displayed-label cells; labels and responses retained; independent "
                "centred empirical-error calibration and sensitivity draws; reference "
                "anchors rebuilt per draw; future effects fixed to Gate C estimates"
            ),
        },
        "sensitivity": simulation.to_dict(),
        "gate_decision": {
            "gating": False,
            "reason": "pilot-point-effect sensitivity is diagnostic, not design power",
            "lowest_provider_diagnostic_sensitivity": (
                simulation.lowest_provider_diagnostic_sensitivity
            ),
            "all_provider_diagnostics_reach_80_percent": (
                simulation.all_provider_diagnostics_reach_80_percent
            ),
            "macro_reaches_80_percent": simulation.macro_reaches_80_percent,
            "prospective_design_power_file": display_path(args.design_power_json),
        },
        "interpretation_guardrail": (
            "This calculation cannot identify planned permutation-test power because "
            "each Gate C delta is observed under one displayed label only. It is a "
            "conservative sensitivity analysis under transferred pilot effects/errors."
        ),
        "limitations": [
            "Only 12 Gate C profiles per country and two profiles per original "
            "country-label cell support the resampling distribution.",
            "Only four Gate C high-country-value items are observed, whereas the "
            "candidate confirmatory manifest contains seven.",
            "Future effects are fixed to Gate C point estimates; only centred "
            "stratified-bootstrap sampling errors are transferred.",
            "The planned assignment-permutation test power remains unidentified from "
            "these observations because alternative-label potential outcomes are absent.",
            "The simulation does not include future API drift, malformed outputs, "
            "or unobserved item/profile heterogeneity.",
        ],
    }
    design_power_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "assumption-based prospective covariance/bootstrap design-power "
            "approximation; not exact assignment power and not confirmatory results"
        ),
        "inputs": output["inputs"],
        "code": output["code"],
        "models": output["models"],
        "model_scope": output["model_scope"],
        "source_audit": output["source_audit"],
        "design": {
            "primary_analysis_unit": "equal-provider macro over four official systems",
            "provider_level_role": "secondary precision diagnostics; never a power gate",
            "profiles_total": design_power.profiles_total,
            "profiles_per_country": design_power.profiles_per_country,
            "pilot_profiles_per_country_label": design_power.pilot_profiles_per_country_label,
            "planned_profiles_per_country_label": design_power.planned_profiles_per_country_label,
            "calibration_replicates": design_power.calibration_replicates,
            "power_replicates": design_power.power_replicates,
            "calibration_seed": design_power.calibration_seed,
            "power_seed": design_power.power_seed,
            "familywise_alpha": design_power.familywise_alpha,
            "power_target": design_power.power_target,
            "null": design_power.null,
            "sesoi_alternative": design_power.sesoi_alternative,
            "decision_rule": (
                "pass only if both equal-provider macro families survive Holm in "
                "at least 80% of simulated 504-profile experiments"
            ),
            "method": (
                "independent joint stratified profile bootstraps within true-country "
                "x original displayed-label cells; centred errors transferred to "
                "fixed SESOI alternatives; provider/statistic covariance retained; "
                "reference anchors rebuilt on every draw"
            ),
        },
        "prospective_power": design_power.to_dict(),
        "gate_decision": {
            "passes": design_power.primary_gate_passes,
            "primary_metric": "macro joint Holm power",
            "estimated_power": design_power.macro_power.joint_holm_power,
            "target": design_power.power_target,
            "provider_diagnostics_are_gating": False,
            "stress_tests_are_gating": False,
        },
        "prospective_stress_test_scope": {
            "gating": False,
            "variance_inflation_factors": [1.5, 2.0],
            "variance_definition": (
                "centred bootstrap errors are multiplied by the square root of "
                "the variance inflation factor"
            ),
            "provider_dropout_definition": (
                "exclude one official provider at a time and recompute the "
                "equal-provider macro over the three retained providers"
            ),
            "uses_frozen_sesoi_alternatives": True,
            "preserves_joint_h1_h2_holm_rule": True,
            "interpretation": (
                "prospective assumption-based design robustness only; no observed "
                "confirmatory outcomes and no change to the frozen primary Gate"
            ),
        },
        "identification": {
            "exact_assignment_test_power_identified": False,
            "reason": (
                "each pilot profile has output under one displayed label, so "
                "alternative-label potential outcomes are unobserved"
            ),
            "assumption": (
                "centred pilot errors within observed true-country x label cells "
                "transfer to balanced 14-profile confirmatory cells"
            ),
            "no_relabelling_or_imputed_potential_outcomes": True,
        },
        "related_non_gating_analysis": {
            "file": display_path(args.output_json),
            "role": (
                "pilot-point-effect sensitivity only; retained for diagnostics and "
                "does not determine the prospective design Gate"
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(output), encoding="utf-8")
    args.design_power_json.parent.mkdir(parents=True, exist_ok=True)
    args.design_power_json.write_text(
        json.dumps(design_power_output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.design_power_markdown.write_text(
        render_design_power_markdown(design_power_output), encoding="utf-8"
    )
    if args.supersede_json.exists() and args.supersede_json != args.output_json:
        previous = json.loads(args.supersede_json.read_text(encoding="utf-8"))
        previous["status"] = "SUPERSEDED: invalid label-reassignment power simulation"
        previous["superseded_by"] = str(args.output_json)
        previous["do_not_use"] = (
            "The old simulation reassigned observed deltas to unobserved displayed "
            "labels and is statistically invalid."
        )
        args.supersede_json.write_text(
            json.dumps(previous, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.supersede_markdown.exists() and args.supersede_markdown != args.output_markdown:
        args.supersede_markdown.write_text(
            "# Superseded\n\nThis report is invalid because it reassigned observed "
            "model deltas to unobserved displayed labels. Use "
            f"[{args.output_markdown.name}]({args.output_markdown.name}).\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "json": str(args.output_json),
                "markdown": str(args.output_markdown),
                "design_power_json": str(args.design_power_json),
                "design_power_markdown": str(args.design_power_markdown),
                "minimum_sensitivity": {
                    provider: values.minimum_co_primary_power
                    for provider, values in simulation.results.items()
                },
                "prospective_macro_gate_passes": design_power.primary_gate_passes,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
