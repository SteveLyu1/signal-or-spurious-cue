#!/usr/bin/env python3
"""Build aggregate-only comprehensive metrics for the frozen Gate C pilot."""

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

from api_pilot.analyze_survey_pilot import (
    argmax_accuracy,
    brier,
    ordinal_crps,
    total_variation,
)
from api_pilot.comprehensive_gate_metrics import comprehensive_bootstrap
from api_pilot.run_gate_c_strict_inference import (
    COUNTRIES,
    audit_complete_results,
    item_result_index,
    load_jsonl,
    reference_rows_and_anchors,
    sha256,
    successful_results,
    verify_anchor_match,
)


METRICS = (
    "regret_o", "regret_r", "utility_v", "g0", "g_o", "g_r", "g_v",
    "override", "override_o", "verified_evidence_interaction",
)
AUXILIARY_METRICS = (
    "m0_accuracy", "mo_accuracy_delta", "mr_accuracy_delta",
    "mv_accuracy_delta", "mo_tv", "mr_tv", "mv_tv",
    "m0_ordinal_crps", "mo_ordinal_crps_delta",
    "mr_ordinal_crps_delta", "mv_ordinal_crps_delta",
)
ORDINAL_TARGETS = {
    "A004", "A005", "A042", "D060", "D061", "D078",
    "E018", "E025", "E027", "E036", "E039", "F118", "F121",
}
CORE_FOUR_MODELS = ("deepseek", "gemini", "mistral", "qwen")


def _forecast(index: dict[tuple[str, str, str, str, str], dict[str, Any]], key: tuple[str, str, str, str, str]) -> np.ndarray:
    row = index.get(key)
    if row is None:
        raise ValueError(f"Missing frozen item result: {key}")
    return np.asarray(row["parsed_forecasts"][key[-1]], dtype=float)


def build_rows(
    results: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    items_by_stratum: dict[str, list[str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    index = item_result_index(results)
    providers = sorted({row["provider"] for row in results})
    direction: dict[str, list[dict[str, Any]]] = {}
    scalar: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for stratum, items in items_by_stratum.items():
        direction[stratum] = []
        scalar[stratum] = {
            metric: [] for metric in (*METRICS, *AUXILIARY_METRICS)
        }
        for provider in providers:
            for profile in profiles:
                profile_id = profile["profile_id"]
                targets = {target["item_id"]: target for target in profile["targets"]}
                for item in items:
                    common = {
                        "provider": provider,
                        "profile_id": profile_id,
                        "country": profile["true_country_code"],
                        "item_id": item,
                        "weight": float(profile["analysis_weight"]),
                    }
                    key = lambda condition, evidence: (provider, profile_id, condition, evidence, item)
                    p0 = _forecast(index, key("M0", "correct"))
                    po = _forecast(index, key("MO", "correct"))
                    pr = _forecast(index, key("MR", "correct"))
                    pv = _forecast(index, key("MV", "correct"))
                    p0s = _forecast(index, key("M0", "shuffled"))
                    pos = _forecast(index, key("MO", "shuffled"))
                    prs = _forecast(index, key("MR", "shuffled"))
                    pvs = _forecast(index, key("MV", "shuffled"))
                    displayed = index[key("MO", "correct")]["displayed_country_code"]
                    if displayed != index[key("MR", "correct")]["displayed_country_code"]:
                        raise ValueError("MO and MR displayed labels differ")
                    codes = index[key("M0", "correct")]["target_option_codes"][item]
                    truth = codes.index(targets[item]["answer_code"])
                    losses = {
                        "m0": brier(p0.tolist(), truth),
                        "mo": brier(po.tolist(), truth),
                        "mr": brier(pr.tolist(), truth),
                        "mv": brier(pv.tolist(), truth),
                        "m0s": brier(p0s.tolist(), truth),
                        "mos": brier(pos.tolist(), truth),
                        "mrs": brier(prs.tolist(), truth),
                        "mvs": brier(pvs.tolist(), truth),
                    }
                    values = {
                        "regret_o": losses["mo"] - losses["m0"],
                        "regret_r": losses["mr"] - losses["m0"],
                        "utility_v": losses["m0"] - losses["mv"],
                        "g0": losses["m0s"] - losses["m0"],
                        "g_o": losses["mos"] - losses["mo"],
                        "g_r": losses["mrs"] - losses["mr"],
                        "g_v": losses["mvs"] - losses["mv"],
                    }
                    values["override"] = values["g0"] - values["g_r"]
                    values["override_o"] = values["g0"] - values["g_o"]
                    values["verified_evidence_interaction"] = (
                        values["g0"] - values["g_v"]
                    )
                    values.update({
                        "m0_accuracy": argmax_accuracy(p0.tolist(), truth),
                        "mo_accuracy_delta": (
                            argmax_accuracy(po.tolist(), truth)
                            - argmax_accuracy(p0.tolist(), truth)
                        ),
                        "mr_accuracy_delta": (
                            argmax_accuracy(pr.tolist(), truth)
                            - argmax_accuracy(p0.tolist(), truth)
                        ),
                        "mv_accuracy_delta": (
                            argmax_accuracy(pv.tolist(), truth)
                            - argmax_accuracy(p0.tolist(), truth)
                        ),
                        "mo_tv": total_variation(p0.tolist(), po.tolist()),
                        "mr_tv": total_variation(p0.tolist(), pr.tolist()),
                        "mv_tv": total_variation(p0.tolist(), pv.tolist()),
                    })
                    if item in ORDINAL_TARGETS:
                        m0_crps = ordinal_crps(p0.tolist(), truth)
                        values.update({
                            "m0_ordinal_crps": m0_crps,
                            "mo_ordinal_crps_delta": (
                                ordinal_crps(po.tolist(), truth) - m0_crps
                            ),
                            "mr_ordinal_crps_delta": (
                                ordinal_crps(pr.tolist(), truth) - m0_crps
                            ),
                            "mv_ordinal_crps_delta": (
                                ordinal_crps(pv.tolist(), truth) - m0_crps
                            ),
                        })
                    direction[stratum].append(
                        {
                            **common,
                            "displayed_label": displayed,
                            "mo_delta": (po - p0).tolist(),
                            "mr_delta": (pr - p0).tolist(),
                        }
                    )
                    for metric, value in values.items():
                        scalar[stratum][metric].append({**common, "value": value})
    expected = len(providers) * len(profiles)
    for stratum, items in items_by_stratum.items():
        if len(direction[stratum]) != expected * len(items):
            raise ValueError(f"Incomplete direction panel for {stratum}")
        full_count = expected * len(items)
        ordinal_count = expected * len(set(items).intersection(ORDINAL_TARGETS))
        for metric, rows in scalar[stratum].items():
            if metric.startswith("m0_ordinal") or "ordinal_crps_delta" in metric:
                if len(rows) != ordinal_count:
                    raise ValueError(f"Invalid ordinal scalar panel for {stratum}/{metric}")
            elif len(rows) != full_count:
                raise ValueError(f"Incomplete scalar panel for {stratum}/{metric}")
    return direction, scalar, {
        "providers": providers,
        "profiles": len(profiles),
        "target_items": len(items_by_stratum["all_targets"]),
        "direction_rows_by_stratum": {name: len(rows) for name, rows in direction.items()},
        "scalar_rows_by_stratum_and_metric": {
            name: {metric: len(rows) for metric, rows in metrics.items()}
            for name, metrics in scalar.items()
        },
    }


def render_markdown(output: dict[str, Any]) -> str:
    def fmt(result: dict[str, Any]) -> str:
        return f'{result["estimate"]:.4f} [{result["ci_low"]:.4f}, {result["ci_high"]:.4f}]'

    lines = [
        "# Gate C Strict Comprehensive Metrics",
        "",
        "> Status: frozen measurement pilot, non-confirmatory. Gate C remains open.",
        "",
        "All estimates use survey weights normalized within true survey country, followed by equal-country, equal-item, and equal-provider aggregation. Confidence intervals are profile-cluster percentile bootstrap intervals. Directional intervals rebuild reference anchors on every replicate.",
        "",
        "The frozen pilot has 10 target items, not 20: four high-country-value items, four representative items, and two equivalence controls. Each profile was queried in two five-item API batches.",
        "",
    ]
    for set_name, label in (("official_four", "Core four-model sensitivity view"), ("five_systems", "All five models")):
        lines += [f"## {label}", "", "| Stratum | betaO | betaR | betaO-betaR | RegretO | RegretR | UtilityV | G0 | GR | Override |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        strata = output["analysis"]["provider_sets"][set_name]["strata"]
        for name in ("high_country_value", "representative", "equivalence_controls", "all_targets"):
            row = strata[name]
            d = row["direction"]
            p = row["prediction"]
            lines.append(
                f"| {name} | {fmt(d['beta_o']['macro'])} | {fmt(d['beta_r']['macro'])} | {fmt(d['beta_o_minus_beta_r']['macro'])} | "
                f"{fmt(p['regret_o']['macro'])} | {fmt(p['regret_r']['macro'])} | {fmt(p['utility_v']['macro'])} | "
                f"{fmt(p['g0']['macro'])} | {fmt(p['g_r']['macro'])} | {fmt(p['override']['macro'])} |"
            )
        lines += ["", "### High minus equivalence interaction", "", "| Metric | Estimate [95% CI] |", "|---|---:|"]
        interaction = output["analysis"]["provider_sets"][set_name]["interactions"]["high_minus_equivalence"]
        for metric, result in interaction["direction"].items():
            lines.append(f"| {metric} | {fmt(result['macro'])} |")
        for metric, result in interaction["prediction"].items():
            lines.append(f"| {metric} | {fmt(result['macro'])} |")
        lines.append("")
    lines += [
        "## Interpretation guardrails",
        "",
        "Model-level intervals and leave-one-model-out point estimates are secondary and are retained in the JSON output. GPT-5.4 is included in the all-model view. These pilot estimates do not close the pending robustness, data-processing-permission, or confirmatory-power gates.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "api_pilot/restricted_outputs/gate_c_v3_complete.jsonl")
    parser.add_argument("--profiles", type=Path, default=ROOT / "dataset/derived/gate_c_profiles_6x12.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/analysis/gate_c_manifest_v1.json")
    parser.add_argument("--gate-report", type=Path, default=ROOT / "dataset/analysis/pilot_gate_report.json")
    parser.add_argument("--panel", type=Path, default=ROOT / "dataset/derived/country_panel_6.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/metadata/variable_metadata.json")
    parser.add_argument("--prompt-bundle", type=Path, default=ROOT / "api_pilot/prompts/survey-confirmatory-v3.json")
    parser.add_argument("--output", type=Path, default=ROOT / "api_pilot/outputs/gate_c_v3_comprehensive_metrics.json")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "api_pilot/reports/diagnostics/GATE_C_COMPREHENSIVE_METRICS.md",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026072517)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    profiles = load_jsonl(args.profiles)
    results = successful_results(args.results)
    source_audit = audit_complete_results(results, profiles, args.prompt_bundle)
    strata = {
        "high_country_value": list(manifest["target_strata"]["high_country_value"]),
        "representative": list(manifest["target_strata"]["representative"]),
        "equivalence_controls": list(manifest["target_strata"]["equivalence_controls"]),
    }
    strata["all_targets"] = [item for name in ("high_country_value", "representative", "equivalence_controls") for item in strata[name]]
    if len(strata["all_targets"]) != 10 or len(set(strata["all_targets"])) != 10:
        raise ValueError("Frozen Gate C manifest must define exactly 10 unique targets")
    direction, scalar, row_audit = build_rows(results, profiles, strata)
    resampler, anchors, reference_counts = reference_rows_and_anchors(args.panel, args.metadata, strata["all_targets"])
    anchor_error = verify_anchor_match(anchors, args.gate_report, strata["all_targets"])
    providers = row_audit["providers"]
    if set(providers) != {*CORE_FOUR_MODELS, "openai"}:
        raise ValueError(f"Unexpected frozen provider roster: {providers}")
    analysis = comprehensive_bootstrap(
        direction,
        scalar,
        anchors,
        anchor_bootstrap=resampler,
        provider_sets={"official_four": CORE_FOUR_MODELS, "five_systems": tuple(providers)},
        expected_providers=providers,
        expected_countries=COUNTRIES,
        items_by_stratum=strata,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Gate C frozen measurement pilot; non-confirmatory aggregate-only analysis",
        "gate_status": "OPEN",
        "inputs": {name: sha256(path) for name, path in {
            "results": args.results, "profiles": args.profiles, "manifest": args.manifest,
            "gate_report": args.gate_report, "panel": args.panel, "metadata": args.metadata,
            "prompt_bundle": args.prompt_bundle,
        }.items()},
        "design": {
            "profiles": len(profiles),
            "countries": list(COUNTRIES),
            "target_items": 10,
            "api_batches_per_profile": 2,
            "weighting": "survey weights normalized in true country; equal country/item/provider macro",
            "metric_definitions": {
                "regret_o": "Brier(MO correct)-Brier(M0 correct)",
                "regret_r": "Brier(MR correct)-Brier(M0 correct)",
                "utility_v": "Brier(M0 correct)-Brier(MV correct)",
                "g0": "Brier(M0 shuffled)-Brier(M0 correct)",
                "g_r": "Brier(MR shuffled)-Brier(MR correct)",
                "override": "G0-GR",
            },
        },
        "source_audit": source_audit,
        "row_audit": row_audit,
        "reference": {"counts_by_country": reference_counts, "anchor_match_max_abs_error": anchor_error},
        "analysis": analysis,
        "model_note": "GPT-5.4 is included in the complete five-model view.",
        "interpretation_guardrail": "Pilot estimates do not close robustness, permission, provenance, or confirmatory gates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_markdown(output), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "bootstrap": args.bootstrap_replicates}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
