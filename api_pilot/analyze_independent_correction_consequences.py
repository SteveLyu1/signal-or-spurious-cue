#!/usr/bin/env python3
"""Prediction consequences and item robustness for the corrected replication.

This is a post-review descriptive secondary analysis. It never changes the
frozen H1/H2 decisions and never calls an external model API.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot.analyze_independent_correction_replication import (  # noqa: E402
    verify_provenance_contract,
)
from api_pilot.analyze_survey_pilot import (  # noqa: E402
    argmax_accuracy,
    brier,
    total_variation,
)
from api_pilot.run_confirmatory_strict_inference import (  # noqa: E402
    audit_results,
    expected_call_map,
    latest_rows,
)
from api_pilot.run_gate_c_strict_inference import (  # noqa: E402
    COUNTRIES,
    build_inference_rows,
    item_result_index,
    load_jsonl,
    reference_rows_and_anchors,
    sha256,
)
from api_pilot.strict_inference import (  # noqa: E402
    direction_profile_bootstrap,
    paired_profile_bootstrap,
)
from api_pilot.survey_pilot_runner import (  # noqa: E402
    load_task_manifest,
    validate_v2_source_bindings,
)


RESULTS = ROOT / "api_pilot/restricted_outputs/independent_correction_replication_v2.jsonl"
PROFILES = ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl"
TASK = ROOT / "dataset/analysis/independent_correction_replication_manifest_v2.json"
ANALYSIS = ROOT / "dataset/analysis/independent_correction_analysis_manifest_v2.json"
PROMPT = ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json"
PROVENANCE = ROOT / "dataset/analysis/independent_correction_provenance_amendment_v1.json"
PANEL = ROOT / "dataset/derived/country_panel_6.csv"
METADATA = ROOT / "dataset/metadata/variable_metadata.json"
OUT_JSON = ROOT / "api_pilot/outputs/independent_correction_consequences.json"
OUT_CSV = ROOT / "api_pilot/outputs/final_tables/table_8_independent_correction_consequences.csv"


def scalar_rows(
    results: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    items: list[str],
) -> dict[str, list[dict[str, Any]]]:
    index = item_result_index(results)
    providers = sorted({row["provider"] for row in results})
    output = {
        metric: []
        for metric in (
            "regret_o",
            "regret_r",
            "regret_attenuation_o_minus_r",
            "utility_v",
            "mo_accuracy_delta",
            "mr_accuracy_delta",
            "mv_accuracy_delta",
            "mo_tv",
            "mr_tv",
            "mv_tv",
            "mo_mr_tv",
        )
    }
    for provider in providers:
        for profile in profiles:
            profile_id = profile["profile_id"]
            targets = {target["item_id"]: target for target in profile["targets"]}
            for item in items:
                key = lambda condition: (provider, profile_id, condition, "correct", item)
                source = {condition: index[key(condition)] for condition in ("M0", "MO", "MR", "MV")}
                codes = source["M0"]["target_option_codes"][item]
                truth = codes.index(targets[item]["answer_code"])
                forecasts = {
                    condition: np.asarray(row["parsed_forecasts"][item], dtype=float)
                    for condition, row in source.items()
                }
                losses = {
                    condition: brier(values.tolist(), truth)
                    for condition, values in forecasts.items()
                }
                common = {
                    "provider": provider,
                    "profile_id": profile_id,
                    "country": profile["true_country_code"],
                    "item_id": item,
                    "weight": float(profile["analysis_weight"]),
                }
                values = {
                    "regret_o": losses["MO"] - losses["M0"],
                    "regret_r": losses["MR"] - losses["M0"],
                    "regret_attenuation_o_minus_r": losses["MO"] - losses["MR"],
                    "utility_v": losses["M0"] - losses["MV"],
                    "mo_accuracy_delta": argmax_accuracy(forecasts["MO"].tolist(), truth)
                    - argmax_accuracy(forecasts["M0"].tolist(), truth),
                    "mr_accuracy_delta": argmax_accuracy(forecasts["MR"].tolist(), truth)
                    - argmax_accuracy(forecasts["M0"].tolist(), truth),
                    "mv_accuracy_delta": argmax_accuracy(forecasts["MV"].tolist(), truth)
                    - argmax_accuracy(forecasts["M0"].tolist(), truth),
                    "mo_tv": total_variation(forecasts["M0"].tolist(), forecasts["MO"].tolist()),
                    "mr_tv": total_variation(forecasts["M0"].tolist(), forecasts["MR"].tolist()),
                    "mv_tv": total_variation(forecasts["M0"].tolist(), forecasts["MV"].tolist()),
                    "mo_mr_tv": total_variation(forecasts["MO"].tolist(), forecasts["MR"].tolist()),
                }
                for metric, value in values.items():
                    output[metric].append({**common, "value": float(value)})
    return output


def bootstrap_scalar(
    rows: list[dict[str, Any]],
    *,
    providers: list[str],
    items: list[str],
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    return paired_profile_bootstrap(
        rows,
        n_bootstrap=n_bootstrap,
        seed=seed,
        expected_providers=providers,
        expected_countries=COUNTRIES,
        expected_items=items,
    ).__dict__


def main() -> int:
    task = load_task_manifest(TASK)
    if task is None:
        raise ValueError("Frozen correction task manifest is required")
    validate_v2_source_bindings(task, ANALYSIS)
    verify_provenance_contract(PROVENANCE)
    if sha256(PROFILES) != task["source"]["profile_file_sha256"]:
        raise ValueError("Profile hash differs from the frozen task")
    if sha256(PROMPT) != task["prompt"]["sha256"]:
        raise ValueError("Prompt hash differs from the frozen task")

    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    items = analysis["target_strata"]["high_country_value"]
    profiles = load_jsonl(PROFILES)
    expected = expected_call_map(profiles, analysis, task, PROMPT)
    results, audit = audit_results(latest_rows(RESULTS), expected)
    if not audit["complete"]:
        raise ValueError("Correction replication is incomplete")
    providers = sorted({row["provider"] for row in results})

    direction, utility, row_audit = build_inference_rows(
        results, profiles, items, require_balanced_assignment=True
    )
    resampler, anchors, reference_counts = reference_rows_and_anchors(PANEL, METADATA, items)
    scalar = scalar_rows(results, profiles, items)
    n_bootstrap = 10_000

    overall: dict[str, Any] = {}
    for index, (metric, rows) in enumerate(scalar.items()):
        overall[metric] = bootstrap_scalar(
            rows,
            providers=providers,
            items=items,
            seed=2026073000 + index,
            n_bootstrap=n_bootstrap,
        )

    by_item: dict[str, Any] = {}
    for item_index, item in enumerate(items):
        item_resampler, item_anchors, _ = reference_rows_and_anchors(
            PANEL, METADATA, [item]
        )
        item_direction = [row for row in direction if row["item_id"] == item]
        d = direction_profile_bootstrap(
            item_direction,
            item_anchors,
            anchor_bootstrap=item_resampler,
            require_anchor_reestimation=True,
            n_bootstrap=2_000,
            seed=2026073100 + item_index,
            expected_providers=providers,
            expected_countries=COUNTRIES,
            expected_items=[item],
        )
        metrics: dict[str, Any] = {
            "beta_o": d.beta_o.__dict__,
            "beta_r": d.beta_r.__dict__,
            "beta_o_minus_beta_r": d.beta_diff.__dict__,
        }
        for metric_index, metric in enumerate(("regret_o", "regret_r", "regret_attenuation_o_minus_r", "utility_v")):
            item_rows = [row for row in scalar[metric] if row["item_id"] == item]
            metrics[metric] = bootstrap_scalar(
                item_rows,
                providers=providers,
                items=[item],
                seed=2026073200 + 20 * item_index + metric_index,
                n_bootstrap=2_000,
            )
        by_item[item] = metrics

    leave_one_item_out: dict[str, Any] = {}
    for item_index, omitted in enumerate(items):
        kept = [item for item in items if item != omitted]
        kept_resampler, kept_anchors, _ = reference_rows_and_anchors(
            PANEL, METADATA, kept
        )
        kept_direction = [row for row in direction if row["item_id"] in kept]
        d = direction_profile_bootstrap(
            kept_direction,
            kept_anchors,
            anchor_bootstrap=kept_resampler,
            require_anchor_reestimation=True,
            n_bootstrap=2_000,
            seed=2026074000 + item_index,
            expected_providers=providers,
            expected_countries=COUNTRIES,
            expected_items=kept,
        )
        u_rows = [row for row in utility if row["item_id"] in kept]
        u = paired_profile_bootstrap(
            u_rows,
            n_bootstrap=2_000,
            seed=2026074100 + item_index,
            expected_providers=providers,
            expected_countries=COUNTRIES,
            expected_items=kept,
        )
        leave_one_item_out[omitted] = {
            "kept_items": kept,
            "beta_o": d.beta_o.__dict__,
            "beta_r": d.beta_r.__dict__,
            "beta_o_minus_beta_r": d.beta_diff.__dict__,
            "utility_v": u.__dict__,
        }

    rows_out: list[dict[str, Any]] = []
    for metric, value in overall.items():
        rows_out.append({
            "scope": "overall",
            "unit": "all_seven_items",
            "metric": metric,
            "estimate": value["estimate"],
            "ci_low": value["ci_low"],
            "ci_high": value["ci_high"],
            "analysis_role": "post_review_descriptive_secondary",
        })
    for item, values in by_item.items():
        for metric, value in values.items():
            rows_out.append({
                "scope": "item",
                "unit": item,
                "metric": metric,
                "estimate": value["estimate"],
                "ci_low": value["ci_low"],
                "ci_high": value["ci_high"],
                "analysis_role": "post_review_exploratory_no_multiplicity_claim",
            })
    for omitted, values in leave_one_item_out.items():
        for metric in ("beta_o", "beta_r", "beta_o_minus_beta_r", "utility_v"):
            value = values[metric]
            rows_out.append({
                "scope": "leave_one_item_out",
                "unit": omitted,
                "metric": metric,
                "estimate": value["estimate"],
                "ci_low": value["ci_low"],
                "ci_high": value["ci_high"],
                "analysis_role": "post_review_robustness_secondary",
            })

    payload = {
        "status": "complete",
        "analysis_role": "post-review descriptive consequences and item robustness; no H1/H2 decisions changed",
        "api_calls_made": 0,
        "source_audit": audit,
        "row_audit": row_audit,
        "providers": providers,
        "items": items,
        "reference_counts": reference_counts,
        "bootstrap_replicates_overall": n_bootstrap,
        "bootstrap_replicates_item_and_loo": 2_000,
        "overall": overall,
        "by_item": by_item,
        "leave_one_item_out": leave_one_item_out,
        "interpretation_guardrail": (
            "Item estimates are exploratory and unadjusted for multiplicity. "
            "Leave-one-item-out estimates assess concentration, not generalisation "
            "to a new item population."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(json.dumps({"status": "complete", "rows": len(rows_out), "api_calls_made": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
