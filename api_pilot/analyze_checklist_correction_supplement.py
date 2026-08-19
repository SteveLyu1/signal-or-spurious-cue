#!/usr/bin/env python3
"""Analyze the post-review checklist conversion correction supplement."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot.run_confirmatory_strict_inference import (  # noqa: E402
    OFFICIAL_MODELS,
    audit_results,
    expected_call_map,
    latest_rows,
    write_json,
)
from api_pilot.run_gate_c_strict_inference import (  # noqa: E402
    COUNTRIES,
    build_inference_rows,
    load_jsonl,
    reference_rows_and_anchors,
    sha256,
    verify_anchor_match,
)
from api_pilot.strict_inference import run_strict_confirmatory_inference  # noqa: E402
from api_pilot.survey_pilot_runner import load_task_manifest  # noqa: E402


AFFECTED = ("A029", "A030", "A039", "A042", "A124_03", "A124_06", "A124_08", "A124_09")
NEIGHBOUR = ("A124_03", "A124_06", "A124_08", "A124_09")
CHILD = ("A029", "A030", "A039", "A042")


def inference_dict(
    rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    items: list[str],
    panel: Path,
    metadata: Path,
    gate_report: Path,
    *,
    bootstrap: int,
    permutations: int,
    seed_offset: int,
) -> dict[str, Any]:
    direction, utility, audit = build_inference_rows(
        rows, profiles, items, require_balanced_assignment=True
    )
    resampler, anchors, reference_counts = reference_rows_and_anchors(
        panel, metadata, items
    )
    anchor_error = verify_anchor_match(anchors, gate_report, items)
    inference = run_strict_confirmatory_inference(
        direction,
        utility,
        anchors,
        anchor_bootstrap=resampler,
        n_bootstrap=bootstrap,
        n_permutations=permutations,
        bootstrap_seed=2026072701 + seed_offset,
        permutation_seed=2026072702 + seed_offset,
        expected_providers=tuple(sorted(OFFICIAL_MODELS)),
        expected_countries=COUNTRIES,
        expected_items=items,
    )
    return {
        "items": items,
        "row_audit": audit,
        "reference_counts": reference_counts,
        "anchor_max_abs_error": anchor_error,
        "inference": inference.to_dict(),
    }


def summary_row(value: dict[str, Any]) -> dict[str, float]:
    result = value["inference"]
    direction = result["direction_bootstrap"]
    return {
        "beta_o": direction["beta_o"]["estimate"],
        "beta_r": direction["beta_r"]["estimate"],
        "beta_o_minus_beta_r": direction["beta_diff"]["estimate"],
        "utility_v": result["utility_h2"]["estimate"],
    }


def render(output: dict[str, Any]) -> str:
    lines = [
        "# Checklist Conversion Correction Supplement",
        "",
        f"> Status: {output['status']}",
        "",
    ]
    if output["status"] != "complete":
        audit = output["supplement_source_audit"]
        lines += [
            output["reason"],
            "",
            "| Planned | Observed | Missing | Incomplete |",
            "|---:|---:|---:|---:|",
            f"| {audit['planned']} | {audit['observed_unique']} | {audit['missing']} | {audit['incomplete']} |",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "Blind review identified eight checklist indicators whose displayed question and response labels did not preserve the source response process. The original output remains unchanged. This report gives a no-new-API exclusion sensitivity and a post-review corrected-wording supplement.",
        "",
        "## Corrected items",
        "",
        "| Scope | Items |",
        "|---|---:|",
        f"| All affected checklist indicators | {len(AFFECTED)} |",
        f"| Neighbour checklist | {len(NEIGHBOUR)} |",
        f"| Child-quality checklist | {len(CHILD)} |",
        "",
        "## Exact-subset wording comparison",
        "",
        "The same 72 records, metadata assignments, five models, four five-item batches and eight affected items are used on both sides. The prompt content differs only in the corrected item wording and displayed binary labels. The calls occurred at different times, so the before/after difference also contains backend sampling and temporal variation and is not a pure causal wording effect.",
        "",
        "| Scope | Version | betaO | betaR | betaO-betaR | UtilityV |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope in ("all_affected", "neighbour", "child_quality"):
        for version in ("original_wording", "corrected_wording"):
            row = summary_row(output["exact_subset_comparison"][scope][version])
            lines.append(
                f"| {scope} | {version} | {row['beta_o']:.4f} | {row['beta_r']:.4f} | "
                f"{row['beta_o_minus_beta_r']:.4f} | {row['utility_v']:.4f} |"
            )
    exclusion = output["exclusion_sensitivity"]["remaining_high_country_value"]
    row = summary_row(exclusion)
    lines += [
        "",
        "## Exclusion sensitivity",
        "",
        "All eight affected items are removed from the original 504-record analysis. Only two originally high-country-value items remain, so this is a sensitivity check rather than a replacement confirmatory test.",
        "",
        "| Remaining high-value items | betaO | betaR | betaO-betaR | UtilityV |",
        "|---|---:|---:|---:|---:|",
        f"| {', '.join(exclusion['items'])} | {row['beta_o']:.4f} | {row['beta_r']:.4f} | "
        f"{row['beta_o_minus_beta_r']:.4f} | {row['utility_v']:.4f} |",
        "",
        "## Interpretation",
        "",
        "These estimates are corrective supplementary evidence generated after confirmatory outcomes and blind-review feedback were known. They show sensitivity to the conversion correction but do not identify a pure wording effect and do not replace the preregistered H1/H2 decisions.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "api_pilot/restricted_outputs/checklist_correction_supplement_v1.jsonl")
    parser.add_argument("--profiles", type=Path, default=ROOT / "api_pilot/restricted_outputs/checklist_correction_profiles_6x12.jsonl")
    parser.add_argument("--task-manifest", type=Path, default=ROOT / "dataset/analysis/checklist_correction_supplement_manifest_v1.json")
    parser.add_argument("--analysis-manifest", type=Path, default=ROOT / "dataset/analysis/confirmatory_manifest_v2.json")
    parser.add_argument("--prompt-bundle", type=Path, default=ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json")
    parser.add_argument("--original-logical", type=Path, default=ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_logical_five_model.jsonl")
    parser.add_argument("--full-profiles", type=Path, default=ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl")
    parser.add_argument("--panel", type=Path, default=ROOT / "dataset/derived/country_panel_6.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/metadata/variable_metadata.json")
    parser.add_argument("--gate-report", type=Path, default=ROOT / "dataset/analysis/pilot_gate_report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "api_pilot/outputs/checklist_correction_analysis.json")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "api_pilot/reports/final/CHECKLIST_CORRECTION_SUPPLEMENT.md",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = load_task_manifest(args.task_manifest)
    if task is None:
        raise ValueError("Correction supplement requires a frozen task manifest")
    profiles = load_jsonl(args.profiles)
    if len(profiles) != 72 or sha256(args.profiles) != task["source"]["profile_file_sha256"]:
        raise ValueError("Correction profiles differ from the frozen task manifest")
    expected = expected_call_map(
        profiles,
        json.loads(args.analysis_manifest.read_text(encoding="utf-8")),
        task,
        args.prompt_bundle,
    )
    supplement_rows, source_audit = audit_results(latest_rows(args.results), expected)
    if not source_audit["complete"]:
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_api_completion",
            "reason": "The frozen corrected-wording API panel is incomplete. No correction estimate has been generated.",
            "supplement_source_audit": source_audit,
        }
        write_json(args.output, output)
        args.report.write_text(render(output), encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False))
        return 0

    original_rows = load_jsonl(args.original_logical)
    selected_ids = {profile["profile_id"] for profile in profiles}
    original_subset = [row for row in original_rows if row["profile_id"] in selected_ids]
    full_profiles = load_jsonl(args.full_profiles)
    high_items = json.loads(args.analysis_manifest.read_text(encoding="utf-8"))["target_strata"]["high_country_value"]
    remaining_high = [item for item in high_items if item not in AFFECTED]

    scopes = {
        "all_affected": list(AFFECTED),
        "neighbour": list(NEIGHBOUR),
        "child_quality": list(CHILD),
    }
    comparison: dict[str, Any] = {}
    for index, (scope, items) in enumerate(scopes.items()):
        comparison[scope] = {
            "original_wording": inference_dict(
                original_subset, profiles, items, args.panel, args.metadata,
                args.gate_report, bootstrap=args.bootstrap_replicates,
                permutations=args.permutations, seed_offset=100 * index,
            ),
            "corrected_wording": inference_dict(
                supplement_rows, profiles, items, args.panel, args.metadata,
                args.gate_report, bootstrap=args.bootstrap_replicates,
                permutations=args.permutations, seed_offset=100 * index + 10,
            ),
        }

    exclusion = inference_dict(
        original_rows,
        full_profiles,
        remaining_high,
        args.panel,
        args.metadata,
        args.gate_report,
        bootstrap=args.bootstrap_replicates,
        permutations=args.permutations,
        seed_offset=900,
    )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "analysis_type": "post-review corrective supplement",
        "confirmatory_decisions_replaced": False,
        "supplement_source_audit": source_audit,
        "affected_items": list(AFFECTED),
        "exact_subset_comparison": comparison,
        "exclusion_sensitivity": {"remaining_high_country_value": exclusion},
        "replicates": {
            "bootstrap": args.bootstrap_replicates,
            "permutations": args.permutations,
        },
        "inputs": {
            name: sha256(path)
            for name, path in {
                "results": args.results,
                "profiles": args.profiles,
                "task_manifest": args.task_manifest,
                "analysis_manifest": args.analysis_manifest,
                "prompt_bundle": args.prompt_bundle,
                "original_logical": args.original_logical,
                "full_profiles": args.full_profiles,
            }.items()
        },
    }
    write_json(args.output, output)
    args.report.write_text(render(output), encoding="utf-8")
    print(json.dumps({"status": "complete", "affected_items": len(AFFECTED)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
