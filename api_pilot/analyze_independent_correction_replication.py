#!/usr/bin/env python3
"""Analyze the independent corrected-wording replication of H1 and H2."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api_pilot.analyze_checklist_correction_supplement import inference_dict  # noqa: E402
from api_pilot.run_confirmatory_strict_inference import audit_results, expected_call_map, latest_rows, write_json  # noqa: E402
from api_pilot.run_gate_c_strict_inference import load_jsonl, sha256  # noqa: E402
from api_pilot.survey_pilot_runner import (  # noqa: E402
    load_task_manifest,
    validate_v2_source_bindings,
)


def verify_provenance_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    bindings = {
        "task_manifest": "task_manifest_sha256",
        "analysis_manifest": "analysis_manifest_sha256",
        "prompt_bundle": "prompt_bundle_sha256",
        "restricted_profiles": "restricted_profiles_sha256",
        "selection_audit": "selection_audit_sha256",
        "invalidated_v1_record": "invalidated_v1_record_sha256",
    }
    for file_key, hash_key in bindings.items():
        candidate = ROOT / contract[file_key]
        if not candidate.is_file() or sha256(candidate) != contract[hash_key]:
            raise ValueError(f"Provenance binding failed: {file_key}")
    selection = json.loads((ROOT / contract["selection_audit"]).read_text(encoding="utf-8"))
    if (
        selection.get("status") != "verified_zero_source_record_overlap"
        or selection.get("original_source_record_count") != 504
        or selection.get("replication_source_record_count") != 72
        or selection.get("intersection_count") != 0
    ):
        raise ValueError("Source-record exclusion audit is not valid")
    invalidation = json.loads((ROOT / contract["invalidated_v1_record"]).read_text(encoding="utf-8"))
    if invalidation.get("status") != "invalidated_never_analyze":
        raise ValueError("Invalid replication v1 is not quarantined")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "api_pilot/restricted_outputs/independent_correction_replication_v2.jsonl")
    parser.add_argument("--task-manifest", type=Path, default=ROOT / "dataset/analysis/independent_correction_replication_manifest_v2.json")
    parser.add_argument("--analysis-manifest", type=Path, default=ROOT / "dataset/analysis/independent_correction_analysis_manifest_v2.json")
    parser.add_argument("--prompt-bundle", type=Path, default=ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json")
    parser.add_argument("--panel", type=Path, default=ROOT / "dataset/derived/country_panel_6.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/metadata/variable_metadata.json")
    parser.add_argument("--gate-report", type=Path, default=ROOT / "dataset/analysis/pilot_gate_report.json")
    parser.add_argument("--provenance-contract", type=Path, default=ROOT / "dataset/analysis/independent_correction_provenance_amendment_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "api_pilot/outputs/independent_correction_replication_v2_analysis.json")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "api_pilot/reports/final/INDEPENDENT_CORRECTION_REPLICATION_V2.md",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    args = parser.parse_args()

    task = load_task_manifest(args.task_manifest)
    if task is None:
        raise ValueError("Independent replication requires a frozen v2 task manifest")
    validate_v2_source_bindings(task, args.analysis_manifest)
    provenance = verify_provenance_contract(args.provenance_contract)
    profiles = load_jsonl(args.profiles)
    analysis = json.loads(args.analysis_manifest.read_text(encoding="utf-8"))
    source = task["source"]
    if sha256(args.profiles) != source["profile_file_sha256"]:
        raise ValueError("Restricted profile hash differs from the frozen task")
    if sha256(args.prompt_bundle) != task["prompt"]["sha256"]:
        raise ValueError("Prompt bundle hash differs from the frozen task")
    primary_items = analysis["target_strata"]["high_country_value"]
    filler_items = analysis["target_strata"]["filler"]
    if primary_items != task["analysis_contract"]["primary_items"]:
        raise ValueError("Primary-item analysis contract drift")
    if filler_items != task["analysis_contract"]["filler_items_excluded_from_inference"]:
        raise ValueError("Filler-item analysis contract drift")
    flattened = [item for batch in analysis["target_batches"].values() for item in batch]
    if set(flattened) != set(primary_items).union(filler_items) or len(flattened) != 10:
        raise ValueError("Frozen batch composition differs from the analysis contract")
    expected = expected_call_map(profiles, analysis, task, args.prompt_bundle)
    rows, audit = audit_results(latest_rows(args.results), expected)
    if not audit["complete"]:
        output = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "pending_api_completion", "source_audit": audit}
    else:
        items = primary_items
        result = inference_dict(rows, profiles, items, args.panel, args.metadata, args.gate_report, bootstrap=args.bootstrap_replicates, permutations=args.permutations, seed_offset=1700)
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "analysis_type": "post-review independent corrected-wording replication",
            "original_confirmatory_decisions_replaced": False,
            "source_audit": audit,
            "primary_items": items,
            "filler_items_excluded": analysis["target_strata"]["filler"],
            "result": result,
            "inputs": {name: sha256(path) for name, path in {"profiles": args.profiles, "results": args.results, "task_manifest": args.task_manifest, "analysis_manifest": args.analysis_manifest, "prompt_bundle": args.prompt_bundle, "provenance_contract": args.provenance_contract}.items()},
            "provenance_contract": provenance,
        }
    write_json(args.output, output)
    if output["status"] == "complete":
        inf = output["result"]["inference"]
        d = inf["direction_bootstrap"]
        u = inf["utility_h2"]
        holm = inf["holm_adjusted_p"]
        raw = inf["raw_hypothesis_p"]
        assignment = inf["assignment_permutation"]
        h1_supported = bool(holm["H1"] < 0.05 and d["beta_o"]["estimate"] > 0 and d["beta_diff"]["estimate"] > 0)
        h2_supported = bool(holm["H2"] < 0.05 and u["estimate"] > 0)
        lines = [
            "# Independent Corrected-wording Replication",
            "",
            "This post-review, sample-disjoint corrective replication uses 72 source records that were not among the original 504 confirmatory records. It is not an independent dataset or research-team replication and does not replace the prospectively frozen confirmation.",
            "",
            "| Hypothesis/component | Estimate | 95% CI | Raw p | Holm p | Decision |",
            "|---|---:|---:|---:|---:|---|",
            f"| H1a: betaO > 0 | {d['beta_o']['estimate']:.4f} | [{d['beta_o']['ci_low']:.4f}, {d['beta_o']['ci_high']:.4f}] | {d['beta_o']['p_greater']:.6g} |  | component |",
            f"| H1b: betaO-betaR > 0 | {d['beta_diff']['estimate']:.4f} | [{d['beta_diff']['ci_low']:.4f}, {d['beta_diff']['ci_high']:.4f}] | {d['beta_diff']['p_greater']:.6g} |  | component |",
            f"| H1 IUT |  |  | {raw['H1']:.6g} | {holm['H1']:.6g} | {'supported' if h1_supported else 'not supported'} |",
            f"| H2: UtilityV > 0 | {u['estimate']:.4f} | [{u['ci_low']:.4f}, {u['ci_high']:.4f}] | {raw['H2']:.6g} | {holm['H2']:.6g} | {'supported' if h2_supported else 'not supported'} |",
            "",
            f"Assignment-permutation H1 IUT p: {assignment['h1_iut_p']:.6g}. SESOI exceedance uses betaO-betaR = 0.05 and UtilityV = 0.02 and is reported separately in the JSON output.",
        ]
    else:
        lines = ["# Independent Corrected-wording Replication", "", "Status: pending API completion."]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "planned": audit["planned"], "observed": audit["observed_unique"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
