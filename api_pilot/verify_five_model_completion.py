#!/usr/bin/env python3
"""Verify the amended five-model experiment and write its completion contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "api_pilot/outputs/five_model_panel_sync.json"
OUTPUT = ROOT / "api_pilot/outputs/experiment_completion_contract.json"
REVIEW_PACKET = ROOT / "api_pilot/human_review/BLIND_QUALITY_REVIEW.md"
REVIEW_CSV = ROOT / "api_pilot/human_review/blind_quality_review_template.csv"
CORRECTION_REVIEW_CSV = ROOT / "api_pilot/human_review/checklist_correction_review.csv"
CORRECTION_MANIFEST = "dataset/analysis/checklist_correction_supplement_manifest_v1.json"
CORRECTION_ANALYSIS = "api_pilot/outputs/checklist_correction_analysis.json"
CORRECTION_REVIEW = "api_pilot/human_review/checklist_correction_review.csv"
INDEPENDENT_MANIFEST = "dataset/analysis/independent_correction_replication_manifest_v2.json"
INDEPENDENT_ANALYSIS = "api_pilot/outputs/independent_correction_replication_v2_analysis.json"
INDEPENDENT_COMPLETION = "api_pilot/outputs/independent_correction_replication_v2_completion.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_jsonl(path: Path, expected: int) -> dict[str, Any]:
    if not path.exists():
        return {"passes": False, "observed": 0, "expected": expected}
    call_ids: set[str] = set()
    observed = 0
    unsuccessful = 0
    duplicates = 0
    malformed = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        observed += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            malformed += 1
        elif call_id in call_ids:
            duplicates += 1
        else:
            call_ids.add(call_id)
        if not row.get("success") or row.get("terminal_failure"):
            unsuccessful += 1
    passes = (
        observed == expected
        and len(call_ids) == expected
        and not unsuccessful
        and not duplicates
        and not malformed
    )
    return {
        "passes": passes,
        "observed": observed,
        "observed_unique": len(call_ids),
        "expected": expected,
        "unsuccessful": unsuccessful,
        "duplicates": duplicates,
        "malformed": malformed,
        "sha256": sha256(path),
    }


def report_internal_audit_passes(name: str, value: dict[str, Any]) -> bool:
    if name == "cache":
        return bool(
            (value.get("source_audit") or {}).get("complete")
            and (value.get("cache_audit") or {}).get("complete")
        )
    if name == "strict":
        return bool((value.get("source_audit") or {}).get("complete"))
    if name == "comprehensive":
        return all(
            (value.get(key) or {}).get("complete")
            for key in ("source_audit", "cache_audit", "item_panel_audit")
        )
    if name == "sentinel":
        waves = value.get("wave_audits") or {}
        return bool(
            value.get("complete") is True
            and (value.get("core_audit") or {}).get("complete")
            and waves
            and all((audit or {}).get("complete") for audit in waves.values())
        )
    if name == "batch_context":
        return bool(
            value.get("complete") is True
            and (value.get("audit") or {}).get("complete")
        )
    if name == "robustness":
        return value.get("complete") is True
    return False


def correction_supplement_check(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless the post-review correction is complete and hash-bound."""
    manifest_path = root / CORRECTION_MANIFEST
    analysis_path = root / CORRECTION_ANALYSIS
    if not manifest_path.is_file() or not analysis_path.is_file():
        return {"passes": False, "reason": "correction manifest or analysis is absent"}, {}
    try:
        manifest = load(manifest_path)
        analysis = load(analysis_path)
    except (OSError, json.JSONDecodeError) as error:
        return {"passes": False, "reason": f"invalid correction artifact: {error}"}, {}
    expected = int((manifest.get("expected") or {}).get("total_calls") or 0)
    output_value = str(manifest.get("immutable_output_path") or "")
    output_path = root / output_value if output_value else Path()
    output_audit = audit_jsonl(output_path, expected) if output_value else {
        "passes": False, "expected": expected, "observed": 0
    }
    source = manifest.get("source") or {}
    prompt = manifest.get("prompt") or {}
    paths = {
        "profiles": root / str(source.get("profile_file") or ""),
        "prompt_bundle": root / str(prompt.get("file") or ""),
    }
    inputs = analysis.get("inputs") or {}
    hashes_match = bool(
        output_audit.get("passes")
        and inputs.get("results") == output_audit.get("sha256")
        and inputs.get("task_manifest") == sha256(manifest_path)
        and paths["profiles"].is_file()
        and sha256(paths["profiles"]) == source.get("profile_file_sha256")
        and inputs.get("profiles") == source.get("profile_file_sha256")
        and paths["prompt_bundle"].is_file()
        and sha256(paths["prompt_bundle"]) == prompt.get("sha256")
        and inputs.get("prompt_bundle") == prompt.get("sha256")
    )
    source_audit = analysis.get("supplement_source_audit") or {}
    analysis_complete = bool(
        analysis.get("status") == "complete"
        and analysis.get("analysis_type") == "post-review corrective supplement"
        and analysis.get("confirmatory_decisions_replaced") is False
        and source_audit.get("complete") is True
        and source_audit.get("planned") == expected
        and source_audit.get("observed_unique") == expected
        and source_audit.get("missing") == 0
        and source_audit.get("incomplete") == 0
    )
    return {
        "passes": bool(expected == 5_760 and hashes_match and analysis_complete),
        "expected": expected,
        "output_audit": output_audit,
        "hashes_match": hashes_match,
        "analysis_complete": analysis_complete,
        "manifest_sha256": sha256(manifest_path),
        "analysis_sha256": sha256(analysis_path),
    }, analysis


def independent_replication_check(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / INDEPENDENT_MANIFEST
    analysis_path = root / INDEPENDENT_ANALYSIS
    completion_path = root / INDEPENDENT_COMPLETION
    if not all(path.is_file() for path in (manifest_path, analysis_path, completion_path)):
        return {"passes": False, "expected": 2_880, "reason": "independent replication artifacts are pending"}, {}
    manifest = load(manifest_path)
    analysis = load(analysis_path)
    completion = load(completion_path)
    expected = int(manifest["expected"]["total_calls"])
    expected_each = int(manifest["expected"]["calls_per_provider"])
    provider_expected = {
        provider: expected_each for provider in sorted(manifest["provider_models"])
    }
    passes = bool(
        expected == 2_880
        and completion.get("complete") is True
        and completion.get("status") == "complete"
        and completion.get("expected") == expected
        and completion.get("observed_unique") == expected
        and completion.get("success_by_provider") == provider_expected
        and completion.get("result_complete") is True
        and completion.get("analysis_complete") is True
        and completion.get("task_manifest_sha256") == sha256(manifest_path)
        and completion.get("analysis_sha256") == sha256(analysis_path)
        and analysis.get("status") == "complete"
        and analysis.get("original_confirmatory_decisions_replaced") is False
    )
    return {
        "passes": passes,
        "expected": expected,
        "completion_sha256": sha256(completion_path),
        "manifest_sha256": sha256(manifest_path),
        "analysis_sha256": sha256(analysis_path),
        "success_by_provider": completion.get("success_by_provider") or {},
    }, analysis


def main() -> int:
    sync = load(SYNC)
    pilot_status_path = ROOT / "api_pilot/outputs/gate_c_v3_postprocess_status.json"
    pilot_status = load(pilot_status_path) if pilot_status_path.exists() else {}
    pilot_calls = int((pilot_status.get("audit") or {}).get("planned") or 0)
    checks = []
    for name, panel in sync["panels"].items():
        path = ROOT / panel["output"]
        audit = audit_jsonl(path, int(panel["expected"]))
        audit["id"] = f"panel_{name}"
        audit["sync_declared_complete"] = panel["complete"]
        audit["passes"] = bool(audit["passes"] and panel["complete"])
        checks.append(audit)
    reports = {
        "cache": ROOT / "api_pilot/outputs/confirmatory_core_v2_cache_audit.json",
        "strict": ROOT / "api_pilot/outputs/confirmatory_core_v2_strict_inference.json",
        "comprehensive": ROOT / "api_pilot/outputs/confirmatory_core_v2_comprehensive_metrics.json",
        "sentinel": ROOT / "api_pilot/outputs/confirmatory_sentinel_analysis.json",
        "batch_context": ROOT / "api_pilot/outputs/confirmatory_batch_context_analysis.json",
        "robustness": ROOT / "api_pilot/outputs/five_model_robustness_gate.json",
    }
    required = {
        "cache": {"cached_logical_panel_complete"},
        "strict": {"strict_confirmatory_inference_complete"},
        "comprehensive": {"confirmatory_secondary_metrics_complete"},
        "sentinel": {"complete"},
        "batch_context": {"complete"},
        # Robustness gates are report-only. Both pass and fail are complete
        # scientific outcomes; only a pending analysis keeps the contract open.
        "robustness": {"pass", "fail"},
    }
    documents = {}
    for name, path in reports.items():
        value = load(path) if path.exists() else {}
        documents[name] = value
        checks.append({
            "id": f"report_{name}",
            "passes": value.get("status") in required[name]
            and report_internal_audit_passes(name, value),
            "status": value.get("status"),
            "sha256": sha256(path) if path.exists() else None,
        })
    correction_check, correction_analysis = correction_supplement_check(ROOT)
    checks.append({"id": "checklist_correction_supplement", **correction_check})
    documents["checklist_correction"] = correction_analysis
    independent_check, independent_analysis = independent_replication_check(ROOT)
    checks.append({"id": "independent_corrected_wording_replication", **independent_check})
    documents["independent_corrected_wording_replication"] = independent_analysis
    cache = documents.get("cache") or {}
    logical_output = cache.get("logical_output") or {}
    logical_path_value = logical_output.get("path")
    logical_path = ROOT / logical_path_value if logical_path_value else None
    expected_logical = int((cache.get("cache_audit") or {}).get("logical_rows") or 0)
    logical_observed = (
        sum(
            1 for line in logical_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        if logical_path and logical_path.exists()
        else 0
    )
    logical_hash = sha256(logical_path) if logical_path and logical_path.exists() else None
    checks.append({
        "id": "logical_cached_panel",
        "passes": bool(
            expected_logical > 0
            and logical_observed == expected_logical
            and logical_hash == logical_output.get("sha256")
        ),
        "observed": logical_observed,
        "expected": expected_logical,
        "sha256": logical_hash,
    })
    machine_complete = all(row["passes"] for row in checks)
    signoff = ROOT / "api_pilot/human_review/human_quality_review_signoff.json"
    human = load(signoff) if signoff.exists() else {}
    review_packet_hash = sha256(REVIEW_PACKET) if REVIEW_PACKET.exists() else None
    review_csv_hash = sha256(REVIEW_CSV) if REVIEW_CSV.exists() else None
    correction_review_csv_hash = (
        sha256(CORRECTION_REVIEW_CSV) if CORRECTION_REVIEW_CSV.exists() else None
    )
    human_checks = {
        "approved_status": human.get("status") == "approved",
        "reviewer_recorded": bool(str(human.get("reviewer") or "").strip()),
        "review_time_recorded": bool(str(human.get("reviewed_at") or "").strip()),
        "packet_hash_matches": bool(
            review_packet_hash
            and human.get("review_packet_sha256") == review_packet_hash
        ),
        "original_review_hash_matches": bool(
            review_csv_hash
            and human.get("original_review_csv_sha256") == review_csv_hash
        ),
        "correction_review_hash_matches": bool(
            correction_review_csv_hash
            and human.get("correction_review_csv_sha256") == correction_review_csv_hash
        ),
        "correction_rows_reviewed": human.get("correction_reviewed_rows") == 8,
        "original_rows_reviewed": human.get("reviewed_rows") == 34,
        "no_unresolved_rejections": human.get("unresolved_rejections") == [],
    }
    human_complete = all(human_checks.values())
    result = {
        "schema_version": "five-model-experiment-completion-v1",
        "status": "complete" if machine_complete and human_complete else ("awaiting_human_review" if machine_complete else "incomplete"),
        "machine_complete": machine_complete,
        "model_roster": ["gpt-5.4", "deepseek-v4-pro", "gemini-3.6-flash", "mistral-medium-3-5", "qwen3.7-max-2026-06-08"],
        "planned_api_calls": pilot_calls + sum(
            int(panel["expected"]) for panel in sync["panels"].values()
        ) + int(correction_check.get("expected") or 0) + int(independent_check.get("expected") or 0),
        "checks": checks,
        "failures": [row for row in checks if not row["passes"]],
        "documents": documents,
        "human_review": {
            "required": True,
            "passes": human_complete,
            "path": signoff.relative_to(ROOT).as_posix(),
            "review_packet": REVIEW_PACKET.relative_to(ROOT).as_posix(),
            "review_packet_sha256": review_packet_hash,
            "original_review_csv_sha256": review_csv_hash,
            "correction_review_csv_sha256": correction_review_csv_hash,
            "checks": human_checks,
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "machine_complete": machine_complete}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
