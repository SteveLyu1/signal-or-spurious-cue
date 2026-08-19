#!/usr/bin/env python3
"""Fail-closed completion contract for the frozen confirmatory experiment.

This module is offline-only.  It never imports a provider client or sends an
HTTP request.  It verifies the immutable manifests, result panels and analysis
reports, then writes a compact machine-readable contract and hand-off files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CORE_MANIFEST = "dataset/analysis/confirmatory_core_task_manifest_v2.json"
ROBUSTNESS_SUITE = "dataset/analysis/gate_c_robustness_suite_v2.json"
SENTINEL_SUITE = "dataset/analysis/confirmatory_sentinel_suite_v1.json"
CACHE_REPORT = "api_pilot/outputs/confirmatory_core_v2_cache_audit.json"
STRICT_REPORT = "api_pilot/outputs/confirmatory_core_v2_strict_inference.json"
COMPREHENSIVE_REPORT = (
    "api_pilot/outputs/confirmatory_core_v2_comprehensive_metrics.json"
)
SENTINEL_REPORT = "api_pilot/outputs/confirmatory_sentinel_analysis.json"
RUNTIME_QUALITY_REPORT = "api_pilot/outputs/formal_runtime_quality.json"
CONTRACT_OUTPUT = "api_pilot/outputs/experiment_completion_contract.json"
SUMMARY_MD = "api_pilot/reports/final/FINAL_EXPERIMENT_RESULTS.md"
SUMMARY_CSV = "api_pilot/outputs/final_experiment_results.csv"
FAILURES_CSV = "api_pilot/outputs/final_failure_inventory.csv"
REVIEW_PACKET = "api_pilot/human_review/BLIND_QUALITY_REVIEW.md"
REVIEW_TEMPLATE = "api_pilot/human_review/blind_quality_review_template.csv"
REVIEW_SIGNOFF = "api_pilot/human_review/human_quality_review_signoff.json"
REVIEW_SIGNOFF_TEMPLATE = (
    "api_pilot/human_review/human_quality_review_signoff.template.json"
)

ROBUSTNESS_REPORTS = {
    "repeat_identical": "api_pilot/outputs/repeat_identical_robustness_analysis.json",
    "single_item": "api_pilot/outputs/single_item_robustness_analysis.json",
    "wording_b": "api_pilot/outputs/wording_b_robustness_analysis.json",
    "option_reverse": "api_pilot/outputs/option_reverse_robustness_analysis.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def audit_jsonl(path: Path, expected: int) -> dict[str, Any]:
    """Audit the latest immutable JSONL without loading its payloads in memory."""
    audit = {
        "path": path.as_posix(), "expected": expected, "observed": 0,
        "unique_call_ids": 0, "duplicate_call_ids": 0, "failed_rows": 0,
        "terminal_rows": 0, "malformed_rows": 0, "complete": False,
        "sha256": None, "failed_call_ids": [], "terminal_call_ids": [],
        "duplicate_ids": [], "malformed_line_numbers": [],
    }
    if not path.exists():
        return audit
    call_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            audit["observed"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                audit["malformed_rows"] += 1
                audit["malformed_line_numbers"].append(line_number)
                continue
            call_id = row.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                audit["malformed_rows"] += 1
                audit["malformed_line_numbers"].append(line_number)
            elif call_id in call_ids:
                audit["duplicate_call_ids"] += 1
                audit["duplicate_ids"].append(call_id)
            else:
                call_ids.add(call_id)
            if not row.get("success"):
                audit["failed_rows"] += 1
                audit["failed_call_ids"].append(call_id or f"line:{line_number}")
            if row.get("terminal_failure"):
                audit["terminal_rows"] += 1
                audit["terminal_call_ids"].append(call_id or f"line:{line_number}")
    audit["unique_call_ids"] = len(call_ids)
    audit["sha256"] = sha256(path)
    audit["complete"] = (
        audit["observed"] == expected
        and audit["unique_call_ids"] == expected
        and not any(
            audit[name]
            for name in (
                "duplicate_call_ids", "failed_rows", "terminal_rows",
                "malformed_rows",
            )
        )
    )
    return audit


def exact_status(
    report: dict[str, Any], status: str, boolean_fields: Iterable[str]
) -> bool:
    return report.get("status") == status and all(
        report.get(field) is True for field in boolean_fields
    )


def _record(
    checks: list[dict[str, Any]], failures: list[dict[str, Any]],
    check_id: str, passes: bool, detail: str,
) -> None:
    checks.append({"id": check_id, "passes": bool(passes), "detail": detail})
    if not passes:
        failures.append({"id": check_id, "detail": detail})


def _hash_bound(
    root: Path, checks: list[dict[str, Any]], failures: list[dict[str, Any]],
    check_id: str, relative_path: str, expected: str,
) -> bool:
    if not relative_path or not expected:
        _record(
            checks, failures, check_id, False,
            f"missing frozen path or hash; file={relative_path!r}; expected={expected!r}",
        )
        return False
    path = root / relative_path
    observed = sha256(path) if path.is_file() else None
    passes = observed == expected
    _record(
        checks, failures, check_id, passes,
        f"expected={expected}; observed={observed}; file={relative_path}",
    )
    return passes


def human_review_status(root: Path, packet_hash: str | None) -> dict[str, Any]:
    path = root / REVIEW_SIGNOFF
    result = {
        "required": True, "status": "pending", "passes": False,
        "path": REVIEW_SIGNOFF, "reason": "signed human review is absent",
    }
    if not path.exists():
        return result
    try:
        signoff = load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        result["reason"] = f"invalid sign-off JSON: {type(error).__name__}"
        return result
    checks = {
        "approved": signoff.get("status") == "approved",
        "reviewer": bool(str(signoff.get("reviewer", "")).strip()),
        "reviewed_at": bool(str(signoff.get("reviewed_at", "")).strip()),
        "packet_hash": packet_hash is not None
        and signoff.get("review_packet_sha256") == packet_hash,
        "no_unresolved_rejects": not signoff.get("unresolved_rejections"),
    }
    result.update({
        "status": "approved" if all(checks.values()) else "invalid_or_incomplete",
        "passes": all(checks.values()), "checks": checks,
        "reason": "all sign-off fields verified" if all(checks.values())
        else "one or more sign-off fields are missing or inconsistent",
    })
    return result


def _safe_load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "file is absent"
    try:
        return load_json(path), None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def evaluate(root: Path = ROOT) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}

    core_path = root / CORE_MANIFEST
    core, error = _safe_load(core_path)
    _record(checks, failures, "core_manifest_readable", core is not None, error or CORE_MANIFEST)
    if core is None:
        return _contract(checks, failures, audits, None)

    source = core.get("source") or {}
    prompt = core.get("prompt") or {}
    for name, path_key, hash_key in (
        ("profiles", "profile_file", "profile_file_sha256"),
        ("analysis_manifest", "analysis_manifest_file", "analysis_manifest_sha256"),
        ("label_assignment", "label_assignment_file", "label_assignment_sha256"),
        ("donor_assignment", "donor_assignment_file", "donor_assignment_sha256"),
    ):
        _hash_bound(root, checks, failures, f"frozen_{name}_hash", source.get(path_key, ""), source.get(hash_key, ""))
    _hash_bound(root, checks, failures, "frozen_prompt_hash", prompt.get("file", ""), prompt.get("sha256", ""))

    raw_path = root / core["immutable_output_path"]
    raw_audit = audit_jsonl(raw_path, int(core["expected"]["total_calls"]))
    audits["raw_core"] = raw_audit
    _record(checks, failures, "raw_core_complete", raw_audit["complete"], json.dumps(raw_audit, ensure_ascii=False))

    logical_rel = core["analysis_contract"]["cached_logical_panel"]["derived_output_path"]
    logical_audit = audit_jsonl(root / logical_rel, 64_512)
    audits["logical_panel"] = logical_audit
    _record(checks, failures, "logical_panel_complete", logical_audit["complete"], json.dumps(logical_audit, ensure_ascii=False))

    cache, cache_error = _safe_load(root / CACHE_REPORT)
    cache_ok = cache is not None and exact_status(
        cache, "cached_logical_panel_complete", ("logical_panel_generated",)
    ) and (cache.get("cache_audit") or {}).get("complete") is True
    if cache_ok:
        cache_ok = (
            cache["inputs"].get("raw_results_sha256") == raw_audit["sha256"]
            and (cache.get("logical_output") or {}).get("sha256") == logical_audit["sha256"]
            and (cache.get("cache_audit") or {}).get("logical_rows") == 64_512
        )
    _record(checks, failures, "cache_report_complete_and_bound", cache_ok, cache_error or (cache or {}).get("status", "missing"))

    strict, strict_error = _safe_load(root / STRICT_REPORT)
    strict_ok = strict is not None and exact_status(
        strict, "strict_confirmatory_inference_complete",
        ("hypothesis_results_generated",),
    ) and (strict.get("source_audit") or {}).get("complete") is True
    if strict_ok:
        strict_ok = (
            strict["inputs"].get("results") == raw_audit["sha256"]
            and strict["inputs"].get("task_manifest") == sha256(core_path)
            and isinstance((strict.get("inference") or {}).get("holm_adjusted_p"), dict)
        )
    _record(checks, failures, "strict_inference_complete_and_bound", strict_ok, strict_error or (strict or {}).get("status", "missing"))

    comprehensive, comp_error = _safe_load(root / COMPREHENSIVE_REPORT)
    comp_ok = comprehensive is not None and exact_status(
        comprehensive, "confirmatory_secondary_metrics_complete",
        ("estimates_generated",),
    ) and (comprehensive.get("source_audit") or {}).get("complete") is True
    if comp_ok:
        comp_ok = (
            comprehensive["inputs"].get("raw_results") == raw_audit["sha256"]
            and comprehensive["inputs"].get("logical_results") == logical_audit["sha256"]
            and (comprehensive.get("cache_audit") or {}).get("complete") is True
            and (comprehensive.get("item_panel_audit") or {}).get("complete") is True
        )
    _record(checks, failures, "comprehensive_metrics_complete_and_bound", comp_ok, comp_error or (comprehensive or {}).get("status", "missing"))

    robustness_suite, robustness_error = _safe_load(root / ROBUSTNESS_SUITE)
    _record(checks, failures, "robustness_suite_readable", robustness_suite is not None, robustness_error or ROBUSTNESS_SUITE)
    if robustness_suite:
        _record(
            checks, failures, "robustness_suite_exact_roster",
            set(robustness_suite.get("manifests") or {}) == set(ROBUSTNESS_REPORTS),
            f"observed={sorted((robustness_suite.get('manifests') or {}).keys())}",
        )
        for name, entry in robustness_suite["manifests"].items():
            if name not in ROBUSTNESS_REPORTS:
                continue
            manifest_rel = entry["file"]
            manifest_hash_ok = _hash_bound(root, checks, failures, f"robustness_{name}_manifest_hash", manifest_rel, entry["sha256"])
            manifest = load_json(root / manifest_rel) if manifest_hash_ok else None
            output_audit = audit_jsonl(root / manifest["immutable_output_path"], int(manifest["expected"]["total_calls"])) if manifest else {"complete": False}
            audits[f"robustness_{name}"] = output_audit
            report, report_error = _safe_load(root / ROBUSTNESS_REPORTS[name])
            report_ok = report is not None and report.get("status") == "complete" and report.get("complete") is True and (report.get("gate_decision") or {}).get("passes") is True and report.get("manifest_sha256") == entry["sha256"]
            _record(checks, failures, f"robustness_{name}_complete", bool(output_audit.get("complete")) and report_ok, report_error or (report or {}).get("status", "missing"))

    sentinel_suite, sentinel_error = _safe_load(root / SENTINEL_SUITE)
    _record(checks, failures, "sentinel_suite_readable", sentinel_suite is not None, sentinel_error or SENTINEL_SUITE)
    sentinel_suite_hash = sha256(root / SENTINEL_SUITE) if sentinel_suite else None
    if sentinel_suite:
        _record(
            checks, failures, "sentinel_suite_exact_waves",
            set(sentinel_suite.get("waves") or {}) == {"before", "mid", "after"},
            f"observed={sorted((sentinel_suite.get('waves') or {}).keys())}",
        )
        _record(checks, failures, "sentinel_core_manifest_hash", sentinel_suite.get("core_manifest_sha256") == sha256(core_path), f"suite={sentinel_suite.get('core_manifest_sha256')}; core={sha256(core_path)}")
        for wave, entry in sentinel_suite["waves"].items():
            manifest_ok = _hash_bound(root, checks, failures, f"sentinel_{wave}_manifest_hash", entry["manifest"], entry["manifest_sha256"])
            manifest = load_json(root / entry["manifest"]) if manifest_ok else None
            wave_audit = audit_jsonl(root / manifest["immutable_output_path"], int(manifest["expected"]["total_calls"])) if manifest else {"complete": False}
            audits[f"sentinel_{wave}"] = wave_audit
            _record(checks, failures, f"sentinel_{wave}_complete", bool(wave_audit.get("complete")), json.dumps(wave_audit, ensure_ascii=False))
    sentinel, sentinel_report_error = _safe_load(root / SENTINEL_REPORT)
    sentinel_ok = sentinel is not None and exact_status(sentinel, "complete", ("complete", "estimates_generated")) and sentinel.get("suite_sha256") == sentinel_suite_hash and (sentinel.get("core_audit") or {}).get("complete") is True and all((value or {}).get("complete") is True for value in (sentinel.get("wave_audits") or {}).values()) and set((sentinel.get("wave_audits") or {})) == {"before", "mid", "after"}
    _record(checks, failures, "sentinel_analysis_complete_and_bound", sentinel_ok, sentinel_report_error or (sentinel or {}).get("status", "missing"))

    runtime, runtime_error = _safe_load(root / RUNTIME_QUALITY_REPORT)
    runtime_csv = (runtime or {}).get("artifacts", {}).get("provider_csv", "")
    runtime_csv_path = root / runtime_csv if runtime_csv else None
    runtime_ok = (
        runtime is not None and runtime.get("status") == "complete"
        and runtime.get("complete") is True
        and runtime.get("expected_total_calls") == 37_344
        and runtime.get("observed_planned_calls") == 37_344
        and set((runtime.get("sources") or {})) == {
            "core", "robustness_repeat_identical", "robustness_single_item",
            "robustness_wording_b", "robustness_option_reverse",
            "sentinel_before", "sentinel_mid", "sentinel_after",
        }
        and all((value.get("audit") or {}).get("complete") is True
                for value in (runtime.get("sources") or {}).values())
        and runtime_csv_path is not None and runtime_csv_path.is_file()
        and sha256(runtime_csv_path) == (runtime.get("artifacts") or {}).get("provider_csv_sha256")
    )
    _record(checks, failures, "runtime_quality_complete_and_bound", runtime_ok,
            runtime_error or (runtime or {}).get("status", "missing"))

    machine_complete = not failures
    documents = {"strict": strict, "comprehensive": comprehensive, "sentinel": sentinel}
    return _contract(checks, failures, audits, documents, machine_complete=machine_complete)


def _contract(
    checks: list[dict[str, Any]], failures: list[dict[str, Any]],
    audits: dict[str, Any], documents: dict[str, Any] | None,
    *, machine_complete: bool = False,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "experiment-completion-contract-v1",
        "machine_complete": machine_complete,
        "status": "machine_complete" if machine_complete else "pending_machine_outputs",
        "checks": checks, "failures": failures, "audits": audits,
        "documents": documents or {},
        "guardrail": (
            "Completion is an integrity state, not a claim that H1 or H2 is true. "
            "Null, mixed and positive outcomes are all valid completed outcomes."
        ),
    }


def metric_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    if not contract.get("machine_complete"):
        return []
    strict = contract["documents"]["strict"]
    comprehensive = contract["documents"]["comprehensive"]
    inference = strict["inference"]
    rows: list[dict[str, Any]] = []
    for key, result in inference["direction_bootstrap"].items():
        rows.append({"scope": "confirmatory_high_macro", "metric": key, **{name: result.get(name) for name in ("estimate", "ci_low", "ci_high", "p_greater")}, "holm_adjusted_p": inference["holm_adjusted_p"].get("H1") if key in {"beta_o", "beta_diff"} else None})
    result = inference["utility_h2"]
    rows.append({"scope": "confirmatory_high_macro", "metric": "utility_v", **{name: result.get(name) for name in ("estimate", "ci_low", "ci_high", "p_greater")}, "holm_adjusted_p": inference["holm_adjusted_p"].get("H2")})
    official = comprehensive["analysis"]["provider_sets"]["official_four"]
    for stratum, values in official["strata"].items():
        for group in ("direction", "prediction"):
            for metric, result in values[group].items():
                macro = result["macro"]
                rows.append({"scope": f"secondary_{stratum}", "metric": metric, "estimate": macro.get("estimate"), "ci_low": macro.get("ci_low"), "ci_high": macro.get("ci_high"), "p_greater": None, "holm_adjusted_p": None})
    return rows


def render_summary(contract: dict[str, Any], review: dict[str, Any]) -> str:
    lines = [
        "# Final Experiment Results", "",
        f"> Machine status: **{contract['status']}**", "",
        "This report is generated offline from hash-bound frozen outputs. "
        "Completion does not depend on whether a hypothesis is supported.", "",
    ]
    if not contract.get("machine_complete"):
        lines += ["No confirmatory estimate is reported because the completion contract is not closed.", "", "## Pending checks", ""]
        lines += [f"- `{row['id']}`: {row['detail']}" for row in contract["failures"]]
    else:
        lines += ["| Scope | Metric | Estimate | 95% CI | Holm p |", "|---|---|---:|---:|---:|"]
        for row in metric_rows(contract)[:4]:
            ci = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            holm = "" if row["holm_adjusted_p"] is None else f"{row['holm_adjusted_p']:.4g}"
            lines.append(f"| {row['scope']} | {row['metric']} | {row['estimate']:.4f} | {ci} | {holm} |")
        sentinel = contract["documents"]["sentinel"]["metrics"]["overall"]
        lines += ["", "## Runtime drift diagnostic", "", f"Mean prompt-identical TV: **{sentinel['mean_tv']:.4f}** (95% CI {sentinel['ci95'][0]:.4f} to {sentinel['ci95'][1]:.4f}). This is diagnostic and does not alter the frozen hypotheses."]
    lines += ["", "## Human quality review", "", f"Status: **{review['status']}**. {review['reason']}", "", "## Interpretation guardrail", "", "Country metadata effects apply only to the frozen models, countries, items and English prompts. G0 must be positive before reduced evidence advantage is described as override. Country-survey contrasts are not cultural essences or evidence about training-data causes.", ""]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def failure_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"category": "completion_check", "source": "completion_contract",
         "failure_id": failure["id"], "detail": failure["detail"]}
        for failure in contract["failures"]
    ]
    for source, audit in contract.get("audits", {}).items():
        for call_id in audit.get("failed_call_ids", []):
            rows.append({"category": "failed_call", "source": source,
                         "failure_id": call_id, "detail": "success=false"})
        for call_id in audit.get("terminal_call_ids", []):
            rows.append({"category": "terminal_call", "source": source,
                         "failure_id": call_id, "detail": "terminal_failure=true"})
        for call_id in audit.get("duplicate_ids", []):
            rows.append({"category": "duplicate_call", "source": source,
                         "failure_id": call_id, "detail": "duplicate call_id"})
        for line in audit.get("malformed_line_numbers", []):
            rows.append({"category": "malformed_row", "source": source,
                         "failure_id": f"line:{line}", "detail": "invalid JSON or call_id"})
    return rows


def write_review_materials(root: Path) -> str:
    prompt = load_json(root / "api_pilot/prompts/survey-confirmatory-v4.json")
    analysis = load_json(root / "dataset/analysis/confirmatory_manifest_v2.json")
    profile_path = root / "dataset/derived/confirmatory_profiles_6x84.jsonl"
    first = json.loads(next(line for line in profile_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()))
    roles = {item: stratum for stratum, items in analysis["target_strata"].items() for item in items}
    review_rows = []
    for item in first["evidence"]:
        item_id = item["item_id"]
        spec = prompt["items"][item_id]
        review_rows.append({"review_unit": item_id, "role": "evidence", "question": spec["question"], "scale": spec["scale"], "question_scale_match": "", "country_referential": "", "ambiguous_or_mistranslated": "", "acceptable": "", "notes": ""})
    for item_id in [item for batch in analysis["target_batches"].values() for item in batch]:
        spec = prompt["items"][item_id]
        review_rows.append({"review_unit": item_id, "role": f"target:{roles[item_id]}", "question": spec["question"], "scale": spec["scale"], "question_scale_match": "", "country_referential": "", "ambiguous_or_mistranslated": "", "acceptable": "", "notes": ""})
    for condition, text in prompt["conditions"].items():
        review_rows.append({"review_unit": f"prompt_{condition}", "role": "metadata_prompt", "question": text or "No country metadata", "scale": "", "question_scale_match": "", "country_referential": "", "ambiguous_or_mistranslated": "", "acceptable": "", "notes": ""})
    write_csv(root / REVIEW_TEMPLATE, review_rows, list(review_rows[0]))
    lines = ["# Blind Quality Review Packet", "", "> Review the frozen wording without inspecting model outputs, effect sizes or significance tests.", "", "Mark every row in `blind_quality_review_template.csv`. Check question-scale agreement, unintended country/government references, ambiguity, translation damage and whether MO/MR/MV provenance statements match their definitions.", "", "## Metadata conditions", ""]
    for condition, text in prompt["conditions"].items():
        lines += [f"### {condition}", "", text or "No country metadata.", ""]
    lines += ["## Item roster", "", "| Item | Role | Question | Scale |", "|---|---|---|---|"]
    for row in review_rows:
        if row["role"] == "metadata_prompt":
            continue
        question = row["question"].replace("|", "\\|")
        scale = row["scale"].replace("|", "\\|")
        lines.append(f"| {row['review_unit']} | {row['role']} | {question} | {scale} |")
    lines += ["", "## Sign-off", "", f"After review, create `{REVIEW_SIGNOFF}` with `status`, `reviewer`, `reviewed_at`, `review_packet_sha256`, and `unresolved_rejections`. Approval requires no unresolved rejection.", ""]
    atomic_text(root / REVIEW_PACKET, "\n".join(lines))
    signoff_template = {
        "status": "pending",
        "reviewer": "",
        "reviewed_at": "",
        "review_packet_sha256": "REPLACE_WITH_PACKET_SHA256",
        "unresolved_rejections": [],
        "notes": "",
    }
    atomic_json(root / REVIEW_SIGNOFF_TEMPLATE, signoff_template)
    return sha256(root / REVIEW_PACKET)


def write_artifacts(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    packet_hash = write_review_materials(root)
    review = human_review_status(root, packet_hash)
    if contract["machine_complete"]:
        contract["status"] = "complete" if review["passes"] else "awaiting_human_review"
    contract["human_review"] = review
    contract["artifacts"] = {
        "summary_markdown": SUMMARY_MD, "summary_csv": SUMMARY_CSV,
        "failure_inventory_csv": FAILURES_CSV, "review_packet": REVIEW_PACKET,
        "review_packet_sha256": packet_hash, "review_template": REVIEW_TEMPLATE,
        "review_signoff_template": REVIEW_SIGNOFF_TEMPLATE,
    }
    rows = metric_rows(contract)
    write_csv(root / SUMMARY_CSV, rows, ["scope", "metric", "estimate", "ci_low", "ci_high", "p_greater", "holm_adjusted_p"])
    write_csv(
        root / FAILURES_CSV, failure_rows(contract),
        ["category", "source", "failure_id", "detail"],
    )
    atomic_text(root / SUMMARY_MD, render_summary(contract, review))
    atomic_json(root / CONTRACT_OUTPUT, {key: value for key, value in contract.items() if key != "documents"})
    return contract


def main() -> int:
    five_model_contract = ROOT / "api_pilot/outputs/experiment_completion_contract.json"
    if five_model_contract.exists():
        current = load_json(five_model_contract)
        if current.get("schema_version") == "five-model-experiment-completion-v1":
            print(json.dumps({
                "status": current.get("status"),
                "machine_complete": current.get("machine_complete"),
                "legacy_four_model_verifier_skipped": True,
                "reason": "the five-model completion contract is authoritative",
            }, ensure_ascii=False))
            return 0
    contract = write_artifacts(ROOT, evaluate(ROOT))
    print(json.dumps({"status": contract["status"], "machine_complete": contract["machine_complete"], "failed_checks": len(contract["failures"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
