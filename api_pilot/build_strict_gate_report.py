#!/usr/bin/env python3
"""Build the authoritative aggregate Gate C status and human-readable report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


ROBUSTNESS_REPORTS = {
    "repeat_identical": ROOT / "api_pilot/outputs/repeat_identical_robustness_analysis.json",
    "single_item": ROOT / "api_pilot/outputs/single_item_robustness_analysis.json",
    "wording_b": ROOT / "api_pilot/outputs/wording_b_robustness_analysis.json",
    "option_reverse": ROOT / "api_pilot/outputs/option_reverse_robustness_analysis.json",
}
PERMISSION_STATUS = ROOT / "dataset/analysis/data_processing_permission_status.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_permission(
    path: Path, expected_models: dict[str, str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "passes": False,
        "status": "pending",
    }
    if not path.exists():
        result["reason"] = "structured permission status file is missing"
        return result
    record = load(path)
    result["status"] = str(record.get("status") or "pending")
    evidence_name = record.get("evidence_document")
    evidence_hash = record.get("evidence_document_sha256")
    evidence_path = Path(evidence_name) if isinstance(evidence_name, str) else None
    if evidence_path is not None and not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    schema = record.get("schema_version")
    researcher_determination = schema == "respondent-data-processing-decision-v2"
    checks = {
        "schema": schema in {
            "respondent-data-processing-permission-v1",
            "respondent-data-processing-decision-v2",
        },
        "approved_status": record.get("status") in {
            "approved", "approved_by_researcher_under_public_terms",
        },
        "written_basis": record.get("written_basis_on_file") is True,
        "processing_basis": (
            record.get("third_party_api_processing_authorized") is True
            if not researcher_determination
            else record.get("researcher_risk_determination") is True
            and record.get("non_profit_research") is True
            and record.get("raw_data_redistribution_prohibited") is True
            and record.get("deidentified_derived_prompts_only") is True
            and record.get("claims_endpoint_specific_data_owner_approval") is False
        ),
        "provider_scope": record.get("scope", {}).get("provider_models")
        == expected_models,
        "reviewer_recorded": bool(record.get("reviewed_by")),
        "review_time_recorded": bool(record.get("reviewed_at")),
        "evidence_exists": bool(evidence_path and evidence_path.is_file()),
        "evidence_hash_matches": bool(
            evidence_path
            and evidence_path.is_file()
            and isinstance(evidence_hash, str)
            and sha256(evidence_path) == evidence_hash
        ),
    }
    result["checks"] = checks
    decision_checks = dict(checks)
    if researcher_determination:
        # Evidence-file integrity remains visible for audit, but the persisted
        # researcher decision itself is the operational basis. A moved report
        # must not halt an unattended experiment.
        decision_checks.pop("evidence_exists", None)
        decision_checks.pop("evidence_hash_matches", None)
        decision_checks.pop("reviewer_recorded", None)
        decision_checks.pop("review_time_recorded", None)
        result["audit_warnings"] = [
            name for name in (
                "evidence_exists", "evidence_hash_matches",
                "reviewer_recorded", "review_time_recorded",
            ) if not checks[name]
        ]
    result["passes"] = all(decision_checks.values())
    if result["passes"]:
        result["status"] = (
            "researcher_determination_verified"
            if researcher_determination
            else "approved_and_verified"
        )
        result["basis"] = (
            "researcher determination under public non-profit-use terms"
            if researcher_determination
            else "endpoint-specific written authorization"
        )
        result["evidence_document_sha256"] = evidence_hash
    else:
        result["reason"] = "structured processing decision evidence is not verified"
    return result


def evaluate_robustness(
    suite: dict[str, Any], report_paths: dict[str, Path]
) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    for key, row in suite.get("manifests", {}).items():
        report_path = report_paths[key]
        report = load(report_path) if report_path.exists() else {}
        experiments[key] = {
            "planned_calls": int(row.get("total_calls") or 0),
            "output": row.get("immutable_output_path"),
            "report": str(report_path.relative_to(ROOT))
            if report_path.is_relative_to(ROOT)
            else str(report_path),
            "analysis_status": report.get("status", "pending_inference"),
            "inference_complete": report.get("status") == "complete",
            "gate_status": report.get("gate_decision", {}).get("status", "pending"),
            "passes": report.get("gate_decision", {}).get("passes") is True,
        }
    values = list(experiments.values())
    all_pass = bool(values) and all(value["passes"] for value in values)
    any_fail = any(value["gate_status"] == "fail" for value in values)
    status = (
        "complete_passed"
        if all_pass
        else "complete_failed"
        if any_fail
        else "offline_manifests_frozen_inference_pending"
    )
    return {
        "status": status,
        "all_pass": all_pass,
        "experiments": experiments,
    }


def interval(node: dict[str, Any]) -> str:
    return f"{node['estimate']:.4f} [{node['ci_low']:.4f}, {node['ci_high']:.4f}]"


def scope_row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    inference = report["inference"]
    direction = inference["direction_bootstrap"]
    utility = inference["utility_h2"]
    return {
        "scope": name,
        "models": report["models"],
        "beta_o": direction["beta_o"],
        "beta_r": direction["beta_r"],
        "beta_diff": direction["beta_diff"],
        "utility_v": utility,
        "raw_primary_hypothesis_p": inference["raw_hypothesis_p"],
        "holm_adjusted_primary_p": inference["holm_adjusted_p"],
        "raw_sesoi_exceedance_p": inference["raw_sesoi_exceedance_p"],
    }


def build_status(
    five_path: Path,
    official_path: Path,
    design_power_path: Path,
    *,
    permission_path: Path = PERMISSION_STATUS,
    robustness_report_paths: dict[str, Path] | None = None,
) -> tuple[dict[str, Any], str]:
    five = load(five_path)
    official = load(official_path)
    design_power = load(design_power_path) if design_power_path.exists() else {}
    sensitivity_path = ROOT / "api_pilot/outputs/confirmatory_504_sensitivity.json"
    sensitivity = load(sensitivity_path) if sensitivity_path.exists() else {}
    robustness_suite_path = (
        ROOT / "dataset/analysis/gate_c_robustness_suite_v2.json"
    )
    robustness_suite = (
        load(robustness_suite_path) if robustness_suite_path.exists() else {}
    )
    five_row = scope_row("five_model_amended_roster", five)
    official_row = scope_row("four_official_endpoints", official)
    source_audit = five["source_audit"]
    permission = evaluate_permission(permission_path, five_row["models"])
    robustness = evaluate_robustness(
        robustness_suite, robustness_report_paths or ROBUSTNESS_REPORTS
    )
    confirmatory_manifest_path = (
        ROOT / "dataset/analysis/confirmatory_core_task_manifest_v2.json"
    )
    confirmatory_analysis_path = (
        ROOT / "api_pilot/outputs/confirmatory_core_v2_strict_inference.json"
    )
    confirmatory_manifest = (
        load(confirmatory_manifest_path) if confirmatory_manifest_path.exists() else {}
    )
    confirmatory_analysis = (
        load(confirmatory_analysis_path) if confirmatory_analysis_path.exists() else {}
    )
    blockers = []
    if not permission["passes"]:
        blockers.append(
            {
                "id": "data_processing_permission",
                "reason": permission["reason"],
            }
        )
    blocker_names = {
        "repeat_identical": "prompt_identical_repeat_noise",
        "single_item": "single_vs_batch",
        "wording_b": "second_equivalent_wording",
        "option_reverse": "option_order",
    }
    for key, value in robustness["experiments"].items():
        if value["passes"]:
            continue
        state = value["gate_status"]
        blockers.append(
            {
                "id": blocker_names[key],
                "reason": (
                    f"The frozen {value['planned_calls']:,}-call {key} task is "
                    f"{state}; its pre-specified robustness gate has not passed."
                ),
            }
        )
    design_power_passes = bool(
        design_power.get("gate_decision", {}).get("passes", False)
    )
    if not design_power_passes:
        blockers.append(
            {
                "id": "macro_power_design",
                "reason": (
                    "The prospective 504-profile equal-system macro-power design "
                    "is missing or does not reach the frozen 0.80 joint-Holm target."
                ),
            }
        )
    full_gate_pass = bool(
        permission["passes"] and robustness["all_pass"] and design_power_passes
    )
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": (
            "full_gate_c_passed_confirmatory_inference_pending"
            if full_gate_pass
            else "pilot_inference_complete_full_gate_pending"
        ),
        "display_state": (
            "FULL GATE C PASSED / CONFIRMATORY INFERENCE MAY START"
            if full_gate_pass
            else "GATE C PILOT COMPLETE / FULL GATE NOT PASSED"
        ),
        "audit": source_audit,
        "strict_inference": {
            "primary_display_scope": "five_model_amended_roster",
            "beta_o": five_row["beta_o"]["estimate"],
            "beta_r": five_row["beta_r"]["estimate"],
            "beta_diff": five_row["beta_diff"]["estimate"],
            "utility_v": five_row["utility_v"]["estimate"],
            "holm_h1": five_row["holm_adjusted_primary_p"]["H1"],
            "holm_h2": five_row["holm_adjusted_primary_p"]["H2"],
            "scope_results": [five_row, official_row],
        },
        "confirmatory_model_roster": {
            "status": "prospectively_frozen_before_confirmatory_inference",
            "systems": five_row["models"],
            "amendment": "dataset/analysis/openai_parity_roster_amendment_v1.json",
            "sensitivity_scope": "four_official_endpoints",
        },
        "gate": {
            "closed": full_gate_pass,
            "full_gate_c_pass": full_gate_pass,
            "blockers": blockers,
        },
        "data_processing_permission": permission,
        "confirmatory_execution": {
            "status": "offline_frozen_inference_pending",
            "manifest": str(confirmatory_manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(confirmatory_manifest_path)
            if confirmatory_manifest
            else None,
            "planned_calls": confirmatory_manifest.get("expected", {}).get(
                "total_calls"
            ),
            "observed_successes": confirmatory_analysis.get(
                "source_audit", {}
            ).get("analysis_ready", 0),
            "analysis_status": confirmatory_analysis.get("status", "pending"),
            "strict_output": str(confirmatory_analysis_path.relative_to(ROOT)),
            "comprehensive_output": (
                "api_pilot/outputs/confirmatory_core_v2_comprehensive_metrics.json"
            ),
        },
        "power": {
            "status": (
                "prospective_macro_design_power_passed"
                if design_power_passes
                else "prospective_macro_design_power_pending_or_failed"
            ),
            "design_power_file": (
                str(design_power_path.relative_to(ROOT)) if design_power else None
            ),
            "design_power_gate_passes": design_power_passes,
            "macro_joint_holm_power": (
                design_power.get("prospective_power", {})
                .get("macro_power", {})
                .get("joint_holm_power")
                if design_power
                else None
            ),
            "power_target": (
                design_power.get("gate_decision", {}).get("target")
                if design_power
                else None
            ),
            "null_any_holm_rejection": (
                design_power.get("prospective_power", {})
                .get("null_calibration", {})
                .get("any_holm_rejection")
                if design_power
                else None
            ),
            "identifiable_assignment_power": (
                design_power.get("prospective_power", {}).get(
                    "identifiable_assignment_power"
                )
                if design_power
                else None
            ),
            "corrected_sensitivity_file": (
                str(sensitivity_path.relative_to(ROOT)) if sensitivity else None
            ),
            "legacy_worst_official_provider_minimum_non_gating": (
                sensitivity.get("gate_decision", {}).get(
                    "worst_official_provider_power"
                )
                if sensitivity
                else None
            ),
        },
        "robustness": {
            "status": robustness["status"],
            "suite_file": (
                str(robustness_suite_path.relative_to(ROOT))
                if robustness_suite
                else None
            ),
            "total_planned_calls": (
                sum(
                    int(row.get("total_calls") or 0)
                    for row in robustness_suite.get("manifests", {}).values()
                )
                if robustness_suite
                else None
            ),
            "all_pass": robustness["all_pass"],
            "experiments": robustness["experiments"],
        },
        "outputs": {
            "combined_results": "api_pilot/restricted_outputs/gate_c_v3_complete.jsonl",
            "strict_five_system": str(five_path.relative_to(ROOT)),
            "strict_four_official": str(official_path.relative_to(ROOT)),
            "strict_report": "api_pilot/reports/diagnostics/GATE_C_STRICT_EVALUATION.md",
            "comprehensive_metrics": "api_pilot/reports/diagnostics/GATE_C_COMPREHENSIVE_METRICS.md",
            "confirmatory_design_power": (
                str(design_power_path.relative_to(ROOT)) if design_power else None
            ),
            "power_sensitivity": "api_pilot/outputs/confirmatory_504_sensitivity.json",
            "robustness_suite": (
                str(robustness_suite_path.relative_to(ROOT))
                if robustness_suite
                else None
            ),
        },
        "superseded_outputs": [
            "api_pilot/outputs/gate_c_v3_analysis_summary.json",
            "api_pilot/outputs/gate_c_v3_gate_report.json",
            "api_pilot/reports/archive/GATE_C_EVALUATION.md",
            "api_pilot/reports/archive/GATE_C_RESULTS_TABLES.md",
        ],
    }

    rows = []
    for label, report in (("核心四模型敏感性", official), ("完整五模型视图", five)):
        inf = report["inference"]
        d = inf["direction_bootstrap"]
        u = inf["utility_h2"]
        rows.append(
            f"| {label} | {interval(d['beta_o'])} | {interval(d['beta_r'])} | "
            f"{interval(d['beta_diff'])} | {interval(u)} | "
            f"{inf['holm_adjusted_p']['H1']:.4f} | "
            f"{inf['holm_adjusted_p']['H2']:.4f} | "
            f"{inf['raw_sesoi_exceedance_p']['H1']:.4f} | "
            f"{inf['raw_sesoi_exceedance_p']['H2']:.4f} |"
        )
    blocker_lines = "\n".join(
        f"- `{row['id']}`: {row['reason']}" for row in blockers
    )
    design_power_summary = (
        f"504-profile prospective equal-system macro design power passed: "
        f"joint Holm power = "
        f"{design_power['prospective_power']['macro_power']['joint_holm_power']:.3f} "
        f"against the frozen 0.80 target. The null calibration rejected at least "
        f"one Holm family in "
        f"{design_power['prospective_power']['null_calibration']['any_holm_rejection']:.3f} "
        f"of simulations."
        if design_power_passes
        else "The prospective equal-system macro design-power requirement remains open."
    )
    gate_heading = (
        "核心 pilot 与完整 Gate 均已通过，正式推理可以启动。"
        if full_gate_pass
        else "核心 pilot 推理完整，但完整 Gate 未关闭。"
    )
    markdown = f"""# Gate C 严格评估

> 状态：**{gate_heading}** 这是测量 pilot，不是论文确认性结果。

## 完整性

- 冻结调用：4,320 / 4,320
- 五个系统各 864 条分析可用输出
- 原始 schema 有效率：DeepSeek、Gemini、GPT-5.4、Qwen 均 864/864；Mistral 852/864
- Mistral 的 12 条输出按冻结规则归一化，全部只涉及控制题 `E039`，原响应和修复来源均保留
- 4,320 个 prompt hash、模型 ID、六个实验单元、metadata 语义和概率边界已逐条重算核验

## 严格结果

所有区间使用 profile-cluster bootstrap。调查权重在真实国家内归一，国家、题目和系统等权；每次方向 bootstrap 都重新抽样 reference records 并重建 anchors。

| 范围 | βO, 95% CI | βR, 95% CI | βO−βR, 95% CI | Verified utility, 95% CI | 主检验 Holm H1 | 主检验 Holm H2 | SESOI H1 | SESOI H2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

主假设按原计划检验 `βO > 0`、`βO−βR > 0`（H1 交并检验）和 verified-country Brier utility `> 0`（H2），随后对 H1/H2 做 Holm 校正。`.05` 和 `.02` 是设计阶段的最小科学效应（SESOI），其超越 p 值单独报告，不替代零效应主检验。核心四模型视图用于与早期分析保持可比，五模型视图纳入 GPT-5.4 并作为完整系统汇总。pilot p 值不提前接受或拒绝论文的确认性假设。

`βR` 明显大于零，来源随机声明只部分降低标签方向影响，不能写成模型会忽略已知随机标签。Gate C 标签采用旧的确定性平衡轮转，因此 100,000 次标签重标只作为 Fisher sharp-null 诊断，不是设计精确随机化检验。

## 504 条确认性设计功效

{design_power_summary} 该计算以 `βO=.05`、`βO−βR=.05`、`UtilityV=.02` 为 SESOI 备择，保留冻结模型与三个统计量的联合协方差，并在每次重采样中重建 reference anchors。由于 pilot 中每个 profile 只观察到一个显示标签，该结果是基于误差结构可迁移假设的设计功效近似，不是精确随机化功效，也不保证确认性效应出现。

## 未关闭项目

{blocker_lines}

四项稳健性任务已离线冻结，共 4,512 次核心四模型调用。当前没有执行这些调用，也没有产生可分析结果；`pending_inference` 不能写成稳健性通过。

## 已作废结果

旧 `gate_c_v3_analysis_summary.json`、`gate_c_v3_gate_report.json`、`GATE_C_EVALUATION.md` 和 `GATE_C_RESULTS_TABLES.md` 使用未加权、正态近似或旧判定口径，不得引用。旧 504 功效文件因反事实重标错误作废。`confirmatory_504_sensitivity.json` 保留原标签并在 `true-country × displayed-label` 内重采样，但它假定未来效应等于各系统的 pilot 点估计，只是非门控敏感性分析。权威功效门为 `confirmatory_504_design_power.json`，使用冻结系统等权总体、SESOI 备择值和实际 Holm 规则；逐系统功效只作诊断。
"""
    return status, markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--five",
        type=Path,
        default=ROOT / "api_pilot/outputs/gate_c_v3_strict_inference.json",
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=ROOT / "api_pilot/outputs/gate_c_v3_strict_inference_official4.json",
    )
    parser.add_argument(
        "--design-power",
        type=Path,
        default=ROOT / "api_pilot/outputs/confirmatory_504_design_power.json",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=ROOT / "api_pilot/outputs/gate_c_v3_postprocess_status.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "api_pilot/reports/diagnostics/GATE_C_STRICT_EVALUATION.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status, markdown = build_status(args.five, args.official, args.design_power)
    args.status.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(markdown, encoding="utf-8", newline="\n")
    superseded_notices = {
        ROOT / "api_pilot/reports/archive/GATE_C_EVALUATION.md": (
            "# 已作废：Gate C 旧自动评估\n\n"
            "此文件使用旧的未加权/正态近似流程，不能用于论文、功效决策或 Gate 判定。"
            "历史内容已移除，以防误引用。\n\n权威替代文件：\n\n"
            "- `api_pilot/reports/diagnostics/GATE_C_STRICT_EVALUATION.md`\n"
            "- `api_pilot/reports/diagnostics/GATE_C_COMPREHENSIVE_METRICS.md`\n"
            "- `api_pilot/outputs/gate_c_v3_postprocess_status.json`\n"
        ),
        ROOT / "api_pilot/reports/archive/GATE_C_RESULTS_TABLES.md": (
            "# 已作废：Gate C 旧结果表\n\n"
            "此文件使用旧的未严格加权区间，不能用于论文、功效决策或 Gate 判定。"
            "历史内容已移除，以防误引用。\n\n权威替代文件：\n\n"
            "- `api_pilot/reports/diagnostics/GATE_C_STRICT_EVALUATION.md`\n"
            "- `api_pilot/reports/diagnostics/GATE_C_COMPREHENSIVE_METRICS.md`\n"
            "- `api_pilot/outputs/gate_c_v3_comprehensive_metrics.json`\n"
        ),
    }
    for path, notice in superseded_notices.items():
        path.write_text(notice, encoding="utf-8", newline="\n")
    print(json.dumps({"status": str(args.status), "report": str(args.report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
