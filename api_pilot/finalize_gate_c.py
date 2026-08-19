#!/usr/bin/env python3
"""Finalize complete Gate C checkpoints and run frozen downstream analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import survey_pilot_runner as runner


Z_ALPHA_TWO_SIDED = 1.959964
Z_POWER_80 = 0.841621
PILOT_PROFILES = 72
CONFIRMATORY_PROFILES = 504


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_checkpoint(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as connection:
        metadata_row = connection.execute(
            "SELECT value FROM run_meta WHERE key='run_metadata'"
        ).fetchone()
        if metadata_row is None:
            raise ValueError(f"{path}: missing run metadata")
        metadata = json.loads(metadata_row[0])
        rows = [
            json.loads(row[0])
            for row in connection.execute("SELECT result_json FROM results")
        ]
    return metadata, rows


def immutable_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("provider"),
        row.get("model_requested"),
        row.get("profile_id"),
        row.get("condition"),
        row.get("evidence_variant"),
        row.get("batch_id"),
        tuple(row.get("target_ids") or ()),
        row.get("prompt_sha256"),
    )


def expected_tasks(
    metadata: dict[str, Any], profiles_path: Path, prompt_path: Path,
    model_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    providers = list((metadata.get("models") or {}).keys())
    models = dict(metadata["models"])
    models.update(model_overrides or {})
    profiles = runner.load_profiles(profiles_path, 12)
    bundle = runner.load_prompt_bundle(prompt_path)
    tasks = runner.build_tasks(
        profiles,
        providers,
        models,
        bundle=bundle,
        target_batch_size=5,
    )
    if len(tasks) != int(metadata.get("planned_calls") or 0):
        raise ValueError(
            f"rebuilt {len(tasks)} tasks, metadata plans {metadata.get('planned_calls')}"
        )
    return tasks


def merge_results(
    tasks: list[dict[str, Any]],
    checkpoints: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {task["call_id"]: task for task in tasks}
    merged: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    raw_valid_by_provider: dict[str, set[str]] = {}
    repair_calls: set[str] = set()
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        if not checkpoint.exists():
            continue
        _, rows = read_checkpoint(checkpoint)
        for row in rows:
            call_id = row.get("call_id")
            if call_id not in expected:
                provider = row.get("provider")
                expected_model = next(
                    (task["model"] for task in tasks if task["provider"] == provider),
                    None,
                )
                if expected_model and row.get("model_requested") != expected_model:
                    continue
                raise ValueError(f"{checkpoint}: unexpected call ID {call_id}")
            task = expected[call_id]
            expected_identity = (
                task["provider"], task["model"], task["profile"]["profile_id"],
                task["condition"], task["evidence_variant"], task["batch_id"],
                tuple(target["item_id"] for target in task["targets"]),
                runner.sha256_text(task["system_prompt"] + "\n" + task["prompt"]),
            )
            if immutable_identity(row) != expected_identity:
                raise ValueError(f"{checkpoint}: immutable identity mismatch for {call_id}")
            if row.get("success") and not row.get("postprocess_repair"):
                raw_valid_by_provider.setdefault(row["provider"], set()).add(call_id)
            previous = merged.get(call_id)
            if previous is None or row.get("success") or not previous.get("success"):
                merged[call_id] = row
                sources[call_id] = str(checkpoint)
            if row.get("postprocess_repair"):
                repair_calls.add(call_id)
    missing = sorted(set(expected).difference(merged))
    incomplete = sorted(
        call_id for call_id, row in merged.items() if not row.get("success")
    )
    complete = not missing and not incomplete and len(merged) == len(expected)
    ordered = [merged[task["call_id"]] for task in tasks if task["call_id"] in merged]
    audit = {
        "complete": complete,
        "planned": len(expected),
        "merged": len(merged),
        "missing": len(missing),
        "incomplete": len(incomplete),
        "missing_examples": missing[:10],
        "incomplete_examples": incomplete[:10],
        "raw_valid_by_provider": {
            provider: len(call_ids)
            for provider, call_ids in sorted(raw_valid_by_provider.items())
        },
        "postprocess_repairs": len(repair_calls),
        "checkpoint_precedence": [str(path) for path in checkpoints],
        "source_counts": {
            source: sum(value == source for value in sources.values())
            for source in sorted(set(sources.values()))
        },
    }
    return ordered, audit


def standard_error(metric: dict[str, float]) -> float:
    return (metric["ci_high"] - metric["ci_low"]) / (2 * Z_ALPHA_TWO_SIDED)


def required_profiles(metric: dict[str, float], sesoi: float) -> int:
    se = standard_error(metric)
    raw = PILOT_PROFILES * (
        (Z_ALPHA_TWO_SIDED + Z_POWER_80) * se / sesoi
    ) ** 2
    return max(6, math.ceil(raw / 6) * 6)


def build_gate_report(summary: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    providers = summary["providers"]
    provider_results = {}
    required_by_provider = {}
    for provider in providers:
        high = summary["metrics"][provider]["high_country_value"]
        h1 = high["mo_minus_mr_direction_beta"]
        utility = {
            "mean": -high["mv_brier_delta"]["mean"],
            "ci_low": -high["mv_brier_delta"]["ci_high"],
            "ci_high": -high["mv_brier_delta"]["ci_low"],
        }
        evidence = summary["metrics"][provider]["all"]["evidence_gain_m0"]
        raw_valid = audit["raw_valid_by_provider"].get(provider, 0)
        raw_rate = raw_valid / 864
        required = {
            "h1_provenance_attenuation": required_profiles(h1, 0.05),
            "h2_verified_country_utility": required_profiles(utility, 0.02),
        }
        required_by_provider[provider] = required
        provider_results[provider] = {
            "raw_schema_validity": raw_rate,
            "passes_98_percent_raw_validity": raw_rate >= 0.98,
            "postprocessed_analysis_cells": 864,
            "evidence_gain_m0": evidence,
            "evidence_gate_direction_positive": evidence["mean"] > 0,
            "h1_mo_minus_mr_direction_beta": h1,
            "h2_verified_country_brier_utility": utility,
            "required_profiles_normal_approximation": required,
        }
    worst_required = max(
        value
        for provider in required_by_provider.values()
        for value in provider.values()
    )
    core_checks = {
        "all_4320_cells_analysis_available": audit["complete"],
        "all_models_raw_validity_at_least_98_percent": all(
            result["passes_98_percent_raw_validity"]
            for result in provider_results.values()
        ),
        "all_models_positive_mean_evidence_gain": all(
            result["evidence_gate_direction_positive"]
            for result in provider_results.values()
        ),
        "normal_approximation_supports_504": worst_required <= CONFIRMATORY_PROFILES,
    }
    outstanding = {
        "single_item_vs_batch": "not run",
        "second_equivalent_wording": "not run",
        "prompt_identical_repeat_noise": "not run in this Gate C core checkpoint",
        "simulation_power": "not implemented; normal approximation is planning-only",
        "gpt_5_4_model_and_interface": "verified; Responses API",
    }
    macro = summary.get("macro_average") or {}
    high_macro = macro.get("high_country_value") or {}
    all_macro = macro.get("all") or {}
    macro_utility = None
    if high_macro.get("mv_brier_delta"):
        delta = high_macro["mv_brier_delta"]
        macro_utility = {
            "mean": -delta["mean"],
            "ci_low": -delta["ci_high"],
            "ci_high": -delta["ci_low"],
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Gate C core measurement pilot; not confirmatory paper results",
        "provider_results": provider_results,
        "macro_average": {
            "evidence_gain_m0_all_items": all_macro.get("evidence_gain_m0"),
            "high_country_value": {
                "beta_opaque": high_macro.get("mo_direction_beta"),
                "beta_random_disclosed": high_macro.get("mr_direction_beta"),
                "h1_attenuation": high_macro.get("mo_minus_mr_direction_beta"),
                "regret_opaque": high_macro.get("mo_brier_delta"),
                "regret_random_disclosed": high_macro.get("mr_brier_delta"),
                "h2_verified_country_utility": macro_utility,
            },
        },
        "worst_required_profiles_normal_approximation": worst_required,
        "core_checks": core_checks,
        "outstanding_gate_components": outstanding,
        "gate_c_core_pass": all(core_checks.values()),
        "full_gate_c_pass": False,
        "decision": (
            "Core pilot analysed. Do not start confirmatory inference until the "
            "outstanding robustness, simulation-power, and data-processing gates "
            "are resolved."
        ),
        "merge_audit": audit,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Gate C 自动评估报告",
        "",
        f"> 生成时间：`{report['generated_at']}`  ",
        "> 范围：核心测量 pilot，不是论文确认性结果。",
        "",
        "## 总体判定",
        "",
        f"- 4,320单元完整：**{report['merge_audit']['complete']}**",
        f"- 核心Gate通过：**{report['gate_c_core_pass']}**",
        f"- 完整Gate C通过：**{report['full_gate_c_pass']}**",
        f"- 最差模型正态近似所需profiles：**{report['worst_required_profiles_normal_approximation']}**",
        "- 决策：" + report["decision"],
        "",
        "## 模型结果",
        "",
        "| Provider | 原始格式有效率 | E gain均值 | H1 beta | H2 utility | H1所需N | H2所需N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, result in report["provider_results"].items():
        lines.append(
            f"| {provider} | {result['raw_schema_validity']:.2%} | "
            f"{result['evidence_gain_m0']['mean']:.4f} | "
            f"{result['h1_mo_minus_mr_direction_beta']['mean']:.4f} | "
            f"{result['h2_verified_country_brier_utility']['mean']:.4f} | "
            f"{result['required_profiles_normal_approximation']['h1_provenance_attenuation']} | "
            f"{result['required_profiles_normal_approximation']['h2_verified_country_utility']} |"
        )
    lines.extend(
        [
            "",
            "## 尚未完成",
            "",
            *[
                f"- `{name}`：{status}"
                for name, status in report["outstanding_gate_components"].items()
            ],
            "",
            "Mistral归一化修复不会覆盖原始checkpoint。论文需同时报告原始格式有效率与修复后的分析可用率。",
            "",
        ]
    )
    return "\n".join(lines)


def readable_markdown_report(report: dict[str, Any]) -> str:
    def ci(metric: dict[str, float] | None) -> str:
        if not metric:
            return "NA"
        return (
            f"{metric['mean']:.4f} "
            f"[{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]"
        )

    high = report["macro_average"]["high_country_value"]
    lines = [
        "# Gate C 自动评估报告",
        "",
        f"> 生成时间：`{report['generated_at']}`  ",
        "> 范围：72-profile Gate C 测量 pilot，不是论文确认性结果。",
        "",
        "## 总体判定",
        "",
        f"- 4,320 个冻结单元完整：**{report['merge_audit']['complete']}**",
        f"- 核心测量 pilot 通过：**{report['gate_c_core_pass']}**",
        f"- 完整 Gate C 通过：**{report['full_gate_c_pass']}**",
        f"- 正态近似下最差模型所需 profiles：**{report['worst_required_profiles_normal_approximation']}**",
        "- 当前决策：核心指标可用于规划；完成剩余稳健性 Gate 前不启动确认性推理。",
        "",
        "## 五模型宏平均",
        "",
        "区间按六国分层、共享匿名记录 bootstrap；模型等权。Brier regret 为正表示变差，utility 为正表示改善。",
        "",
        "| 指标 | 点估计 [95% CI] | Pilot 解释 |",
        "|---|---:|---|",
        f"| 全题个人证据增益 G0 | {ci(report['macro_average']['evidence_gain_m0_all_items'])} | 正值表示同一行 evidence 优于同国错配 evidence |",
        f"| 高国家信息题 βO | {ci(high['beta_opaque'])} | 来源不明标签的国家方向移动 |",
        f"| 高国家信息题 βR | {ci(high['beta_random_disclosed'])} | 已披露随机标签的残余方向移动 |",
        f"| H1: βO−βR | {ci(high['h1_attenuation'])} | 正值表示随机来源声明削弱标签效应 |",
        f"| MO regret | {ci(high['regret_opaque'])} | 正值表示来源不明标签增加 Brier |",
        f"| MR regret | {ci(high['regret_random_disclosed'])} | 正值表示随机标签仍增加 Brier |",
        f"| H2: MV utility | {ci(high['h2_verified_country_utility'])} | 正值表示核验真实国家降低 Brier |",
        "",
        "## 逐模型核心结果",
        "",
        "| Provider | 原始格式有效率 | G0 [95% CI] | H1 βO−βR [95% CI] | H2 utility [95% CI] | H1 所需 N | H2 所需 N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, result in report["provider_results"].items():
        lines.append(
            f"| {provider} | {result['raw_schema_validity']:.2%} | "
            f"{ci(result['evidence_gain_m0'])} | "
            f"{ci(result['h1_mo_minus_mr_direction_beta'])} | "
            f"{ci(result['h2_verified_country_brier_utility'])} | "
            f"{result['required_profiles_normal_approximation']['h1_provenance_attenuation']} | "
            f"{result['required_profiles_normal_approximation']['h2_verified_country_utility']} |"
        )
    lines.extend(["", "## 尚未完成的 Gate", ""])
    lines.extend(
        f"- `{name}`：{status}"
        for name, status in report["outstanding_gate_components"].items()
    )
    lines.extend(
        [
            "",
            "Mistral 的 12 条归一化修复没有覆盖原始 checkpoint。论文需同时报告原始格式有效率和修复后的分析可用率。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-checkpoint", type=Path, required=True)
    parser.add_argument("--overlay-checkpoint", type=Path, action="append", default=[])
    parser.add_argument(
        "--model-override", action="append", default=[], metavar="PROVIDER=MODEL"
    )
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--combined-results", type=Path, required=True)
    parser.add_argument("--analysis-summary", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    metadata, _ = read_checkpoint(args.main_checkpoint)
    model_overrides = dict(value.split("=", 1) for value in args.model_override)
    tasks = expected_tasks(metadata, args.profiles, args.prompt, model_overrides)
    checkpoints = [args.main_checkpoint, *args.overlay_checkpoint]
    rows, audit = merge_results(tasks, checkpoints)
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": "ready" if audit["complete"] else "waiting",
        "audit": audit,
    }
    atomic_write(args.status, json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    if not audit["complete"]:
        print(json.dumps(status, ensure_ascii=False))
        return 10
    if args.check_only:
        return 0

    atomic_write(
        args.combined_results,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    command = [
        sys.executable,
        "api_pilot/analyze_survey_pilot.py",
        "--results", str(args.combined_results),
        "--profiles", str(args.profiles),
        "--manifest", str(args.manifest),
        "--output", str(args.analysis_summary),
    ]
    completed = subprocess.run(command, cwd=Path.cwd(), check=False)
    if completed.returncode:
        raise RuntimeError(f"analysis failed with exit code {completed.returncode}")
    summary = json.loads(args.analysis_summary.read_text(encoding="utf-8"))
    report = build_gate_report(summary, audit)
    atomic_write(args.gate_report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write(args.report_markdown, readable_markdown_report(report))
    status.update(
        {
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                "combined_results": str(args.combined_results),
                "analysis_summary": str(args.analysis_summary),
                "gate_report": str(args.gate_report),
                "report_markdown": str(args.report_markdown),
            },
        }
    )
    atomic_write(args.status, json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
