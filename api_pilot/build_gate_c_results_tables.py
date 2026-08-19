#!/usr/bin/env python3
"""Build readable Gate C pilot tables from the frozen aggregate summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def ci(metric: dict[str, float] | None, invert: bool = False) -> str:
    if not metric:
        return "NA"
    if invert:
        mean, low, high = -metric["mean"], -metric["ci_high"], -metric["ci_low"]
    else:
        mean, low, high = metric["mean"], metric["ci_low"], metric["ci_high"]
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary: dict[str, Any] = json.loads(args.summary.read_text(encoding="utf-8"))
    gate: dict[str, Any] = json.loads(args.gate_report.read_text(encoding="utf-8"))

    lines = [
        "# Gate C Pilot 结果表",
        "",
        "> 这些结果来自 72 个匿名调查记录、6 个国家、10 个 target、5 个固定 LLM 系统。它们用于测量验证和功效规划，不是 504-profile 确认性结果。",
        "",
        "## Table 1. 工程完整性",
        "",
        "| 系统 | 固定型号 | 计划单元 | 分析可用 | 原始格式有效率 | 平均延迟 | P95 延迟 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    models = {
        "deepseek": "deepseek-v4-pro",
        "gemini": "gemini-3.6-flash",
        "mistral": "mistral-medium-3-5",
        "openai": "gpt-5.4",
        "qwen": "qwen3.7-max-2026-06-08",
    }
    for provider in summary["providers"]:
        usage = summary["usage"][provider]
        result = gate["provider_results"][provider]
        lines.append(
            f"| {provider} | `{models[provider]}` | 864 | 864 | "
            f"{result['raw_schema_validity']:.2%} | "
            f"{usage['mean_latency_ms'] / 1000:.2f}s | {usage['p95_latency_ms'] / 1000:.2f}s |"
        )

    lines.extend(
        [
            "",
            "## Table 2. 逐模型核心机制结果",
            "",
            "`G0>0` 表示同一行个人证据有用。`βO−βR>0` 表示披露随机来源后国家方向效应减弱。Utility 为正表示核验真实国家降低 Brier。",
            "",
            "| 系统 | G0，全题 | βO，高信息题 | βR，高信息题 | H1 βO−βR | MO regret | MR regret | H2 MV utility |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for provider in summary["providers"]:
        all_metrics = summary["metrics"][provider]["all"]
        high = summary["metrics"][provider]["high_country_value"]
        lines.append(
            f"| {provider} | {ci(all_metrics['evidence_gain_m0'])} | "
            f"{ci(high['mo_direction_beta'])} | {ci(high['mr_direction_beta'])} | "
            f"{ci(high['mo_minus_mr_direction_beta'])} | "
            f"{ci(high['mo_brier_delta'])} | {ci(high['mr_brier_delta'])} | "
            f"{ci(high['mv_brier_delta'], invert=True)} |"
        )
    macro_all = summary["macro_average"]["all"]
    macro_high = summary["macro_average"]["high_country_value"]
    lines.append(
        f"| **五模型宏平均** | **{ci(macro_all['evidence_gain_m0'])}** | "
        f"**{ci(macro_high['mo_direction_beta'])}** | "
        f"**{ci(macro_high['mr_direction_beta'])}** | "
        f"**{ci(macro_high['mo_minus_mr_direction_beta'])}** | "
        f"**{ci(macro_high['mo_brier_delta'])}** | "
        f"**{ci(macro_high['mr_brier_delta'])}** | "
        f"**{ci(macro_high['mv_brier_delta'], invert=True)}** |"
    )

    lines.extend(
        [
            "",
            "## Table 3. 题目分层的五模型宏平均",
            "",
            "| 题目层 | βO | βR | H1 βO−βR | MO regret | MR regret | MV utility | MV accuracy Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    names = {
        "high_country_value": "国家信息题",
        "representative": "代表题",
        "equivalence_controls": "等价负控制题",
    }
    for stratum in ("high_country_value", "representative", "equivalence_controls"):
        metrics = summary["macro_average"][stratum]
        lines.append(
            f"| {names[stratum]} | {ci(metrics['mo_direction_beta'])} | "
            f"{ci(metrics['mr_direction_beta'])} | "
            f"{ci(metrics['mo_minus_mr_direction_beta'])} | "
            f"{ci(metrics['mo_brier_delta'])} | {ci(metrics['mr_brier_delta'])} | "
            f"{ci(metrics['mv_brier_delta'], invert=True)} | "
            f"{ci(metrics['mv_accuracy_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Table 4. 测量稳健性状态",
            "",
            "| 检查 | 当前状态 | 可否用于结论 |",
            "|---|---|---|",
            "| 五题 batch 的 JSON/概率合法率 | 通过，最低原始有效率 98.61% | 可以 |",
            "| 同一行 evidence 相对 shuffled evidence | 五模型宏平均 G0 区间高于 0 | 可以支持任务有效性 |",
            "| Prompt-identical repeat noise | 未运行 | 不可判断效应是否高于后端重复噪声 |",
            "| Single-item vs five-item batch | 未运行 | 不可排除 batch 位置效应 |",
            "| 第二种等义 provenance 措辞 | 未运行 | 不可判断 H1 是否依赖单一措辞 |",
            "| 模拟功效 | 未运行；目前仅正态近似 | 不足以冻结确认性样本量 |",
            "| GPT-5.4 模型与接口 | 已核实；Responses API | 按冻结模型 ID 报告 |",
            "",
            "## Pilot 结论边界",
            "",
            "五模型宏平均呈现与 H1 和 H2 一致的候选模式。来源不明标签沿显示国家方向移动，披露标签由独立随机机制产生后，该移动减弱但没有消失；核验真实国家在高国家信息题上降低了 Brier。MO/MR regret 的区间仍覆盖 0，因此 pilot 尚不足以断言无效 metadata 稳定造成预测损害。Mistral 的 H1 点估计与其他系统方向相反，表明模型异质性不能被宏平均掩盖。",
            "",
            "完整 Gate C 仍未通过。重复噪声、batch 一致性、第二措辞、模拟功效和模型来源问题解决前，不应启动或解释为确认性主实验。",
            "",
        ]
    )
    atomic_write(args.output, "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
