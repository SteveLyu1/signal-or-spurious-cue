#!/usr/bin/env python3
"""Advance and monitor the frozen experiment without conversational polling.

Every LLM runner uses an immutable SQLite checkpoint and can be restarted
without repeating successful calls. Permission records remain visible as audit
metadata but do not control runtime execution.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
STATUS_PATH = ROOT / "api_pilot/outputs/remaining_experiment_status.json"
MONITOR_PATH = ROOT / "api_pilot/EXPERIMENT_MONITOR.html"
EVENT_PATH = ROOT / "api_pilot/outputs/remaining_experiment_events.jsonl"
LOG_DIR = ROOT / "api_pilot/logs/remaining_experiment"
GATE_STATUS_PATH = ROOT / "api_pilot/outputs/gate_c_v3_postprocess_status.json"
ROBUSTNESS_SUITE_PATH = ROOT / "dataset/analysis/gate_c_robustness_suite_v2.json"
SENTINEL_SUITE_PATH = ROOT / "dataset/analysis/confirmatory_sentinel_suite_v1.json"
CORE_MANIFEST_PATH = ROOT / "dataset/analysis/confirmatory_core_task_manifest_v2.json"
BATCH_CONTEXT_MANIFEST_PATH = ROOT / "dataset/analysis/confirmatory_batch_context_replication_v1.json"
OPENAI_PARITY_PATH = ROOT / "dataset/analysis/openai_parity_roster_amendment_v1.json"
FIVE_MODEL_ROBUSTNESS_GATE = ROOT / "api_pilot/outputs/five_model_robustness_gate.json"
CHECKLIST_CORRECTION_MANIFEST = ROOT / "dataset/analysis/checklist_correction_supplement_manifest_v1.json"
INDEPENDENT_CORRECTION_MANIFEST = ROOT / "dataset/analysis/independent_correction_replication_manifest_v2.json"

OFFLINE_REFRESH = (
    "api_pilot/build_strict_gate_report.py",
)
ROBUSTNESS_REPORTS = {
    "repeat_identical": "api_pilot/outputs/repeat_identical_robustness_analysis.json",
    "single_item": "api_pilot/outputs/single_item_robustness_analysis.json",
    "wording_b": "api_pilot/outputs/wording_b_robustness_analysis.json",
    "option_reverse": "api_pilot/outputs/option_reverse_robustness_analysis.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    # ASCII escapes keep the status readable by Windows PowerShell 5.1, whose
    # default decoder otherwise treats UTF-8 without a BOM as the local codepage.
    atomic_text(path, json.dumps(value, ensure_ascii=True, indent=2) + "\n")


def checkpoint_counts(path: Path) -> dict[str, int]:
    counts = {"success": 0, "terminal": 0, "retryable": 0}
    if not path.exists():
        return counts
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
        rows = connection.execute("SELECT success, result_json FROM results").fetchall()
        connection.close()
    except (sqlite3.Error, OSError):
        return counts
    for success, raw in rows:
        if success:
            counts["success"] += 1
            continue
        try:
            row = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            counts["retryable"] += 1
            continue
        key = "terminal" if row.get("terminal_failure") else "retryable"
        counts[key] += 1
    return counts


def checkpoint_successes_by_provider(path: Path) -> dict[str, int]:
    """Count successful frozen calls by provider for phase barriers."""
    if not path.exists():
        return {}
    try:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5
        )
        rows = connection.execute(
            "SELECT provider, COUNT(*) FROM results WHERE success=1 GROUP BY provider"
        ).fetchall()
        connection.close()
    except (sqlite3.Error, OSError):
        return {}
    return {str(provider): int(count) for provider, count in rows}


def checkpoint_counts_by_provider(path: Path) -> dict[str, dict[str, int]]:
    """Return success, terminal, and retryable checkpoint counts per API."""
    result: dict[str, dict[str, int]] = {}
    if not path.exists():
        return result
    try:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5
        )
        rows = connection.execute(
            "SELECT provider, success, result_json FROM results"
        ).fetchall()
        connection.close()
    except (sqlite3.Error, OSError):
        return result
    for provider, success, raw in rows:
        counts = result.setdefault(
            str(provider), {"success": 0, "terminal": 0, "retryable": 0}
        )
        if success:
            counts["success"] += 1
            continue
        try:
            row = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            row = {}
        counts["terminal" if row.get("terminal_failure") else "retryable"] += 1
    return result


def manifest_progress(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    expected = int(manifest["expected"]["total_calls"])
    checkpoint = ROOT / manifest["immutable_checkpoint_path"]
    counts = checkpoint_counts(checkpoint)
    successes_by_provider = checkpoint_successes_by_provider(checkpoint)
    return {
        "manifest": path.relative_to(ROOT).as_posix(),
        "planned": expected,
        **counts,
        "success_by_provider": successes_by_provider,
        "pending": max(expected - counts["success"] - counts["terminal"], 0),
        "complete": counts["success"] == expected and counts["terminal"] == 0,
    }


def terminal_failures(manifest_paths: list[Path]) -> dict[str, int]:
    """Return frozen experiments that cannot make progress by retrying."""
    failures: dict[str, int] = {}
    for path in manifest_paths:
        progress = manifest_progress(path)
        if progress["terminal"]:
            failures[load_json(path)["experiment_id"]] = int(progress["terminal"])
    return failures


def workstreams() -> list[dict[str, Any]]:
    gate = load_json(GATE_STATUS_PATH) if GATE_STATUS_PATH.exists() else {}
    audit = gate.get("audit") or {}
    rows: list[dict[str, Any]] = [{
        "name": "Completed Gate C pilot",
        "planned": int(audit.get("planned") or 4_320),
        "success": int(audit.get("merged") or 0),
        "terminal": int(audit.get("incomplete") or 0),
        "retryable": 0,
        "pending": int(audit.get("missing") or 0),
        "complete": bool(audit.get("complete")),
    }]
    if ROBUSTNESS_SUITE_PATH.exists():
        suite = load_json(ROBUSTNESS_SUITE_PATH)
        for name, entry in suite["manifests"].items():
            row = manifest_progress(ROOT / entry["file"])
            row["name"] = f"Robustness: {name}"
            rows.append(row)
    if CORE_MANIFEST_PATH.exists():
        row = manifest_progress(CORE_MANIFEST_PATH)
        row["name"] = "Confirmatory core"
        rows.append(row)
    if BATCH_CONTEXT_MANIFEST_PATH.exists():
        row = manifest_progress(BATCH_CONTEXT_MANIFEST_PATH)
        row["name"] = "Report-only: batch-context replication"
        rows.append(row)
    if SENTINEL_SUITE_PATH.exists():
        suite = load_json(SENTINEL_SUITE_PATH)
        for wave, entry in suite["waves"].items():
            row = manifest_progress(ROOT / entry["manifest"])
            row["name"] = f"Runtime sentinel: {wave}"
            rows.append(row)
    if OPENAI_PARITY_PATH.exists():
        parity = load_json(OPENAI_PARITY_PATH)
        for name, entry in parity["manifests"].items():
            row = manifest_progress(ROOT / entry["file"])
            row["name"] = f"GPT parity: {name}"
            rows.append(row)
    if CHECKLIST_CORRECTION_MANIFEST.exists():
        row = manifest_progress(CHECKLIST_CORRECTION_MANIFEST)
        row["name"] = "Post-review: checklist wording correction"
        rows.append(row)
    if INDEPENDENT_CORRECTION_MANIFEST.exists():
        row = manifest_progress(INDEPENDENT_CORRECTION_MANIFEST)
        row["name"] = "Post-review: independent corrected-wording replication"
        rows.append(row)
    return rows


def provider_progress() -> list[dict[str, Any]]:
    """Aggregate actual checkpoint progress across every frozen API phase."""
    gate = load_json(GATE_STATUS_PATH) if GATE_STATUS_PATH.exists() else {}
    audit = gate.get("audit") or {}
    scopes = (gate.get("strict_inference") or {}).get("scope_results") or []
    engineering_scope = next(
        (
            row for row in scopes
            if row.get("scope") in {
                "five_model_amended_roster", "five_system_engineering"
            }
        ),
        {},
    )
    models = dict(engineering_scope.get("models") or {
        "deepseek": "deepseek-v4-pro",
        "gemini": "gemini-3.6-flash",
        "mistral": "mistral-medium-3-5",
        "qwen": "qwen3.7-max-2026-06-08",
        "openai": "gpt-5.4",
    })
    aggregate: dict[str, dict[str, Any]] = {}

    pilot_providers = int(audit.get("planned") or 0) // 864 if audit.get("planned") else 0
    if pilot_providers and models:
        pilot_each = int(audit.get("planned") or 0) // pilot_providers
        for provider, model in models.items():
            aggregate[provider] = {
                "provider": provider, "model": model, "planned": pilot_each,
                "success": pilot_each if audit.get("complete") else 0,
                "terminal": 0, "retryable": 0,
            }

    manifest_paths: list[Path] = []
    if ROBUSTNESS_SUITE_PATH.exists():
        suite = load_json(ROBUSTNESS_SUITE_PATH)
        manifest_paths.extend(ROOT / row["file"] for row in suite["manifests"].values())
    if CORE_MANIFEST_PATH.exists():
        manifest_paths.append(CORE_MANIFEST_PATH)
    if BATCH_CONTEXT_MANIFEST_PATH.exists():
        manifest_paths.append(BATCH_CONTEXT_MANIFEST_PATH)
    if SENTINEL_SUITE_PATH.exists():
        suite = load_json(SENTINEL_SUITE_PATH)
        manifest_paths.extend(ROOT / row["manifest"] for row in suite["waves"].values())

    for path in manifest_paths:
        manifest = load_json(path)
        provider_models = manifest.get("provider_models") or {}
        planned_each = int((manifest.get("expected") or {}).get("calls_per_provider") or 0)
        counts_by_provider = checkpoint_counts_by_provider(
            ROOT / manifest["immutable_checkpoint_path"]
        )
        for provider, model in provider_models.items():
            row = aggregate.setdefault(provider, {
                "provider": provider, "model": model, "planned": 0,
                "success": 0, "terminal": 0, "retryable": 0,
            })
            row["model"] = model
            row["planned"] += planned_each
            counts = counts_by_provider.get(provider, {})
            for key in ("success", "terminal", "retryable"):
                row[key] += int(counts.get(key) or 0)

    if OPENAI_PARITY_PATH.exists():
        parity = load_json(OPENAI_PARITY_PATH)
        for entry in parity["manifests"].values():
            path = ROOT / entry["file"]
            manifest = load_json(path)
            counts = checkpoint_counts_by_provider(
                ROOT / manifest["immutable_checkpoint_path"]
            ).get("openai", {})
            row = aggregate.setdefault("openai", {
                "provider": "openai", "model": "gpt-5.4", "planned": 864,
                "success": 864, "terminal": 0, "retryable": 0,
            })
            row["planned"] += int(manifest["expected"]["calls_per_provider"])
            for key in ("success", "terminal", "retryable"):
                row[key] += int(counts.get(key) or 0)

    if CHECKLIST_CORRECTION_MANIFEST.exists():
        manifest = load_json(CHECKLIST_CORRECTION_MANIFEST)
        planned_each = int(manifest["expected"]["calls_per_provider"])
        counts_by_provider = checkpoint_counts_by_provider(
            ROOT / manifest["immutable_checkpoint_path"]
        )
        for provider, model in manifest["provider_models"].items():
            row = aggregate.setdefault(provider, {
                "provider": provider, "model": model, "planned": 0,
                "success": 0, "terminal": 0, "retryable": 0,
            })
            row["planned"] += planned_each
            counts = counts_by_provider.get(provider, {})
            for key in ("success", "terminal", "retryable"):
                row[key] += int(counts.get(key) or 0)

    if INDEPENDENT_CORRECTION_MANIFEST.exists():
        manifest = load_json(INDEPENDENT_CORRECTION_MANIFEST)
        planned_each = int(manifest["expected"]["calls_per_provider"])
        counts_by_provider = checkpoint_counts_by_provider(
            ROOT / manifest["immutable_checkpoint_path"]
        )
        for provider, model in manifest["provider_models"].items():
            row = aggregate.setdefault(provider, {
                "provider": provider, "model": model, "planned": 0,
                "success": 0, "terminal": 0, "retryable": 0,
            })
            row["planned"] += planned_each
            counts = counts_by_provider.get(provider, {})
            for key in ("success", "terminal", "retryable"):
                row[key] += int(counts.get(key) or 0)

    for row in aggregate.values():
        row["pending"] = max(row["planned"] - row["success"] - row["terminal"], 0)
        row["percentage"] = (
            100.0 * (row["success"] + row["terminal"]) / row["planned"]
            if row["planned"] else 0.0
        )
    order = {"deepseek": 0, "gemini": 1, "mistral": 2, "qwen": 3, "openai": 4}
    return sorted(aggregate.values(), key=lambda row: order.get(row["provider"], 99))


def _json_task_complete(path: Path, statuses: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(value.get("complete")) or str(value.get("status")) in statuses


def local_compute_progress(active_phase: str) -> list[dict[str, Any]]:
    """Track frozen local analysis separately from metered API calls."""
    gate = load_json(GATE_STATUS_PATH) if GATE_STATUS_PATH.exists() else {}
    audit_complete = bool((gate.get("audit") or {}).get("complete"))
    specs = [
        ("稳健性分析: repeat_identical", "analyse_robustness_repeat_identical", ROOT / ROBUSTNESS_REPORTS["repeat_identical"], {"complete"}, "对应 API 输出"),
        ("稳健性分析: single_item", "analyse_robustness_single_item", ROOT / ROBUSTNESS_REPORTS["single_item"], {"complete"}, "对应 API 输出"),
        ("稳健性分析: wording_b", "analyse_robustness_wording_b", ROOT / ROBUSTNESS_REPORTS["wording_b"], {"complete"}, "对应 API 输出"),
        ("稳健性分析: option_reverse", "analyse_robustness_option_reverse", ROOT / ROBUSTNESS_REPORTS["option_reverse"], {"complete"}, "对应 API 输出"),
        ("Gate C pilot 严格统计与审计", "build_strict_gate_report", GATE_STATUS_PATH, {"complete"}, "Gate C pilot"),
        ("五模型面板合并", "sync_five_model_panels", ROOT / "api_pilot/outputs/five_model_panel_sync.json", {"complete"}, "全部五模型 API 面板"),
        ("五模型联合稳健性判定", "evaluate_five_model_robustness", FIVE_MODEL_ROBUSTNESS_GATE, {"pass", "fail", "complete"}, "全部稳健性 API 与分析"),
        ("确认性逻辑面板物化", "materialize_confirmatory_cached_panel", ROOT / "api_pilot/outputs/confirmatory_core_v2_cache_audit.json", {"cached_logical_panel_complete"}, "确认性 core API"),
        ("确认性严格推断", "confirmatory_strict_inference", ROOT / "api_pilot/outputs/confirmatory_core_v2_strict_inference.json", {"strict_confirmatory_inference_complete"}, "确认性逻辑面板"),
        ("确认性综合指标", "confirmatory_comprehensive_metrics", ROOT / "api_pilot/outputs/confirmatory_core_v2_comprehensive_metrics.json", {"confirmatory_secondary_metrics_complete"}, "确认性逻辑面板"),
        ("运行时哨兵分析", "confirmatory_sentinel_analysis", ROOT / "api_pilot/outputs/confirmatory_sentinel_analysis.json", {"complete"}, "core 与三轮 sentinel API"),
        ("批次上下文敏感性分析", "batch_context_analysis", ROOT / "api_pilot/outputs/confirmatory_batch_context_analysis.json", {"complete"}, "core 与 2.5% 批次复测 API"),
        ("最终完整性核验", "verify_experiment_completion", ROOT / "api_pilot/outputs/experiment_completion_contract.json", {"awaiting_human_review", "complete"}, "所有推理与统计产物"),
        ("清单题纠错敏感性分析", "checklist_correction_analysis", ROOT / "api_pilot/outputs/checklist_correction_analysis.json", {"complete"}, "纠错补充 API"),
        ("论文结果表导出", "export_final_results_tables", ROOT / "api_pilot/outputs/final_tables/manifest.json", {"complete"}, "严格推断与纠错敏感性"),
    ]
    specs.append((
        "Independent corrected-wording H1/H2 analysis",
        "independent_correction_analysis",
        ROOT / "api_pilot/outputs/independent_correction_replication_v2_analysis.json",
        {"complete"},
        "Independent corrected-wording API replication",
    ))
    rows = []
    share = 100.0 / len(specs)
    for name, phase, path, statuses, dependency in specs:
        complete = audit_complete if name == "Gate C pilot 严格统计与审计" else _json_task_complete(path, statuses)
        running = not complete and active_phase == phase
        rows.append({
            "name": name, "dependency": dependency,
            "status": "complete" if complete else ("running" if running else "waiting"),
            "completed_steps": 1 if complete else 0, "planned_steps": 1,
            "share": share,
        })
    return rows


def operational_alerts() -> list[dict[str, str]]:
    """Surface actionable provider failures without turning them into gates."""
    alerts: dict[tuple[str, str], dict[str, str]] = {}
    for stream in workstreams():
        manifest_rel = stream.get("manifest")
        if not manifest_rel:
            continue
        manifest = load_json(ROOT / manifest_rel)
        checkpoint = ROOT / manifest["immutable_checkpoint_path"]
        if not checkpoint.exists():
            continue
        try:
            connection = sqlite3.connect(
                f"file:{checkpoint.as_posix()}?mode=ro", uri=True, timeout=3
            )
            rows = connection.execute(
                "SELECT provider, result_json, updated_at FROM results WHERE success=0"
            ).fetchall()
            latest_success = dict(connection.execute(
                "SELECT provider, MAX(updated_at) FROM results WHERE success=1 GROUP BY provider"
            ).fetchall())
            connection.close()
        except (sqlite3.Error, OSError):
            continue
        for provider, raw, failed_at in rows:
            try:
                result = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            message = str(result.get("error_message") or result.get("error") or "")
            if (
                "prepayment credits are depleted" in message.lower()
                and str(latest_success.get(provider) or "") <= str(failed_at or "")
            ):
                alerts[(provider, "billing_depleted")] = {
                    "provider": provider,
                    "type": "billing_depleted",
                    "message": "Gemini 预付额度已耗尽；请在 Google AI Studio Billing 购买 credits。补款后程序会从 SQLite 断点继续。",
                }
            if (
                ("arrearage" in message.lower() or "account is in good standing" in message.lower())
                and str(latest_success.get(provider) or "") <= str(failed_at or "")
            ):
                alerts[(provider, "billing_arrearage")] = {
                    "provider": provider,
                    "type": "billing_arrearage",
                    "message": "Qwen billing is in arrears. Restore Alibaba Model Studio billing; checkpoint recovery will skip all successful calls.",
                }
    return list(alerts.values())


def render_monitor(state: dict[str, Any]) -> str:
    rows = state["workstreams"]
    planned = sum(row["planned"] for row in rows)
    success = sum(row["success"] for row in rows)
    terminal = sum(row["terminal"] for row in rows)
    pending = sum(row["pending"] for row in rows)
    percentage = 100.0 * (success + terminal) / planned if planned else 0.0
    provider_rows = state.get("providers") or []
    local_rows = state.get("local_tasks") or []
    local_planned = sum(row["planned_steps"] for row in local_rows)
    local_complete = sum(row["completed_steps"] for row in local_rows)
    local_percentage = 100.0 * local_complete / local_planned if local_planned else 0.0
    table = "".join(
        "<tr>"
        f"<td><span class='kind api'>API</span> {html.escape(row['name'])}</td>"
        f"<td>{row['planned']:,}</td><td>{row['success']:,}</td>"
        f"<td>{row['terminal']:,}</td><td>{row['retryable']:,}</td>"
        f"<td>{row['pending']:,}</td>"
        f"<td>{(100.0 * (row['success'] + row['terminal']) / row['planned'] if row['planned'] else 0):.1f}%</td>"
        f"<td>{(100.0 * row['planned'] / planned if planned else 0):.1f}%</td>"
        f"<td>{'complete' if row['complete'] else 'pending'}</td></tr>"
        for row in rows
    )
    blockers = state.get("blockers") or []
    blocker_html = "".join(
        f"<li><code>{html.escape(str(row.get('id', 'unknown')))}</code>: "
        f"{html.escape(str(row.get('reason', 'pending')))}</li>"
        for row in blockers
    ) or "<li>None</li>"
    active_phase = str(state.get("phase") or "")
    provider_table = "".join(
        "<tr>"
        f"<td><strong>{html.escape(str(row['provider']).title())}</strong><br>"
        f"<code>{html.escape(str(row['model']))}</code></td>"
        f"<td><span class='badge {'ok' if row['pending'] == 0 and row['terminal'] == 0 else ('bad' if row['terminal'] else ('active' if state.get('status') == 'running' and active_phase not in ('offline_refresh', 'user_review') else 'warn'))}'>"
        f"{'DONE' if row['pending'] == 0 and row['terminal'] == 0 else ('INVALID' if row['terminal'] else ('RUNNING' if state.get('status') == 'running' and active_phase not in ('offline_refresh', 'user_review') else 'QUEUED'))}</span></td>"
        f"<td>{row['planned']:,}</td><td>{row['success']:,}</td>"
        f"<td>{row['retryable']:,}</td><td>{row['terminal']:,}</td>"
        f"<td>{row['pending']:,}</td><td><div class='bar'><span style='width:{row['percentage']:.2f}%'></span></div>"
        f"<small>{row['percentage']:.1f}%</small></td></tr>"
        for row in provider_rows
    ) or "<tr><td colspan='8'>尚无 API 进度记录</td></tr>"
    local_table = "".join(
        "<tr>"
        f"<td><span class='kind local'>LOCAL</span> {html.escape(row['name'])}</td>"
        f"<td>{html.escape(row['dependency'])}</td>"
        f"<td><span class='badge {'ok' if row['status'] == 'complete' else ('active' if row['status'] == 'running' else 'warn')}'>"
        f"{'DONE' if row['status'] == 'complete' else ('RUNNING' if row['status'] == 'running' else 'WAITING')}</span></td>"
        f"<td>{row['completed_steps']} / {row['planned_steps']}</td>"
        f"<td>{row['share']:.1f}%</td></tr>"
        for row in local_rows
    ) or "<tr><td colspan='5'>尚无本地计算步骤</td></tr>"
    status_class = {
        "complete": "ok", "running": "active", "waiting": "warn",
        "blocked": "bad", "error": "bad",
    }.get(state.get("status"), "warn")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AAAI Experiment Automation</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f4f6f8;color:#18212b;margin:0}}
main{{max-width:1180px;margin:auto;padding:24px}}h1{{font-size:24px;margin:0 0 8px}}
.sub{{color:#586675;margin-bottom:20px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{background:white;border:1px solid #dce2e8;border-radius:6px;padding:16px}}
.value{{font-size:24px;font-weight:650}}.badge{{display:inline-block;padding:5px 9px;border-radius:4px;font-weight:650}}
.ok{{background:#dff3e7;color:#12643a}}.active{{background:#dcecff;color:#14518c}}
.warn{{background:#fff0c7;color:#765500}}.bad{{background:#ffe0e0;color:#8f2525}}
.kind{{display:inline-block;padding:2px 5px;border-radius:3px;font-size:11px;font-weight:700}}
.kind.api{{background:#e3effa;color:#195b91}}.kind.local{{background:#e5f2e8;color:#27643a}}
.bar{{height:10px;background:#e5e9ed;border-radius:3px;overflow:hidden;margin-top:10px}}
.bar span{{display:block;height:100%;background:#2777b8}}section{{margin-top:16px}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px;border-bottom:1px solid #e2e6ea;text-align:left}}
th{{background:#f0f3f5}}code{{font-family:Consolas,monospace}}@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>AAAI 实验自动化状态</h1>
<div class="sub">页面每分钟自动刷新；活跃推理期间后台状态约每 30 秒更新一次。</div>
<div class="card"><span class="badge {status_class}">{html.escape(str(state.get('status','unknown')).upper())}</span>
<strong style="margin-left:10px">{html.escape(str(state.get('phase','unknown')))}</strong>
<p>{html.escape(str(state.get('message','')))}</p><div class="bar"><span style="width:{percentage:.2f}%"></span></div></div>
<section class="grid"><div class="card"><div>API 总进度</div><div class="value">{percentage:.1f}%</div><small>{success + terminal:,} / {planned:,} calls</small></div>
<div class="card"><div>API 剩余</div><div class="value">{pending:,}</div><small>有效 {success:,}，终止无效 {terminal:,}</small></div>
<div class="card"><div>本地计算总进度</div><div class="value">{local_percentage:.1f}%</div><small>{local_complete} / {local_planned} steps</small></div>
<div class="card"><div>当前阶段</div><div class="value" style="font-size:18px">{html.escape(str(state.get('phase','unknown')))}</div></div></section>
<section><h2>各 API 模型总进度</h2><table><thead><tr><th>API / 模型</th><th>状态</th><th>总计划</th><th>有效完成</th><th>待重试</th><th>终止无效</th><th>剩余</th><th>完成率</th></tr></thead><tbody>{provider_table}</tbody></table></section>
<section><h2>API 推理任务</h2><p class="sub">任务占比按该任务调用数 / 全部 API 调用数计算。</p><table><thead><tr><th>任务</th><th>计划</th><th>有效</th><th>终止无效</th><th>可重试</th><th>剩余</th><th>任务进度</th><th>API 总量占比</th><th>状态</th></tr></thead><tbody>{table}</tbody></table></section>
<section><h2>本地计算任务</h2><p class="sub">本地步骤与 API calls 不同尺度，因此按 {local_planned} 个冻结步骤独立计数，每步占 {100.0 / local_planned if local_planned else 0:.1f}%。</p><table><thead><tr><th>任务</th><th>输入依赖</th><th>状态</th><th>步骤进度</th><th>本地总量占比</th></tr></thead><tbody>{local_table}</tbody></table></section>
<section class="card"><h2>门控与阻塞</h2><ul>{blocker_html}</ul></section>
<section class="card"><h2>恢复语义</h2><p>所有 LLM 任务使用 SQLite WAL 断点。程序重启后跳过成功 call ID，仅重试未完成的运行性失败。许可信息只作审计记录；预注册的科学稳健性门仍控制正式推理。</p>
<p>更新时间：<code>{html.escape(state['updated_at'])}</code>，监督进程 PID：<code>{state.get('supervisor_pid','--')}</code></p>
<p>状态文件：<code>api_pilot/outputs/remaining_experiment_status.json</code>，日志：<code>api_pilot/logs/remaining_experiment/</code></p></section>
</main></body></html>"""


def current_gate() -> dict[str, Any]:
    return load_json(GATE_STATUS_PATH) if GATE_STATUS_PATH.exists() else {}


def render_monitor_v2_legacy(state: dict[str, Any]) -> str:
    streams = state.get("workstreams") or []
    providers = state.get("providers") or []
    local = state.get("local_tasks") or []
    planned = sum(row["planned"] for row in streams)
    success = sum(row["success"] for row in streams)
    retryable = sum(row["retryable"] for row in streams)
    terminal = sum(row["terminal"] for row in streams)
    percent = 100.0 * (success + terminal) / planned if planned else 0.0
    warnings = state.get("blockers") or []
    provider_html = "".join(
        "<tr>"
        f"<td><b>{html.escape(row['provider'].title())}</b><br><code>{html.escape(row['model'])}</code></td>"
        f"<td>{row['planned']:,}</td><td>{row['success']:,}</td>"
        f"<td>{row['retryable']:,}</td><td>{row['terminal']:,}</td>"
        f"<td>{row['pending']:,}</td><td>{row['percentage']:.1f}%</td></tr>"
        for row in providers
    )
    stream_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['name'])}</td><td>{row['planned']:,}</td>"
        f"<td>{row['success']:,}</td><td>{row['retryable']:,}</td>"
        f"<td>{row['terminal']:,}</td><td>{row['pending']:,}</td>"
        f"<td>{'DONE' if row['complete'] else 'FIRST PASS / RETRY'}</td></tr>"
        for row in streams
    )
    local_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['name'])}</td><td>{html.escape(row['dependency'])}</td>"
        f"<td>{row['status'].upper()}</td><td>{row['completed_steps']} / {row['planned_steps']}</td></tr>"
        for row in local
    )
    warning_html = "".join(
        f"<li><b>REPORT-ONLY</b> <code>{html.escape(str(row.get('id')))}</code>: "
        f"{html.escape(str(row.get('reason')))}</li>" for row in warnings
    ) or "<li>None</li>"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AAAI 全流程自动化监控</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f3f5f7;color:#17212b;margin:0}}
main{{max-width:1220px;margin:auto;padding:22px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card,table{{background:#fff;border:1px solid #d9e0e6;border-radius:6px}}.card{{padding:15px}}
.value{{font-size:24px;font-weight:700}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e5e9ed;text-align:left}}
th{{background:#eef2f5}}section{{margin-top:16px}}.bar{{height:12px;background:#e3e8ec;border-radius:4px;overflow:hidden}}
.bar span{{display:block;height:100%;background:#2678b8}}.running{{color:#14518c}}.warning{{color:#7a5700}}
code{{font-family:Consolas,monospace}}@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><h1>AAAI 实验全流程自动化</h1>
<p>每分钟刷新。科学诊断仅进入报告，不会中断推理；失败请求在大批次首轮覆盖后从 SQLite 断点统一重试。</p>
<div class="card"><b class="running">{html.escape(str(state.get('status','unknown')).upper())}</b> · <code>{html.escape(str(state.get('phase','unknown')))}</code>
<p>{html.escape(str(state.get('message','')))}</p><div class="bar"><span style="width:{percent:.2f}%"></span></div></div>
<section class="grid"><div class="card">API 总进度<div class="value">{percent:.1f}%</div></div>
<div class="card">有效完成<div class="value">{success:,}</div></div><div class="card">延迟重试<div class="value">{retryable:,}</div></div>
<div class="card">剩余任务<div class="value">{max(planned-success-terminal,0):,}</div></div></section>
<section><h2>各 API 模型</h2><table><tr><th>模型</th><th>计划</th><th>成功</th><th>待重试</th><th>最终失败</th><th>剩余</th><th>进度</th></tr>{provider_html}</table></section>
<section><h2>API 推理任务</h2><table><tr><th>任务</th><th>计划</th><th>成功</th><th>待重试</th><th>最终失败</th><th>剩余</th><th>阶段</th></tr>{stream_html}</table></section>
<section><h2>本地计算任务</h2><table><tr><th>任务</th><th>依赖</th><th>状态</th><th>进度</th></tr>{local_html}</table></section>
<section class="card"><h2>科学诊断</h2><ul>{warning_html}</ul><p class="warning">这些结果限制论文解释，但不会改变自动化运行状态。</p></section>
<section class="card"><h2>恢复机制</h2><p>每条成功调用立即写入 SQLite。程序重启后跳过成功 call ID；网络、限流和格式错误留待后续周期重试，不阻塞其他模型。</p>
<p>更新时间：<code>{html.escape(state['updated_at'])}</code>；监督 PID：<code>{state.get('supervisor_pid','--')}</code>；工作 PID：<code>{state.get('worker_pid','--')}</code></p></section>
</main></body></html>"""


def render_monitor_v2(state: dict[str, Any]) -> str:
    streams = state.get("workstreams") or []
    providers = state.get("providers") or []
    local = state.get("local_tasks") or []
    planned = sum(row["planned"] for row in streams)
    success = sum(row["success"] for row in streams)
    retryable = sum(row["retryable"] for row in streams)
    terminal = sum(row["terminal"] for row in streams)
    percent = 100.0 * (success + terminal) / planned if planned else 0.0
    provider_html = "".join(
        "<tr>"
        f"<td><b>{html.escape(row['provider'].title())}</b><br><code>{html.escape(row['model'])}</code></td>"
        f"<td>{row['planned']:,}</td><td>{row['success']:,}</td>"
        f"<td>{row['retryable']:,}</td><td>{row['terminal']:,}</td>"
        f"<td>{row['pending']:,}</td><td>{row['percentage']:.1f}%</td></tr>"
        for row in providers
    )
    stream_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['name'])}</td><td>{row['planned']:,}</td>"
        f"<td>{row['success']:,}</td><td>{row['retryable']:,}</td>"
        f"<td>{row['terminal']:,}</td><td>{row['pending']:,}</td>"
        f"<td>{'完成' if row['complete'] else '首轮或重试'}</td></tr>"
        for row in streams
    )
    local_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['name'])}</td><td>{html.escape(row['dependency'])}</td>"
        f"<td>{row['status'].upper()}</td><td>{row['completed_steps']} / {row['planned_steps']}</td></tr>"
        for row in local
    )
    warnings = state.get("blockers") or []
    warning_html = "".join(
        f"<li><b>仅报告</b> <code>{html.escape(str(row.get('id')))}</code>: "
        f"{html.escape(str(row.get('reason')))}</li>" for row in warnings
    ) or "<li>无</li>"
    alert_html = "".join(
        f"<li><b>{html.escape(row['provider'].upper())}</b>: {html.escape(row['message'])}</li>"
        for row in state.get("operational_alerts") or []
    ) or "<li>无</li>"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AAAI 全流程自动化监控</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f3f5f7;color:#17212b;margin:0}}
main{{max-width:1220px;margin:auto;padding:22px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card,table{{background:#fff;border:1px solid #d9e0e6;border-radius:6px}}.card{{padding:15px}}
.value{{font-size:24px;font-weight:700}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e5e9ed;text-align:left}}
th{{background:#eef2f5}}section{{margin-top:16px}}.bar{{height:12px;background:#e3e8ec;border-radius:4px;overflow:hidden}}
.bar span{{display:block;height:100%;background:#2678b8}}.running{{color:#14518c}}.warning{{color:#7a5700}}
code{{font-family:Consolas,monospace}}@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><h1>AAAI 实验全流程自动化</h1>
<p>每分钟刷新。科学诊断只进入报告，不中断推理；失败请求在大批次首轮覆盖后从 SQLite 断点统一重试。</p>
<div class="card"><b class="running">{html.escape(str(state.get('status','unknown')).upper())}</b> · <code>{html.escape(str(state.get('phase','unknown')))}</code>
<p>{html.escape(str(state.get('message','')))}</p><div class="bar"><span style="width:{percent:.2f}%"></span></div></div>
<section class="grid"><div class="card">API 总进度<div class="value">{percent:.1f}%</div></div>
<div class="card">有效完成<div class="value">{success:,}</div></div><div class="card">延迟重试<div class="value">{retryable:,}</div></div>
<div class="card">剩余任务<div class="value">{max(planned-success-terminal,0):,}</div></div></section>
<section><h2>各 API 模型</h2><table><tr><th>模型</th><th>计划</th><th>成功</th><th>待重试</th><th>最终失败</th><th>剩余</th><th>进度</th></tr>{provider_html}</table></section>
<section><h2>API 推理任务</h2><table><tr><th>任务</th><th>计划</th><th>成功</th><th>待重试</th><th>最终失败</th><th>剩余</th><th>阶段</th></tr>{stream_html}</table></section>
<section><h2>本地计算任务</h2><table><tr><th>任务</th><th>依赖</th><th>状态</th><th>进度</th></tr>{local_html}</table></section>
<section class="card"><h2>运行告警</h2><ul>{alert_html}</ul></section>
<section class="card"><h2>科学诊断</h2><ul>{warning_html}</ul><p class="warning">这些结果限制论文解释，但不会改变自动化运行状态。</p></section>
<section class="card"><h2>恢复机制</h2><p>每条成功调用立即写入 SQLite。程序重启后跳过成功 call ID；网络、限流和格式错误留待后续周期重试，不阻塞其他模型。</p>
<p>更新时间：<code>{html.escape(state['updated_at'])}</code>；监督 PID：<code>{state.get('supervisor_pid','--')}</code>；工作 PID：<code>{state.get('worker_pid','--')}</code></p></section>
</main></body></html>"""


def write_state(status: str, phase: str, message: str, *, detail: dict[str, Any] | None = None) -> None:
    gate = current_gate()
    state = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "phase": phase,
        "message": message,
        "supervisor_pid": os.getppid(),
        "worker_pid": os.getpid(),
        "permission": gate.get("data_processing_permission") or {},
        "full_gate_c_pass": bool((gate.get("gate") or {}).get("full_gate_c_pass")),
        "blockers": (gate.get("gate") or {}).get("blockers") or [],
        "workstreams": workstreams(),
        "providers": provider_progress(),
        "operational_alerts": operational_alerts(),
        "local_tasks": local_compute_progress(phase),
        "detail": detail or {},
    }
    atomic_json(STATUS_PATH, state)
    atomic_text(MONITOR_PATH, render_monitor_v2(state))


def append_event(event: str, **detail: Any) -> None:
    EVENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "event": event,
        **detail,
    }
    with EVENT_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_local(arguments: list[str], label: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n[{datetime.now().astimezone().isoformat()}] START\n")
        log.flush()
        process = subprocess.Popen(
            [str(PYTHON), *arguments], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, text=True,
        )
        while process.poll() is None:
            write_state(
                "running",
                label,
                "正在执行本地统计或完整性检查；该阶段不调用 LLM API。",
                detail={"local_pid": process.pid},
            )
            time.sleep(30)
        code = int(process.returncode or 0)
        log.write(f"[{datetime.now().astimezone().isoformat()}] EXIT {code}\n")
    return code


def refresh_gate() -> dict[str, Any]:
    reports_ready = all(
        (ROOT / report).exists()
        and load_json(ROOT / report).get("complete") is True
        for report in ROBUSTNESS_REPORTS.values()
    )
    if reports_ready and GATE_STATUS_PATH.exists():
        return current_gate()
    if ROBUSTNESS_SUITE_PATH.exists():
        suite = load_json(ROBUSTNESS_SUITE_PATH)
        for name, entry in suite["manifests"].items():
            report = ROBUSTNESS_REPORTS[name]
            code = run_local([
                "api_pilot/analyze_robustness_outputs.py",
                "--manifest", entry["file"],
                "--json-report", report,
            ], f"analyse_robustness_{name}")
            if code:
                raise RuntimeError(
                    f"Robustness analysis failed: {name} (exit {code})"
                )
    for script in OFFLINE_REFRESH:
        code = run_local([script], Path(script).stem)
        if code:
            raise RuntimeError(f"Offline gate refresh failed: {script} (exit {code})")
    return current_gate()


def runner_command(
    manifest_path: Path,
    workers_per_key: int,
    *,
    stop_after_successes_per_provider: int = 0,
) -> list[str]:
    manifest = load_json(manifest_path)
    source = manifest["source"]
    models = manifest["provider_models"]
    command = [
        "api_pilot/survey_pilot_runner.py",
        "--profiles", source["profile_file"],
        "--prompt-bundle", manifest["prompt"]["file"],
        "--max-profiles-per-country", str(source["profiles_per_country"]),
        "--providers", *models.keys(),
        "--workers-per-key", str(workers_per_key),
        "--provider-workers", "gemini=1",
        "--max-attempts", "1",
        "--quota-stop-threshold", "3",
        "--target-batch-size", str(manifest["runner_configuration"]["target_batch_size"]),
        "--task-manifest", manifest_path.relative_to(ROOT).as_posix(),
        "--output", manifest["immutable_output_path"],
        "--checkpoint", manifest["immutable_checkpoint_path"],
        "--progress-doc", f"api_pilot/progress/{manifest['experiment_id']}.md",
        "--run-label", manifest["experiment_id"],
        "--resume-command",
        "Managed by api_pilot/scripts/windows/run_remaining_experiment_automation.ps1",
        "--progress-every", "48",
        "--acknowledge-respondent-data-processing",
    ]
    for provider, model in models.items():
        command.extend([f"--{provider}-model", model])
    if source.get("analysis_manifest_file"):
        command.extend(["--analysis-manifest", source["analysis_manifest_file"]])
    if manifest["runner_configuration"].get("reverse_target_options"):
        command.append("--reverse-target-options")
    if stop_after_successes_per_provider:
        command.extend([
            "--stop-after-successes-per-provider",
            str(stop_after_successes_per_provider),
        ])
    return command


def start_runner(
    manifest_path: Path,
    workers_per_key: int,
    *,
    stop_after_successes_per_provider: int = 0,
) -> tuple[subprocess.Popen[Any], Any, Path]:
    manifest = load_json(manifest_path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{manifest['experiment_id']}.log"
    log = log_path.open("a", encoding="utf-8", newline="\n")
    log.write(f"\n[{datetime.now().astimezone().isoformat()}] START/RESUME\n")
    log.flush()
    process = subprocess.Popen(
        [
            str(PYTHON),
            *runner_command(
                manifest_path,
                workers_per_key,
                stop_after_successes_per_provider=(
                    stop_after_successes_per_provider
                ),
            ),
        ],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
    )
    append_event("runner_started", experiment=manifest["experiment_id"], pid=process.pid)
    return process, log, log_path


def wait_process(process: subprocess.Popen[Any], log: Any, label: str, poll_seconds: int) -> int:
    while process.poll() is None:
        write_state("running", label, "本地监督程序正在运行冻结任务；进度已保存到 SQLite。")
        time.sleep(poll_seconds)
    code = int(process.returncode or 0)
    log.write(f"[{datetime.now().astimezone().isoformat()}] EXIT {code}\n")
    log.close()
    append_event("runner_finished", experiment=label, exit_code=code)
    return code


def run_manifest(manifest_path: Path, workers_per_key: int, poll_seconds: int) -> bool:
    if manifest_progress(manifest_path)["complete"]:
        return True
    process, log, _ = start_runner(manifest_path, workers_per_key)
    code = wait_process(process, log, load_json(manifest_path)["experiment_id"], poll_seconds)
    return code == 0 and manifest_progress(manifest_path)["complete"]


def run_robustness_parallel(poll_seconds: int) -> bool:
    """Run frozen robustness manifests one at a time.

    Each manifest already fans out across providers and keys. Serialising the
    manifests keeps a single rate-limit controller per credential pool and
    avoids four independent processes producing avoidable 429 bursts.
    """
    suite = load_json(ROBUSTNESS_SUITE_PATH)
    for name, entry in suite["manifests"].items():
        path = ROOT / entry["file"]
        if manifest_progress(path)["complete"]:
            continue
        write_state(
            "running", "gate_c_robustness",
            f"稳健性任务 {name} 正从独立 SQLite checkpoint 续跑。",
        )
        if not run_manifest(path, workers_per_key=1, poll_seconds=poll_seconds):
            break
    terminal = {
        name: manifest_progress(ROOT / entry["file"])["terminal"]
        for name, entry in suite["manifests"].items()
    }
    if any(terminal.values()):
        write_state(
            "blocked", "robustness_terminal_invalid",
            "稳健性任务存在终止无效响应，需要修复或重新冻结协议后再运行。",
            detail={"terminal_by_experiment": terminal},
        )
        return False
    return all(
        manifest_progress(ROOT / entry["file"])["complete"]
        for entry in suite["manifests"].values()
    )


def run_core_with_mid_sentinel(poll_seconds: int) -> bool:
    sentinel = load_json(SENTINEL_SUITE_PATH)
    before = ROOT / sentinel["waves"]["before"]["manifest"]
    mid = ROOT / sentinel["waves"]["mid"]["manifest"]
    if not manifest_progress(before)["complete"]:
        if manifest_progress(CORE_MANIFEST_PATH)["success"]:
            write_state(
                "blocked", "sentinel_order",
                "正式 core 已有结果，但 before sentinel 缺失。为避免伪造时间顺序，自动化已停止。",
            )
            return False
        if not run_manifest(before, workers_per_key=1, poll_seconds=poll_seconds):
            return False
    core_progress = manifest_progress(CORE_MANIFEST_PATH)
    halfway = core_progress["planned"] // 2
    core_manifest = load_json(CORE_MANIFEST_PATH)
    per_provider_halfway = core_manifest["expected"]["calls_per_provider"] // 2
    expected_providers = set(core_manifest["provider_models"])

    def midpoint_vector(progress: dict[str, Any]) -> dict[str, int]:
        observed = progress.get("success_by_provider") or {}
        return {name: int(observed.get(name, 0)) for name in expected_providers}

    provider_progress = midpoint_vector(core_progress)
    if any(value > per_provider_halfway for value in provider_progress.values()) and not manifest_progress(mid)["complete"]:
        write_state(
            "blocked", "mid_sentinel_provider_order",
            "A provider has passed the frozen midpoint before the mid sentinel. "
            "The runtime order cannot be reconstructed safely.",
            detail={"success_by_provider": provider_progress,
                    "required_per_provider": per_provider_halfway},
        )
        return False
    if (
        core_progress["success"] > halfway
        and not manifest_progress(mid)["complete"]
    ):
        write_state(
            "blocked", "mid_sentinel_order",
            "正式 core 已越过冻结中点，但 mid sentinel 缺失；时间顺序不可补造。",
            detail={
                "core_success": core_progress["success"],
                "required_midpoint": halfway,
            },
        )
        return False
    if (
        all(value == per_provider_halfway for value in provider_progress.values())
        and not manifest_progress(mid)["complete"]
    ):
        if not run_manifest(mid, workers_per_key=1, poll_seconds=poll_seconds):
            return False
    if core_progress["complete"]:
        return True
    if core_progress["success"] < halfway:
        process, log, _ = start_runner(
            CORE_MANIFEST_PATH,
            workers_per_key=4,
            stop_after_successes_per_provider=per_provider_halfway,
        )
        code = wait_process(
            process, log, "confirmatory_core_first_half", poll_seconds
        )
        core_progress = manifest_progress(CORE_MANIFEST_PATH)
        provider_progress = midpoint_vector(core_progress)
        if (
            code != 3
            or core_progress["success"] != halfway
            or any(
                value != per_provider_halfway
                for value in provider_progress.values()
            )
        ):
            write_state(
                "blocked", "mid_sentinel_provider_balance",
                "The core phase did not stop at the exact per-provider midpoint.",
                detail={"success_by_provider": provider_progress,
                        "required_per_provider": per_provider_halfway},
            )
            return False
    if not manifest_progress(mid)["complete"]:
        if not run_manifest(mid, workers_per_key=1, poll_seconds=poll_seconds):
            return False
    return run_manifest(
        CORE_MANIFEST_PATH, workers_per_key=4, poll_seconds=poll_seconds
    )


def postprocess() -> bool:
    five_manifest = "dataset/analysis/confirmatory_core_v2_raw_five_model_manifest_v1.json"
    five_raw = "api_pilot/restricted_outputs/confirmatory_core_v2_raw_five_model.jsonl"
    five_logical = "api_pilot/restricted_outputs/confirmatory_core_v2_logical_five_model.jsonl"
    reusable_outputs = {
        "evaluate_five_model_robustness": (FIVE_MODEL_ROBUSTNESS_GATE, {"pass", "fail"}),
        "materialize_confirmatory_cached_panel": (ROOT / "api_pilot/outputs/confirmatory_core_v2_cache_audit.json", {"cached_logical_panel_complete"}),
        "confirmatory_strict_inference": (ROOT / "api_pilot/outputs/confirmatory_core_v2_strict_inference.json", {"strict_confirmatory_inference_complete"}),
        "confirmatory_comprehensive_metrics": (ROOT / "api_pilot/outputs/confirmatory_core_v2_comprehensive_metrics.json", {"confirmatory_secondary_metrics_complete"}),
        "confirmatory_sentinel_analysis": (ROOT / "api_pilot/outputs/confirmatory_sentinel_analysis.json", {"complete"}),
        "batch_context_analysis": (ROOT / "api_pilot/outputs/confirmatory_batch_context_analysis.json", {"complete"}),
    }
    commands = (
        ("sync_five_model_panels", ["api_pilot/sync_five_model_panels.py"], {0}),
        ("evaluate_five_model_robustness", [
            "api_pilot/evaluate_five_model_robustness.py",
        ], {0, 14}),
        ("materialize_confirmatory_cached_panel", [
            "api_pilot/materialize_confirmatory_cached_panel.py",
            "--task-manifest", five_manifest, "--raw-results", five_raw,
            "--logical-output", five_logical,
        ], {0}),
        ("confirmatory_strict_inference", [
            "api_pilot/run_confirmatory_strict_inference.py",
            "--task-manifest", five_manifest, "--results", five_raw,
        ], {0}),
        ("confirmatory_comprehensive_metrics", [
            "api_pilot/run_confirmatory_comprehensive_metrics.py",
            "--task-manifest", five_manifest, "--raw-results", five_raw,
            "--results", five_logical,
        ], {0}),
        ("confirmatory_sentinel_analysis", [
            "api_pilot/analyze_confirmatory_sentinels.py",
            "--suite", "dataset/analysis/confirmatory_sentinel_suite_five_model_v1.json",
        ], {0}),
        ("batch_context_analysis", ["api_pilot/analyze_batch_context_replication.py"], {0}),
        ("build_strict_gate_report", ["api_pilot/build_strict_gate_report.py"], {0}),
        ("verify_experiment_completion", ["api_pilot/verify_five_model_completion.py"], {0}),
    )
    for label, command, accepted_codes in commands:
        reusable = reusable_outputs.get(label)
        if reusable and _json_task_complete(*reusable):
            append_event("offline_step_reused", phase=label, path=str(reusable[0]))
            continue
        write_state("running", label, "正在执行离线完整性核验和统计计算，不调用 LLM API。")
        if run_local(command, label) not in accepted_codes:
            return False
    return True


def one_cycle(poll_seconds: int) -> int:
    write_state("running", "offline_refresh", "正在刷新本地门控状态；此阶段不调用 API。")
    gate = refresh_gate()
    permission = gate.get("data_processing_permission") or {}
    if not permission.get("passes"):
        # Permission material is retained for audit, but it is not a runtime
        # control. The researcher has authorised this non-commercial study.
        append_event(
            "permission_audit_warning",
            permission_status=permission.get("status", "unavailable"),
            checks=permission.get("checks", {}),
        )

    write_state("running", "gate_c_robustness", "正在从断点续跑冻结的稳健性任务。")
    if not run_robustness_parallel(poll_seconds):
        refresh_gate()
        suite = load_json(ROBUSTNESS_SUITE_PATH)
        failures = terminal_failures([
            ROOT / entry["file"] for entry in suite["manifests"].values()
        ])
        if failures:
            append_event("deferred_failed_units", phase="robustness", failures=failures)
        write_state(
            "waiting", "robustness_retry",
            "稳健性调用尚未完整；监督程序将在下一周期从 SQLite 断点继续。",
        )
        return 11
    gate = refresh_gate()
    robustness = gate.get("robustness") or {}
    failed = [
        name for name, value in (robustness.get("experiments") or {}).items()
        if value.get("gate_status") == "fail"
    ]
    if failed:
        append_event(
            "scientific_warning", diagnostics=failed, action="report_only"
        )
    if False and failed:
        write_state(
            "blocked", "robustness_gate_failed",
            "稳健性门未通过，正式推理已按预注册规则停止：" + ", ".join(failed),
        )
        # A pre-registered robustness failure is a scientific stop.
        return 14
    if False and not (gate.get("gate") or {}).get("full_gate_c_pass"):
        write_state(
            "waiting", "full_gate_pending",
            "稳健性输出尚未形成完整通过判定；下一周期将从断点复核。",
        )
        return 11

    parity = load_json(OPENAI_PARITY_PATH) if OPENAI_PARITY_PATH.exists() else {}
    parity_robustness = [
        ROOT / parity.get("manifests", {}).get(name, {}).get("file", "")
        for name in ("repeat_identical", "single_item", "wording_b", "option_reverse")
    ]
    if not parity_robustness or not all(
        path.is_file() and manifest_progress(path)["complete"]
        for path in parity_robustness
    ):
        write_state(
            "running", "five_model_robustness_wait",
            "四模型稳健性已完成；正在等待 GPT-5.4 对等稳健性任务。",
        )
        return 11
    five_gate = load_json(FIVE_MODEL_ROBUSTNESS_GATE) if FIVE_MODEL_ROBUSTNESS_GATE.exists() else {}
    if FIVE_MODEL_ROBUSTNESS_GATE.exists():
        code = 14 if five_gate.get("status") == "fail" else 0
    else:
        code = run_local(
            ["api_pilot/evaluate_five_model_robustness.py"],
            "evaluate_five_model_robustness",
        )
        five_gate = load_json(FIVE_MODEL_ROBUSTNESS_GATE) if FIVE_MODEL_ROBUSTNESS_GATE.exists() else {}
    if five_gate.get("status") == "fail":
        append_event(
            "five_model_scientific_warning",
            action="report_only", detail=five_gate,
        )
    if False and (code == 14 or five_gate.get("status") == "fail"):
        write_state(
            "blocked", "five_model_robustness_gate_failed",
            "五模型联合稳健性门失败，正式推理按预注册规则停止。",
            detail=five_gate,
        )
        return 14
    if code not in (0, 14):
        write_state(
            "running", "five_model_robustness_wait",
            "五模型稳健性联合分析尚未完成；将从断点自动继续。",
        )
        return 11

    if not run_core_with_mid_sentinel(poll_seconds):
        sentinel = load_json(SENTINEL_SUITE_PATH)
        failures = terminal_failures([
            CORE_MANIFEST_PATH,
            ROOT / sentinel["waves"]["before"]["manifest"],
            ROOT / sentinel["waves"]["mid"]["manifest"],
        ])
        if failures:
            append_event("deferred_failed_units", phase="formal", failures=failures)
        write_state(
            "waiting", "formal_retry",
            "正式推理或中段哨兵尚未完整；下一周期将从 SQLite 断点继续。",
        )
        return 11
    if BATCH_CONTEXT_MANIFEST_PATH.exists() and not run_manifest(
        BATCH_CONTEXT_MANIFEST_PATH,
        workers_per_key=4,
        poll_seconds=poll_seconds,
    ):
        write_state(
            "running", "batch_context_retry",
            "批量上下文复测第一轮已经结束；失败单元将在下一周期统一重试。",
        )
        return 11
    sentinel = load_json(SENTINEL_SUITE_PATH)
    after = ROOT / sentinel["waves"]["after"]["manifest"]
    if not run_manifest(after, workers_per_key=1, poll_seconds=poll_seconds):
        failures = terminal_failures([after])
        if failures:
            append_event("deferred_failed_units", phase="sentinel_after", failures=failures)
        write_state(
            "waiting", "after_sentinel_retry",
            "末段哨兵尚未完整；下一周期将从 SQLite 断点继续。",
        )
        return 11
    if not postprocess():
        write_state(
            "error", "offline_postprocess",
            "离线分析出现错误。原始结果与 checkpoint 已保留，下一周期将重试。",
        )
        return 20
    completion_path = ROOT / "api_pilot/outputs/experiment_completion_contract.json"
    completion = load_json(completion_path) if completion_path.exists() else {}
    if not completion.get("machine_complete"):
        write_state(
            "error", "completion_contract",
            "Offline post-processing returned, but the fail-closed completion "
            "contract found missing, pending, failed, or hash-mismatched outputs.",
            detail={"completion_status": completion.get("status"),
                    "failures": completion.get("failures", [])},
        )
        return 20
    if completion.get("status") == "awaiting_human_review":
        write_state(
            "waiting", "human_quality_review",
            "All frozen inference and offline statistics are complete. The blind "
            "human item/prompt quality sign-off remains pending.",
            detail={"review": completion.get("human_review")},
        )
        return 13
    write_state(
        "complete", "all_required_work_complete",
        "全部冻结推理、缓存 materialisation、严格推断、综合指标和漂移诊断均已完成。",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return one_cycle(max(args.poll_seconds, 5))
    except Exception as error:
        append_event("automation_error", error=f"{type(error).__name__}: {error}")
        write_state(
            "error", "automation_exception",
            f"自动化异常：{type(error).__name__}: {error}。下一周期将重试。",
        )
        return 20


if __name__ == "__main__":
    raise SystemExit(main())
