#!/usr/bin/env python3
"""Render a sanitized, auto-refreshing HTML view of an inference checkpoint."""

from __future__ import annotations

import argparse
import atexit
import html
import json
import os
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_pid_file(path: Path) -> None:
    if path.exists():
        try:
            previous = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            previous = 0
        if previous != os.getpid() and process_exists(previous):
            raise RuntimeError(f"monitor already running with PID {previous}")
    atomic_write(path, str(os.getpid()) + "\n")

    def cleanup() -> None:
        try:
            if path.exists() and path.read_text(encoding="ascii").strip() == str(os.getpid()):
                path.unlink()
        except OSError:
            pass

    atexit.register(cleanup)


def read_checkpoint(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as connection:
        metadata_row = connection.execute(
            "SELECT value FROM run_meta WHERE key='run_metadata'"
        ).fetchone()
        if metadata_row is None:
            raise ValueError("checkpoint has no frozen run metadata")
        metadata = json.loads(metadata_row[0])
        rows = [
            json.loads(row[0])
            for row in connection.execute("SELECT result_json FROM results")
        ]
    return metadata, rows


def checkpoint_mtime(path: Path) -> float:
    candidates = [path, path.with_name(path.name + "-wal")]
    return max(candidate.stat().st_mtime for candidate in candidates if candidate.exists())


def failure_label(rows: list[dict[str, Any]]) -> str:
    failures = [row for row in rows if not row.get("success")]
    if not failures:
        return ""
    statuses = Counter(row.get("status_code") for row in failures)
    types = Counter(row.get("error_type") for row in failures)
    if statuses[503]:
        return "model capacity / high demand (HTTP 503)"
    if statuses[429]:
        return "quota / rate limit (HTTP 429)"
    if any(statuses[status] for status in (401, 402, 403)):
        return "credential or billing"
    if types["transport_error"]:
        return "retryable network transport"
    if any(row.get("terminal_failure") for row in failures):
        return "terminal probability / format validation"
    return "retryable API failure"


def percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 100.0


def status_for(
    valid: int,
    terminal: int,
    planned: int,
    retryable: int,
    recent_checkpoint: bool,
    failure: str,
) -> tuple[str, str]:
    pending = max(planned - valid - terminal, 0)
    if pending == 0 and terminal == 0:
        return "DONE", "ok"
    if pending == 0:
        return "DONE WITH INVALID", "bad"
    if "capacity" in failure:
        return "WAITING FOR MODEL CAPACITY", "warn"
    if "quota" in failure and recent_checkpoint:
        return "RECOVERING (LOW CONCURRENCY)", "active"
    if "quota" in failure:
        return "WAITING FOR QUOTA", "warn"
    if "credential" in failure:
        return "BLOCKED: CREDENTIAL", "bad"
    if recent_checkpoint:
        return "RUNNING", "active"
    if retryable:
        return "PAUSED / RETRYABLE", "warn"
    return "PAUSED", "warn"


def render_postprocess_status(path: Path | None) -> str:
    if path is None or not path.exists():
        return (
            "<p><span class='badge warn'>Gate C 推理状态未知</span> "
            "后处理状态尚未生成。</p>"
            "<p><span class='badge warn'>完整 Gate 未开放</span> "
            "尚不能启动确认性推理。</p>"
        )
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        audit = status.get("audit") or {}
        gate = status.get("gate") or {}
        outputs = status.get("outputs") or {}
        gate_report_path = outputs.get("gate_report")
        if not gate and gate_report_path:
            candidate = Path(gate_report_path)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                gate = json.loads(candidate.read_text(encoding="utf-8"))

        inference_complete = bool(audit.get("complete")) or (
            str(status.get("state") or "").lower() == "complete"
        )
        full_gate_open = bool(
            gate.get("closed")
            if "closed" in gate
            else gate.get("full_gate_c_pass", False)
        )
        inference_label = (
            "Gate C 推理已完成" if inference_complete else "Gate C 推理未完成"
        )
        inference_class = "ok" if inference_complete else "warn"
        gate_label = "完整 Gate 已开放" if full_gate_open else "完整 Gate 未开放"
        gate_class = "ok" if full_gate_open else "warn"
        content = (
            f"<p><span class='badge {inference_class}'>{inference_label}</span> "
            f"已合并 {int(audit.get('merged') or 0):,} / "
            f"{int(audit.get('planned') or 0):,}，缺失 "
            f"{int(audit.get('missing') or 0):,}，未完成 "
            f"{int(audit.get('incomplete') or 0):,}。</p>"
            f"<p><span class='badge {gate_class}'>{gate_label}</span> "
            + (
                "确认性推理可以按冻结方案启动。"
                if full_gate_open
                else "核心推理完成不代表稳健性、许可与模型来源等门槛已经关闭。"
            )
            + "</p>"
        )
        strict = status.get("strict_inference") or {}
        if strict:
            content += (
                "<p>严格加权 Gate C pilot："
                f"βO={float(strict.get('beta_o') or 0):.4f}，"
                f"βR={float(strict.get('beta_r') or 0):.4f}，"
                f"βO−βR={float(strict.get('beta_diff') or 0):.4f}，"
                f"MV utility={float(strict.get('utility_v') or 0):.4f}。</p>"
            )
        power = status.get("power") or {}
        if power.get("design_power_gate_passes"):
            content += (
                "<p><span class='badge ok'>504 条宏观设计功效通过</span> "
                f"联合 Holm 功效={float(power.get('macro_joint_holm_power') or 0):.3f}，"
                f"冻结门槛={float(power.get('power_target') or 0):.2f}。"
                "这是基于 pilot 误差结构可迁移假设的前瞻近似。</p>"
            )
        robustness = status.get("robustness") or {}
        if robustness.get("status") == "offline_manifests_frozen_inference_pending":
            content += (
                "<p><span class='badge warn'>稳健性任务已冻结 / 推理待许可</span> "
                f"四项任务共 {int(robustness.get('total_planned_calls') or 0):,} 次调用；"
                "当前没有把待运行状态当作通过。</p>"
            )
        if robustness.get("experiments"):
            rows = "".join(
                "<tr>"
                f"<td><code>{html.escape(str(name))}</code></td>"
                f"<td>{int(value.get('planned_calls') or 0):,}</td>"
                f"<td>{html.escape(str(value.get('analysis_status') or 'pending'))}</td>"
                f"<td>{html.escape(str(value.get('gate_status') or 'pending'))}</td>"
                "</tr>"
                for name, value in robustness["experiments"].items()
            )
            content += (
                "<table><thead><tr><th>稳健性任务</th><th>调用</th>"
                "<th>分析</th><th>Gate</th></tr></thead><tbody>"
                + rows
                + "</tbody></table>"
            )
        permission = status.get("data_processing_permission") or {}
        if permission:
            permission_class = "ok" if permission.get("passes") else "warn"
            content += (
                f"<p><span class='badge {permission_class}'>数据处理许可："
                f"{html.escape(str(permission.get('status') or 'pending'))}</span> "
                "正式记录不得在该项通过前发送到第三方端点。</p>"
            )
        confirmatory = status.get("confirmatory_execution") or {}
        if confirmatory:
            content += (
                "<p><span class='badge warn'>正式核心推理</span> "
                f"{int(confirmatory.get('observed_successes') or 0):,} / "
                f"{int(confirmatory.get('planned_calls') or 0):,}；分析状态 "
                f"<code>{html.escape(str(confirmatory.get('analysis_status') or 'pending'))}</code>。"
                "</p>"
            )
        blockers = gate.get("blockers") or []
        if not blockers:
            blockers = [
                {"id": key, "reason": value}
                for key, value in (gate.get("outstanding_gate_components") or {}).items()
                if value
            ]
        if blockers:
            entries = "".join(
                f"<li><code>{html.escape(str(row.get('id') or 'unknown'))}</code>: "
                f"{html.escape(str(row.get('reason') or 'pending'))}</li>"
                for row in blockers
            )
            content += f"<p>完整 Gate 尚未关闭：</p><ul>{entries}</ul>"
        if outputs:
            content += "<p>可信输出：" + ", ".join(
                f"<code>{html.escape(str(value))}</code>"
                for value in outputs.values()
            ) + "</p>"
        return content
    except (OSError, ValueError, TypeError):
        return "<p><span class='badge bad'>STATUS ERROR</span></p>"


def render_page_v2(
    *,
    interval: int,
    overall_status: str,
    overall_class: str,
    updated_local: str,
    checkpoint_utc: str,
    checkpoint_age: float,
    planned_total: int,
    valid_total: int,
    terminal_total: int,
    pending_total: int,
    provider_rows: list[str],
    cell_rows: list[str],
    checkpoint: Path,
    active_overlays: list[Path],
    postprocess_html: str,
    generated_at: datetime,
) -> str:
    refresh_minutes = max(interval // 60, 1)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{interval}">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AAAI Experiment Monitor</title>
  <style>
    :root {{ color-scheme:light; --ink:#17202a; --muted:#65717e; --line:#d9dee3;
      --panel:#fff; --bg:#f4f6f7; --green:#18794e; --amber:#9a6700;
      --red:#b42318; --blue:#175cd3; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink);
      font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }}
    main {{ width:min(1220px,calc(100% - 32px)); margin:24px auto 48px; }}
    header {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin-bottom:18px; }}
    h1 {{ font-size:24px; margin:0 0 6px; }} h2 {{ font-size:16px; margin:0 0 12px; }}
    p {{ margin:4px 0; color:var(--muted); }} code {{ font-family:"Cascadia Code",Consolas,monospace; font-size:12px; }}
    .badge {{ display:inline-block; border-radius:4px; padding:4px 8px; font-weight:700; white-space:nowrap; }}
    .ok {{ color:var(--green); background:#e7f6ec; }} .active {{ color:var(--blue); background:#eaf1ff; }}
    .warn {{ color:var(--amber); background:#fff4d6; }} .bad {{ color:var(--red); background:#feeceb; }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:16px; }}
    .metric,.section {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; }}
    .metric {{ padding:14px; }} .metric b {{ display:block; font-size:22px; margin-top:3px; }}
    .metric span {{ color:var(--muted); font-size:12px; }} .section {{ padding:16px; margin-top:12px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }} th,td {{ padding:9px 10px; text-align:left;
      border-bottom:1px solid var(--line); vertical-align:middle; }} th {{ color:var(--muted); font-size:12px; font-weight:600; }}
    tr:last-child td {{ border-bottom:0; }} .bar {{ width:130px; height:7px; background:#e9edf0; border-radius:4px; overflow:hidden; }}
    .bar span {{ display:block; height:100%; background:var(--blue); }} small {{ color:var(--muted); }}
    .meta {{ text-align:right; }} .notice {{ padding:10px 12px; border-left:3px solid var(--amber); background:#fff8e8; color:#654b00; }}
    @media (max-width:760px) {{ header {{ display:block; }} .meta {{ text-align:left; margin-top:10px; }}
      .summary {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} main {{ width:min(100% - 20px,1220px); margin-top:14px; }} }}
  </style>
</head>
<body><main>
  <header><div><h1>AAAI 实验推理监控</h1>
    <span class="badge {overall_class}">{overall_status}</span>
    <p>Gate C 五系统并行推理。页面与后台快照每 {refresh_minutes} 分钟刷新一次。</p></div>
    <div class="meta"><p>页面生成：{html.escape(updated_local)}</p><p>Checkpoint 写入：{checkpoint_utc}</p>
    <p>快照延迟：{int(checkpoint_age)} 秒</p></div></header>
  <div class="summary">
    <div class="metric"><span>计划调用</span><b>{planned_total:,}</b></div>
    <div class="metric"><span>分析可用 checkpoint</span><b>{valid_total:,}</b></div>
    <div class="metric"><span>终止无效输出</span><b>{terminal_total:,}</b></div>
    <div class="metric"><span>尚未完成</span><b>{pending_total:,}</b></div>
  </div>
  <div class="notice">推理完成不等于完整研究 Gate 通过。断点恢复会跳过已经成功的 call ID。</div>
  <section class="section"><h2>模型进度</h2><table><thead><tr><th>Provider / Model</th><th>状态</th>
    <th>分析可用</th><th>归一化修复</th><th>终止无效</th><th>可重试</th><th>待完成</th><th>进度</th><th>当前问题</th></tr></thead>
    <tbody>{''.join(provider_rows)}</tbody></table></section>
  <section class="section"><h2>实验单元</h2><table><thead><tr><th>Metadata</th><th>Evidence</th><th>计划</th>
    <th>分析可用</th><th>终止无效</th><th>待完成</th></tr></thead><tbody>{''.join(cell_rows)}</tbody></table></section>
  <section class="section"><h2>恢复信息</h2>
    <p>SQLite：<code>{html.escape(str(checkpoint))}</code></p>
    <p>恢复 overlays：<code>{html.escape(', '.join(str(path) for path in active_overlays) or 'none')}</code></p>
    <p>恢复命令：<code>powershell -ExecutionPolicy Bypass -File api_pilot/scripts/archive/gate_c/run_gate_c.ps1</code></p>
    <p>监控页只读取聚合状态，不展示 key、profile、prompt 或逐条预测。生成时间：{generated_at.isoformat()}</p>
  </section>
  <section class="section"><h2>完成后评估</h2>{postprocess_html}
    <p>严格结果使用调查权重、六国等权宏平均和 profile-cluster bootstrap，并在每次方向 bootstrap 中重估 reference anchors。Gate C 的标签采用确定性平衡轮转，因此标签重分配只作 sharp-null 诊断，不是设计精确随机化检验。</p>
  </section>
</main></body></html>"""


def render(
    checkpoint: Path,
    interval: int,
    overlay_checkpoints: list[Path] | None = None,
    replacement_checkpoints: list[Path] | None = None,
    postprocess_status: Path | None = None,
) -> str:
    metadata, rows = read_checkpoint(checkpoint)
    models = metadata.get("models") or {}
    by_call_id = {row["call_id"]: row for row in rows}
    active_overlays = []
    provider_mtimes = {
        provider: checkpoint_mtime(checkpoint) for provider in models
    }
    for overlay in overlay_checkpoints or []:
        if not overlay.exists():
            continue
        overlay_metadata, overlay_rows = read_checkpoint(overlay)
        for row in overlay_rows:
            provider = row.get("provider")
            if provider not in models:
                raise ValueError(f"overlay has unknown provider: {provider}")
            if row.get("model_requested") != models[provider]:
                raise ValueError(
                    f"overlay model mismatch for {provider}: {row.get('model_requested')}"
                )
            by_call_id[row["call_id"]] = row
        for provider in overlay_metadata.get("models") or {}:
            provider_mtimes[provider] = max(
                provider_mtimes.get(provider, 0), checkpoint_mtime(overlay)
            )
        active_overlays.append(overlay)
    for replacement in replacement_checkpoints or []:
        if not replacement.exists():
            continue
        replacement_metadata, replacement_rows = read_checkpoint(replacement)
        replacement_models = replacement_metadata.get("models") or {}
        if len(replacement_models) != 1:
            raise ValueError("replacement checkpoint must contain exactly one provider")
        provider, model = next(iter(replacement_models.items()))
        if provider not in models:
            raise ValueError(f"replacement has unknown provider: {provider}")
        models[provider] = model
        by_call_id = {
            call_id: row
            for call_id, row in by_call_id.items()
            if row.get("provider") != provider
        }
        for row in replacement_rows:
            if row.get("provider") != provider or row.get("model_requested") != model:
                raise ValueError(f"invalid replacement row for {provider}")
            by_call_id[row["call_id"]] = row
        provider_mtimes[provider] = checkpoint_mtime(replacement)
        active_overlays.append(replacement)
    rows = list(by_call_id.values())
    providers = list(models)
    cells = [tuple(cell) for cell in metadata.get("cells") or []]
    planned_total = int(metadata.get("planned_calls") or 0)
    planned_provider = planned_total // max(len(providers), 1)
    planned_cell = planned_total // max(len(cells), 1)
    now = datetime.now(timezone.utc)
    latest_checkpoint_mtime = max(provider_mtimes.values())
    checkpoint_age = max(time.time() - latest_checkpoint_mtime, 0)
    recent_checkpoint = checkpoint_age < max(interval * 1.5, 120)

    valid_total = sum(bool(row.get("success")) for row in rows)
    terminal_total = sum(bool(row.get("terminal_failure")) for row in rows)
    retryable_total = sum(
        not row.get("success") and not row.get("terminal_failure") for row in rows
    )
    completed_total = valid_total + terminal_total
    pending_total = max(planned_total - completed_total, 0)

    provider_rows = []
    provider_statuses = []
    for provider in providers:
        subset = [row for row in rows if row.get("provider") == provider]
        valid = sum(bool(row.get("success")) for row in subset)
        terminal = sum(bool(row.get("terminal_failure")) for row in subset)
        retryable = sum(
            not row.get("success") and not row.get("terminal_failure") for row in subset
        )
        repaired = sum(bool(row.get("postprocess_repair")) for row in subset)
        pending = max(planned_provider - valid - terminal, 0)
        failure = failure_label(subset)
        provider_recent = (
            time.time() - provider_mtimes.get(provider, 0) < 120
        )
        status, css_class = status_for(
            valid, terminal, planned_provider, retryable, provider_recent, failure
        )
        provider_statuses.append(status)
        provider_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(provider)}</strong><br><code>{html.escape(str(models[provider]))}</code></td>"
            f"<td><span class='badge {css_class}'>{status}</span></td>"
            f"<td>{valid:,}</td><td>{repaired:,}</td><td>{terminal:,}</td><td>{retryable:,}</td><td>{pending:,}</td>"
            f"<td><div class='bar'><span style='width:{percent(valid + terminal, planned_provider):.2f}%'></span></div>"
            f"<small>{percent(valid + terminal, planned_provider):.1f}%</small></td>"
            f"<td>{html.escape(failure or 'none')}</td>"
            "</tr>"
        )

    cell_rows = []
    for condition, evidence in cells:
        subset = [
            row
            for row in rows
            if row.get("condition") == condition
            and row.get("evidence_variant") == evidence
        ]
        valid = sum(bool(row.get("success")) for row in subset)
        terminal = sum(bool(row.get("terminal_failure")) for row in subset)
        pending = max(planned_cell - valid - terminal, 0)
        cell_rows.append(
            "<tr>"
            f"<td><code>{html.escape(condition)}</code></td>"
            f"<td>{html.escape(evidence)}</td><td>{planned_cell:,}</td>"
            f"<td>{valid:,}</td><td>{terminal:,}</td><td>{pending:,}</td>"
            "</tr>"
        )

    if pending_total == 0 and terminal_total == 0:
        overall_status, overall_class = "GATE C INFERENCE COMPLETE", "ok"
    elif any(status.startswith("BLOCKED") for status in provider_statuses):
        overall_status, overall_class = "RUNNING WITH CREDENTIAL BLOCK", "bad"
    elif any(status == "WAITING FOR QUOTA" for status in provider_statuses):
        overall_status, overall_class = "RUNNING WITH QUOTA WAIT", "warn"
    elif recent_checkpoint:
        overall_status, overall_class = "RUNNING", "active"
    else:
        overall_status, overall_class = "PAUSED", "warn"

    updated_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    checkpoint_utc = datetime.fromtimestamp(
        latest_checkpoint_mtime, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
    return render_page_v2(
        interval=interval,
        overall_status=overall_status,
        overall_class=overall_class,
        updated_local=updated_local,
        checkpoint_utc=checkpoint_utc,
        checkpoint_age=checkpoint_age,
        planned_total=planned_total,
        valid_total=valid_total,
        terminal_total=terminal_total,
        pending_total=pending_total,
        provider_rows=provider_rows,
        cell_rows=cell_rows,
        checkpoint=checkpoint,
        active_overlays=active_overlays,
        postprocess_html=render_postprocess_status(postprocess_status),
        generated_at=now,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--overlay-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--replacement-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--postprocess-status", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pid_file = args.pid_file or args.html.with_suffix(".pid")
    acquire_pid_file(pid_file)
    while True:
        try:
            atomic_write(
                args.html,
                render(
                    args.checkpoint,
                    args.interval_seconds,
                    args.overlay_checkpoint,
                    args.replacement_checkpoint,
                    args.postprocess_status,
                ),
            )
            print(
                f"updated={datetime.now(timezone.utc).isoformat()} html={args.html}",
                flush=True,
            )
        except Exception as error:
            print(f"monitor error: {type(error).__name__}: {error}", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
