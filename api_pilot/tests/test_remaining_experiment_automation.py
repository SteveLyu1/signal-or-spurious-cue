from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from api_pilot import automate_remaining_experiment as automation


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_workstream_total_and_completed_pilot():
    rows = automation.workstreams()
    assert sum(row["planned"] for row in rows) == 60_600
    assert rows[0]["success"] == 4_320
    assert sum(row["success"] for row in rows) >= 4_320
    assert [row["name"] for row in rows] == [
        "Completed Gate C pilot",
        "Robustness: repeat_identical",
        "Robustness: single_item",
        "Robustness: wording_b",
        "Robustness: option_reverse",
        "Confirmatory core",
        "Report-only: batch-context replication",
        "Runtime sentinel: before",
        "Runtime sentinel: mid",
        "Runtime sentinel: after",
        "GPT parity: repeat_identical",
        "GPT parity: single_item",
        "GPT parity: wording_b",
        "GPT parity: option_reverse",
        "GPT parity: confirmatory_core",
        "GPT parity: sentinel_before",
        "GPT parity: sentinel_mid",
        "GPT parity: sentinel_after",
            "GPT parity: batch_context",
            "Post-review: checklist wording correction",
            "Post-review: independent corrected-wording replication",
        ]


def test_permission_audit_warning_does_not_block_inference(monkeypatch):
    monkeypatch.setattr(
        automation,
        "refresh_gate",
        lambda: {
            "data_processing_permission": {"passes": False},
            "gate": {"full_gate_c_pass": False, "blockers": []},
        },
    )
    states = []
    monkeypatch.setattr(
        automation, "write_state",
        lambda status, phase, message, **kwargs: states.append((status, phase)),
    )
    events = []
    monkeypatch.setattr(
        automation, "append_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )
    monkeypatch.setattr(automation, "run_robustness_parallel", lambda _: False)
    monkeypatch.setattr(automation, "load_json", lambda _: {"manifests": {}})
    assert automation.one_cycle(5) == 11
    assert events[0][0] == "permission_audit_warning"
    assert ("running", "gate_c_robustness") in states


def test_runner_command_uses_frozen_paths_models_and_acknowledgement():
    command = automation.runner_command(
        ROOT / "dataset/analysis/confirmatory_core_task_manifest_v2.json", 4
    )
    joined = " ".join(command)
    assert "--acknowledge-respondent-data-processing" in command
    assert "--dry-run" not in command
    assert "--checkpoint api_pilot/restricted_outputs/confirmatory_core_v2_raw.checkpoint.sqlite3" in joined
    assert "--analysis-manifest dataset/analysis/confirmatory_manifest_v2.json" in joined
    for model in (
        "deepseek-v4-pro", "gemini-3.6-flash", "mistral-medium-3-5",
        "qwen3.7-max-2026-06-08",
    ):
        assert model in command


def test_runner_phase_boundary_is_operational_not_a_task_subset():
    command = automation.runner_command(
        ROOT / "dataset/analysis/confirmatory_core_task_manifest_v2.json",
        4,
        stop_after_successes_per_provider=4_032,
    )
    assert command[-2:] == [
        "--stop-after-successes-per-provider", "4032"
    ]
    assert "--max-calls-per-provider" not in command


def test_checkpoint_counts_distinguishes_terminal_and_retryable(tmp_path):
    path = tmp_path / "checkpoint.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE results (call_id TEXT PRIMARY KEY, provider TEXT, "
        "success INTEGER, result_json TEXT, updated_at TEXT)"
    )
    rows = [
        ("ok", "x", 1, {"success": True}),
        ("terminal", "x", 0, {"success": False, "terminal_failure": True}),
        ("retry", "x", 0, {"success": False}),
    ]
    for call_id, provider, success, result in rows:
        connection.execute(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
            (call_id, provider, success, json.dumps(result), "now"),
        )
    connection.commit()
    connection.close()
    assert automation.checkpoint_counts(path) == {
        "success": 1, "terminal": 1, "retryable": 1,
    }
    assert automation.checkpoint_successes_by_provider(path) == {"x": 1}


def test_terminal_failures_are_not_retryable(tmp_path):
    checkpoint = tmp_path / "checkpoint.sqlite3"
    connection = sqlite3.connect(checkpoint)
    connection.execute(
        "CREATE TABLE results (call_id TEXT PRIMARY KEY, provider TEXT, "
        "success INTEGER, result_json TEXT, updated_at TEXT)"
    )
    connection.execute(
        "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
        ("bad", "x", 0, json.dumps({"terminal_failure": True}), "now"),
    )
    connection.commit()
    connection.close()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "experiment_id": "terminal_experiment",
            "expected": {"total_calls": 1},
            "immutable_checkpoint_path": checkpoint.name,
        }),
        encoding="utf-8",
    )
    original_root = automation.ROOT
    try:
        automation.ROOT = tmp_path
        assert automation.terminal_failures([manifest]) == {
            "terminal_experiment": 1
        }
    finally:
        automation.ROOT = original_root


def test_monitor_is_utf8_and_has_one_minute_refresh():
    state = {
        "updated_at": "now", "status": "waiting", "phase": "permission_gate",
        "message": "等待许可", "workstreams": [], "blockers": [],
        "providers": [{
            "provider": "gemini", "model": "gemini-test", "planned": 100,
            "success": 25, "retryable": 1, "terminal": 0, "pending": 75,
            "percentage": 25.0,
        }],
        "local_tasks": [{
            "name": "严格统计", "dependency": "API 输出", "status": "waiting",
            "completed_steps": 0, "planned_steps": 1, "share": 100.0,
        }],
    }
    page = automation.render_monitor_v2(state)
    assert 'content="60"' in page
    assert "等待许可" in page
    assert "API" in page
    assert "各 API 模型" in page
    assert "gemini-test" in page
    assert "25.0%" in page
    assert "API 推理任务" in page
    assert "本地计算任务" in page


def test_provider_progress_matches_frozen_total():
    rows = automation.provider_progress()
    assert sum(row["planned"] for row in rows) == sum(
        row["planned"] for row in automation.workstreams()
    )
    assert {row["provider"] for row in rows} == {
        "deepseek", "gemini", "mistral", "qwen", "openai",
    }
    assert next(row for row in rows if row["provider"] == "openai")["planned"] == 12_120


def test_local_compute_progress_has_separate_fixed_denominator():
    rows = automation.local_compute_progress("offline_refresh")
    assert len(rows) == 16
    assert sum(row["planned_steps"] for row in rows) == 16
    assert round(sum(row["share"] for row in rows), 8) == 100.0
    assert all(row["status"] in {"complete", "running", "waiting"} for row in rows)


def test_run_local_emits_heartbeat_until_child_finishes(monkeypatch, tmp_path):
    class Process:
        pid = 123
        returncode = 0

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    monkeypatch.setattr(automation, "ROOT", tmp_path)
    monkeypatch.setattr(automation, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(automation.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(automation.time, "sleep", lambda _: None)
    states = []
    monkeypatch.setattr(
        automation,
        "write_state",
        lambda status, phase, message, **kwargs: states.append(
            (status, phase, kwargs["detail"]["local_pid"])
        ),
    )
    assert automation.run_local(["example.py"], "offline_example") == 0
    assert states == [("running", "offline_example", 123)]


def test_postprocess_accepts_report_only_robustness_failure(monkeypatch):
    labels = []

    def fake_run_local(_command, label):
        labels.append(label)
        return 14 if label == "evaluate_five_model_robustness" else 0

    monkeypatch.setattr(automation, "run_local", fake_run_local)
    monkeypatch.setattr(automation, "write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(automation, "_json_task_complete", lambda *args: False)
    assert automation.postprocess() is True
    assert labels[0:2] == [
        "sync_five_model_panels", "evaluate_five_model_robustness"
    ]
    assert labels[-1] == "verify_experiment_completion"
