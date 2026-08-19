from __future__ import annotations

import json
from pathlib import Path

from api_pilot import verify_experiment_completion as completion


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_jsonl_audit_is_fail_closed_for_duplicate_and_failure(tmp_path):
    path = tmp_path / "results.jsonl"
    write_jsonl(path, [
        {"call_id": "a", "success": True},
        {"call_id": "a", "success": False, "terminal_failure": True},
    ])
    audit = completion.audit_jsonl(path, 2)
    assert not audit["complete"]
    assert audit["duplicate_call_ids"] == 1
    assert audit["failed_call_ids"] == ["a"]
    assert audit["terminal_call_ids"] == ["a"]


def test_pending_reports_cannot_close_machine_contract(tmp_path):
    core = {
        "source": {}, "prompt": {},
        "expected": {"total_calls": 1},
        "immutable_output_path": "raw.jsonl",
        "analysis_contract": {"cached_logical_panel": {
            "derived_output_path": "logical.jsonl"
        }},
    }
    path = tmp_path / completion.CORE_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(core), encoding="utf-8")
    contract = completion.evaluate(tmp_path)
    assert not contract["machine_complete"]
    assert contract["status"] == "pending_machine_outputs"
    assert {row["id"] for row in contract["failures"]} >= {
        "raw_core_complete", "strict_inference_complete_and_bound",
        "comprehensive_metrics_complete_and_bound",
        "sentinel_analysis_complete_and_bound",
        "runtime_quality_complete_and_bound",
    }


def test_human_signoff_must_bind_exact_review_packet_hash(tmp_path):
    path = tmp_path / completion.REVIEW_SIGNOFF
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": "approved", "reviewer": "Reviewer A",
        "reviewed_at": "2026-07-25T12:00:00+08:00",
        "review_packet_sha256": "wrong", "unresolved_rejections": [],
    }), encoding="utf-8")
    status = completion.human_review_status(tmp_path, "expected")
    assert not status["passes"]
    assert status["checks"]["packet_hash"] is False


def test_metric_rows_require_machine_completion():
    assert completion.metric_rows({"machine_complete": False}) == []
