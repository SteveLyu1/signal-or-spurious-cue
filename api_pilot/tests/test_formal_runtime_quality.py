from __future__ import annotations

import json

from api_pilot import summarize_formal_runtime_quality as quality


def test_output_audit_summarises_attempts_tokens_and_latency(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {
        "call_id": "c1", "provider": "p", "model_requested": "m",
        "success": True, "terminal_failure": False, "status_code": 200,
        "latency_ms": 20, "attempts": 2,
        "attempt_history": [
            {"status_code": 429, "latency_ms": 10},
            {"status_code": 200, "latency_ms": 20},
        ],
        "input_tokens": 100, "output_tokens": 20,
        "reasoning_tokens": None, "cached_tokens": 5,
        "error_type": None,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    audit, providers = quality.audit_output(path, {"c1": {"provider": "p", "model": "m"}})
    assert audit["complete"]
    assert providers["p"]["valid_rate"] == 1.0
    assert providers["p"]["recorded_attempts"] == 2
    assert providers["p"]["recorded_retries"] == 1
    assert providers["p"]["http_status_attempts"] == {"200": 1, "429": 1}
    assert providers["p"]["input_tokens_total"] == 100
    assert providers["p"]["reasoning_tokens_missing_calls"] == 1


def test_output_audit_fails_closed_on_missing_terminal_or_duplicate(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {
        "call_id": "c1", "provider": "p", "model_requested": "m",
        "success": False, "terminal_failure": True, "status_code": 200,
        "latency_ms": 5, "attempts": 1,
        "attempt_history": [{"status_code": 200, "latency_ms": 5}],
        "error_type": "validation_error",
    }
    path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    expected = {
        "c1": {"provider": "p", "model": "m"},
        "c2": {"provider": "p", "model": "m"},
    }
    audit, providers = quality.audit_output(path, expected)
    assert not audit["complete"]
    assert audit["duplicate_call_ids"] == 1
    assert audit["missing_calls"] == 1
    assert providers["p"]["terminal_failures"] == 1
    assert providers["p"]["valid_calls"] == 0
