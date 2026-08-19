from __future__ import annotations

import json

from api_pilot import verify_five_model_completion as verifier


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_report_only_failure_and_batch_context_close_machine_contract(
    monkeypatch, tmp_path
):
    panels = {}
    for name in ("core", "batch_context"):
        output = tmp_path / "outputs" / f"{name}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"call_id": name, "success": True}) + "\n",
            encoding="utf-8",
        )
        panels[name] = {
            "output": output.relative_to(tmp_path).as_posix(),
            "expected": 1,
            "complete": True,
        }
    sync = tmp_path / "outputs" / "sync.json"
    write_json(sync, {"panels": panels})
    reports = tmp_path / "api_pilot" / "outputs"
    logical = tmp_path / "outputs" / "logical.jsonl"
    logical.write_text('{"derived": true}\n', encoding="utf-8")
    write_json(
        reports / "confirmatory_core_v2_cache_audit.json",
        {
            "status": "cached_logical_panel_complete",
            "source_audit": {"complete": True},
            "cache_audit": {"complete": True, "logical_rows": 1},
            "logical_output": {
                "path": logical.relative_to(tmp_path).as_posix(),
                "sha256": verifier.hashlib.sha256(logical.read_bytes()).hexdigest(),
            },
        },
    )
    write_json(
        reports / "confirmatory_core_v2_strict_inference.json",
        {
            "status": "strict_confirmatory_inference_complete",
            "source_audit": {"complete": True},
        },
    )
    write_json(
        reports / "confirmatory_core_v2_comprehensive_metrics.json",
        {
            "status": "confirmatory_secondary_metrics_complete",
            "source_audit": {"complete": True},
            "cache_audit": {"complete": True},
            "item_panel_audit": {"complete": True},
        },
    )
    write_json(
        reports / "confirmatory_sentinel_analysis.json",
        {
            "status": "complete",
            "complete": True,
            "core_audit": {"complete": True},
            "wave_audits": {"before": {"complete": True}},
        },
    )
    write_json(
        reports / "confirmatory_batch_context_analysis.json",
        {"status": "complete", "complete": True, "audit": {"complete": True}},
    )
    write_json(
        reports / "five_model_robustness_gate.json",
        {"status": "fail", "complete": True},
    )
    write_json(
        reports / "gate_c_v3_postprocess_status.json",
        {"audit": {"planned": 3}},
    )
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "SYNC", sync)
    monkeypatch.setattr(verifier, "OUTPUT", reports / "contract.json")
    monkeypatch.setattr(
        verifier,
        "correction_supplement_check",
        lambda root: ({"passes": True, "expected": 5_760}, {"status": "complete"}),
    )
    monkeypatch.setattr(
        verifier,
        "independent_replication_check",
        lambda root: ({"passes": True, "expected": 2_880}, {"status": "complete"}),
    )
    review_packet = tmp_path / "api_pilot" / "human_review" / "BLIND_QUALITY_REVIEW.md"
    review_packet.parent.mkdir(parents=True, exist_ok=True)
    review_packet.write_text("review packet", encoding="utf-8")
    monkeypatch.setattr(verifier, "REVIEW_PACKET", review_packet)
    review_csv = review_packet.parent / "blind_quality_review_template.csv"
    correction_csv = review_packet.parent / "checklist_correction_review.csv"
    review_csv.write_text("review", encoding="utf-8")
    correction_csv.write_text("correction", encoding="utf-8")
    monkeypatch.setattr(verifier, "REVIEW_CSV", review_csv)
    monkeypatch.setattr(verifier, "CORRECTION_REVIEW_CSV", correction_csv)

    assert verifier.main() == 0
    contract = json.loads(verifier.OUTPUT.read_text(encoding="utf-8"))
    assert contract["machine_complete"] is True
    assert contract["status"] == "awaiting_human_review"
    assert contract["planned_api_calls"] == 8_645
    assert contract["human_review"]["checks"]["approved_status"] is False


def test_human_review_requires_identity_time_hash_and_no_rejections(
    monkeypatch, tmp_path
):
    packet = tmp_path / "BLIND_QUALITY_REVIEW.md"
    review_csv = tmp_path / "review.csv"
    correction_csv = tmp_path / "correction.csv"
    packet.write_text("frozen packet", encoding="utf-8")
    review_csv.write_text("review", encoding="utf-8")
    correction_csv.write_text("correction", encoding="utf-8")
    monkeypatch.setattr(verifier, "REVIEW_PACKET", packet)
    expected_hash = verifier.hashlib.sha256(packet.read_bytes()).hexdigest()
    signoff = {
        "status": "approved",
        "reviewer": "Reviewer A",
        "reviewed_at": "2026-07-26T18:00:00+08:00",
        "review_packet_sha256": expected_hash,
        "original_review_csv_sha256": verifier.hashlib.sha256(review_csv.read_bytes()).hexdigest(),
        "correction_review_csv_sha256": verifier.hashlib.sha256(correction_csv.read_bytes()).hexdigest(),
        "correction_reviewed_rows": 8,
        "reviewed_rows": 34,
        "unresolved_rejections": [],
    }
    checks = {
        "approved_status": signoff.get("status") == "approved",
        "reviewer_recorded": bool(signoff.get("reviewer")),
        "review_time_recorded": bool(signoff.get("reviewed_at")),
        "packet_hash_matches": signoff.get("review_packet_sha256") == expected_hash,
        "original_review_hash_matches": signoff.get("original_review_csv_sha256") == verifier.hashlib.sha256(review_csv.read_bytes()).hexdigest(),
        "correction_review_hash_matches": signoff.get("correction_review_csv_sha256") == verifier.hashlib.sha256(correction_csv.read_bytes()).hexdigest(),
        "correction_rows_reviewed": signoff.get("correction_reviewed_rows") == 8,
        "original_rows_reviewed": signoff.get("reviewed_rows") == 34,
        "no_unresolved_rejections": signoff.get("unresolved_rejections") == [],
    }
    assert all(checks.values())


def test_correction_supplement_is_fail_closed_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    check, analysis = verifier.correction_supplement_check(tmp_path)
    assert check["passes"] is False
    assert analysis == {}
