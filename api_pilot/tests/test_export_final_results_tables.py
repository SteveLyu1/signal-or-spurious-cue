from __future__ import annotations

import csv
import json

from api_pilot import export_final_results_tables as exporter


def test_final_table_export_matches_authoritative_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "OUTPUT_DIR", tmp_path)
    assert exporter.main() == 0

    with (tmp_path / "table_1_main_hypotheses.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    decisions = {row["hypothesis"]: row["decision"] for row in rows}
    assert decisions["H1"] == "not_supported"
    assert decisions["H2"] == "supported_in_initial_version"
    h1 = next(row for row in rows if row["hypothesis"] == "H1")
    assert h1["preregistered_assignment_p"]

    with (tmp_path / "table_3_model_metrics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        model_rows = list(csv.DictReader(handle))
    assert {row["provider"] for row in model_rows} == {
        "deepseek", "gemini", "mistral", "openai", "qwen"
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["machine_complete"] is True
    contract = json.loads(
        (exporter.ROOT / "api_pilot/outputs/experiment_completion_contract.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["human_review_complete"] is contract["human_review"]["passes"]
    assert "table_5_checklist_correction_sensitivity.csv" in manifest["tables"]
    assert "table_6_independent_correction_replication.csv" in manifest["tables"]
    assert "table_17_clean_replacement_consistency.csv" in manifest["tables"]
    assert "clean_replacement_consistency" in manifest["source_sha256"]
    with (tmp_path / "table_6_independent_correction_replication.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        replication_rows = list(csv.DictReader(handle))
    replication_decisions = {
        row["hypothesis"]: row["decision"] for row in replication_rows
    }
    assert replication_decisions["H1"] == "not_supported"
    assert replication_decisions["H2"] == "supported"
    assert {
        row["analysis_role"] for row in replication_rows
    } == {"primary_sample_disjoint_corrected_protocol"}
    assert set(manifest["table_sha256"]) == set(manifest["tables"])
    assert len(manifest["source_sha256"]) == 10
    for digest in manifest["table_sha256"].values():
        assert len(digest) == 64
