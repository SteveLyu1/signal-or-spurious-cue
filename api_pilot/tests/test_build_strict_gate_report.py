from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from api_pilot.build_strict_gate_report import build_status


class StrictGateReportTests(unittest.TestCase):
    def test_researcher_public_terms_determination_passes_without_claiming_owner_approval(self):
        root = Path(__file__).resolve().parents[2]
        models = {
            "deepseek": "deepseek-v4-pro",
            "gemini": "gemini-3.6-flash",
            "mistral": "mistral-medium-3-5",
            "qwen": "qwen3.7-max-2026-06-08",
            "openai": "gpt-5.4",
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            evidence = base / "determination.md"
            evidence.write_text("researcher determination", encoding="utf-8")
            permission = base / "permission.json"
            permission.write_text(json.dumps({
                "schema_version": "respondent-data-processing-decision-v2",
                "status": "approved_by_researcher_under_public_terms",
                "written_basis_on_file": True,
                "researcher_risk_determination": True,
                "non_profit_research": True,
                "raw_data_redistribution_prohibited": True,
                "deidentified_derived_prompts_only": True,
                "claims_endpoint_specific_data_owner_approval": False,
                "scope": {"provider_models": models},
                "evidence_document": str(evidence),
                "evidence_document_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "reviewed_by": "researcher",
                "reviewed_at": "2026-07-25T16:00:55+08:00",
            }), encoding="utf-8")
            from api_pilot.build_strict_gate_report import evaluate_permission
            result = evaluate_permission(permission, models)
        self.assertTrue(result["passes"])
        self.assertEqual(result["status"], "researcher_determination_verified")
        self.assertIn("researcher determination", result["basis"])

    def test_researcher_decision_does_not_block_if_audit_attachment_moves(self):
        from api_pilot.build_strict_gate_report import evaluate_permission
        models = {"qwen": "frozen-model"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps({
                "schema_version": "respondent-data-processing-decision-v2",
                "status": "approved_by_researcher_under_public_terms",
                "written_basis_on_file": True,
                "researcher_risk_determination": True,
                "non_profit_research": True,
                "raw_data_redistribution_prohibited": True,
                "deidentified_derived_prompts_only": True,
                "claims_endpoint_specific_data_owner_approval": False,
                "scope": {"provider_models": models},
                "evidence_document": "moved.md",
                "evidence_document_sha256": "missing",
            }), encoding="utf-8")
            result = evaluate_permission(path, models)
        self.assertTrue(result["passes"])
        self.assertIn("evidence_exists", result["audit_warnings"])

    def test_report_keeps_inference_completion_distinct_from_gate_closure(self):
        root = Path(__file__).resolve().parents[2]
        status, markdown = build_status(
            root / "api_pilot/outputs/gate_c_v3_strict_inference.json",
            root / "api_pilot/outputs/gate_c_v3_strict_inference_official4.json",
            root / "api_pilot/outputs/confirmatory_504_design_power.json",
        )
        self.assertTrue(status["audit"]["complete"])
        self.assertFalse(status["gate"]["closed"])
        blocker_ids = {row["id"] for row in status["gate"]["blockers"]}
        self.assertNotIn("scientific_threshold_evidence", blocker_ids)
        self.assertNotIn("official_openai_provenance_or_roster_revision", blocker_ids)
        self.assertNotIn("macro_power_design", blocker_ids)
        self.assertNotIn("data_processing_permission", blocker_ids)
        self.assertTrue(status["data_processing_permission"]["passes"])
        self.assertIn("single_vs_batch", blocker_ids)
        self.assertTrue(status["power"]["design_power_gate_passes"])
        self.assertGreaterEqual(
            status["power"]["macro_joint_holm_power"],
            status["power"]["power_target"],
        )
        self.assertIn("完整 Gate 未关闭", markdown)
        self.assertIn("最小科学效应", markdown)
        self.assertIn("核心四模型敏感性", markdown)
        self.assertIn("完整五模型视图", markdown)
        self.assertIn("504 条确认性设计功效", markdown)
        self.assertEqual(
            status["display_state"],
            "GATE C PILOT COMPLETE / FULL GATE NOT PASSED",
        )

    def test_gate_opens_only_with_verified_permission_and_all_robustness_passes(self):
        root = Path(__file__).resolve().parents[2]
        models = {
            "deepseek": "deepseek-v4-pro",
            "gemini": "gemini-3.6-flash",
            "mistral": "mistral-medium-3-5",
            "qwen": "qwen3.7-max-2026-06-08",
            "openai": "gpt-5.4",
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            evidence = base / "approval.txt"
            evidence.write_text("approved", encoding="utf-8")
            permission = base / "permission.json"
            permission.write_text(
                json.dumps(
                    {
                        "schema_version": "respondent-data-processing-permission-v1",
                        "status": "approved",
                        "written_basis_on_file": True,
                        "third_party_api_processing_authorized": True,
                        "scope": {"provider_models": models},
                        "evidence_document": str(evidence),
                        "evidence_document_sha256": hashlib.sha256(
                            evidence.read_bytes()
                        ).hexdigest(),
                        "reviewed_by": "institutional reviewer",
                        "reviewed_at": "2026-07-25",
                    }
                ),
                encoding="utf-8",
            )
            reports = {}
            for name in (
                "repeat_identical",
                "single_item",
                "wording_b",
                "option_reverse",
            ):
                path = base / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "gate_decision": {"status": "pass", "passes": True},
                        }
                    ),
                    encoding="utf-8",
                )
                reports[name] = path
            status, _ = build_status(
                root / "api_pilot/outputs/gate_c_v3_strict_inference.json",
                root / "api_pilot/outputs/gate_c_v3_strict_inference_official4.json",
                root / "api_pilot/outputs/confirmatory_504_design_power.json",
                permission_path=permission,
                robustness_report_paths=reports,
            )
        self.assertTrue(status["gate"]["full_gate_c_pass"])
        self.assertTrue(status["data_processing_permission"]["passes"])
        self.assertTrue(status["robustness"]["all_pass"])
        self.assertEqual(status["gate"]["blockers"], [])


if __name__ == "__main__":
    unittest.main()
