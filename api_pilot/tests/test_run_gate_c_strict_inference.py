from __future__ import annotations

import json
import unittest
from pathlib import Path

from api_pilot.run_gate_c_strict_inference import (
    EXPECTED_MODELS,
    audit_complete_results,
    build_inference_rows,
)


class GateCRowBuilderTests(unittest.TestCase):
    def profile(self):
        return {
            "profile_id": "P-CN-001",
            "true_country_code": "CN",
            "analysis_weight": 2.0,
            "balanced_random_country_label": "United States",
            "targets": [
                {"item_id": "Q1", "answer_code": "1", "answer": "yes"}
            ],
            "target_options": {
                "Q1": [
                    {"answer_code": "0", "answer": "no"},
                    {"answer_code": "1", "answer": "yes"},
                ]
            },
        }

    def results(self):
        values = {
            "M0": [0.6, 0.4],
            "MO": [0.4, 0.6],
            "MR": [0.5, 0.5],
            "MV": [0.3, 0.7],
        }
        return [
            {
                "call_id": condition,
                "provider": "model",
                "profile_id": "P-CN-001",
                "condition": condition,
                "evidence_variant": "correct",
                "target_ids": ["Q1"],
                "target_option_codes": {"Q1": ["0", "1"]},
                "parsed_forecasts": {"Q1": forecast},
                "true_country_code": "CN",
                "displayed_country_code": "US" if condition in {"MO", "MR"} else None,
            }
            for condition, forecast in values.items()
        ]

    def test_builds_direction_and_positive_verified_utility(self):
        direction, utility, audit = build_inference_rows(
            self.results(),
            [self.profile()],
            ["Q1"],
            require_balanced_assignment=False,
        )
        self.assertAlmostEqual(direction[0]["mo_delta"][0], -0.2)
        self.assertAlmostEqual(direction[0]["mo_delta"][1], 0.2)
        self.assertAlmostEqual(direction[0]["mr_delta"][0], -0.1)
        self.assertAlmostEqual(direction[0]["mr_delta"][1], 0.1)
        self.assertGreater(utility[0]["value"], 0)
        self.assertEqual(audit["source_cells_checked"], 4)

    def test_fails_when_mo_and_mr_labels_differ(self):
        rows = self.results()
        next(row for row in rows if row["condition"] == "MR")[
            "displayed_country_code"
        ] = "FR"
        with self.assertRaisesRegex(ValueError, "displayed-label mismatch"):
            build_inference_rows(
                rows,
                [self.profile()],
                ["Q1"],
                require_balanced_assignment=False,
            )

    def test_supports_a_complete_provider_subset(self):
        rows = self.results()
        second = []
        for row in rows:
            copied = dict(row)
            copied["call_id"] = f"second-{row['call_id']}"
            copied["provider"] = "second"
            second.append(copied)
        selected = [row for row in rows + second if row["provider"] == "second"]
        direction, utility, audit = build_inference_rows(
            selected,
            [self.profile()],
            ["Q1"],
            require_balanced_assignment=False,
        )
        self.assertEqual(audit["providers"], ["second"])
        self.assertEqual(len(direction), 1)
        self.assertEqual(len(utility), 1)


class FrozenGateCSourceAuditTests(unittest.TestCase):
    def test_frozen_complete_artifact_matches_roster_prompts_and_semantics(self):
        root = Path(__file__).resolve().parents[2]
        results = [
            json.loads(line)
            for line in (
                root / "api_pilot/restricted_outputs/gate_c_v3_complete.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        profiles = [
            json.loads(line)
            for line in (
                root / "dataset/derived/gate_c_profiles_6x12.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        audit = audit_complete_results(
            results,
            profiles,
            root / "api_pilot/prompts/survey-confirmatory-v3.json",
            expected_models=EXPECTED_MODELS,
        )
        self.assertEqual(audit["planned"], 4320)
        self.assertEqual(audit["merged"], 4320)
        self.assertEqual(audit["postprocess_repairs"], 12)
        self.assertEqual(audit["raw_valid_by_provider"]["mistral"], 852)


if __name__ == "__main__":
    unittest.main()
