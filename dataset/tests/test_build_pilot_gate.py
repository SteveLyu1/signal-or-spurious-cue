from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset" / "tools"))

import build_pilot_gate as gate


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_valid(self) -> None:
        first = gate.respondent_split("synthetic-id")
        self.assertEqual(first, gate.respondent_split("synthetic-id"))
        self.assertIn(first, {name for name, _ in gate.SPLIT_BOUNDS})

    def test_fold_is_deterministic_and_bounded(self) -> None:
        fold = gate.respondent_fold("synthetic-id")
        self.assertEqual(fold, gate.respondent_fold("synthetic-id"))
        self.assertGreaterEqual(fold, 0)
        self.assertLess(fold, gate.N_FOLDS)


class SelectionTests(unittest.TestCase):
    def test_controls_must_have_null_compatible_country_gain(self) -> None:
        rows = []
        for index in range(6):
            rows.append(
                {
                    "variable": f"V{index}",
                    "evidence_gain_ci_low": 0.01,
                    "country_gain_ci_low": 0.01 if index < 4 else -0.01,
                    "country_gain_ci_high": 0.03 if index < 4 else 0.01,
                    "country_incremental_gain": 0.02 if index < 4 else 0.0,
                    "reference_country_js_bits": 0.1 if index < 4 else 0.001,
                }
            )
        high, controls = gate.select_targets(rows)
        self.assertEqual(high, ["V0", "V1", "V2", "V3"])
        self.assertEqual(controls, ["V4", "V5"])

    def test_survey_weight_falls_back_for_invalid_values(self) -> None:
        self.assertEqual(gate.survey_weight({"gwght": ""}), 1.0)
        self.assertEqual(gate.survey_weight({"gwght": "-2"}), 1.0)
        self.assertEqual(gate.survey_weight({"gwght": "2.5"}), 2.5)


if __name__ == "__main__":
    unittest.main()
