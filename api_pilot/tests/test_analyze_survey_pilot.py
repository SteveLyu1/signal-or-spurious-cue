from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api_pilot"))

import analyze_survey_pilot as analysis


class ScoringTests(unittest.TestCase):
    def test_brier_scores_deterministic_gold(self) -> None:
        self.assertAlmostEqual(analysis.brier([0.2, 0.5, 0.3], 1), 0.38)
        self.assertAlmostEqual(analysis.brier([0.0, 1.0, 0.0], 1), 0.0)
        self.assertAlmostEqual(analysis.brier([1.0, 0.0, 0.0], 1), 2.0)

    def test_argmax_accuracy_splits_credit_for_ties(self) -> None:
        self.assertEqual(analysis.argmax_accuracy([0.2, 0.5, 0.3], 1), 1.0)
        self.assertEqual(analysis.argmax_accuracy([0.4, 0.4, 0.2], 1), 0.5)

    def test_ordinal_crps_penalises_distant_errors_more(self) -> None:
        near = analysis.ordinal_crps([0.0, 0.0, 0.0, 1.0, 0.0], 2)
        far = analysis.ordinal_crps([0.0, 0.0, 0.0, 0.0, 1.0], 2)
        self.assertLess(near, far)
        self.assertEqual(analysis.ordinal_crps([0.0, 0.0, 1.0, 0.0, 0.0], 2), 0.0)

    def test_direction_beta_uses_within_country_survey_weights(self) -> None:
        rows = [
            {
                "item_id": "Q1",
                "true_country_code": "CN",
                "analysis_weight": 1.0,
                "projection": 1.0,
                "direction_norm_sq": 1.0,
            },
            {
                "item_id": "Q1",
                "true_country_code": "CN",
                "analysis_weight": 3.0,
                "projection": 3.0,
                "direction_norm_sq": 1.0,
            },
            {
                "item_id": "Q1",
                "true_country_code": "US",
                "analysis_weight": 1.0,
                "projection": 2.0,
                "direction_norm_sq": 1.0,
            },
        ]
        # CN contributes (.25 * 1 + .75 * 3)=2.5, US contributes 2. Both
        # countries have unit denominator weight, so the ratio is 4.5/2.
        self.assertAlmostEqual(analysis.direction_beta(rows, "projection"), 2.25)

    def test_direction_beta_is_invariant_to_country_weight_rescaling(self) -> None:
        rows = [
            {
                "item_id": "Q1",
                "true_country_code": country,
                "analysis_weight": weight * scale,
                "projection": projection,
                "direction_norm_sq": norm,
            }
            for country, scale, weight, projection, norm in (
                ("CN", 10.0, 1.0, 1.0, 2.0),
                ("CN", 10.0, 3.0, 3.0, 1.0),
                ("US", 0.1, 1.0, 2.0, 1.0),
            )
        ]
        rescaled = [dict(row) for row in rows]
        for row in rescaled:
            if row["true_country_code"] == "CN":
                row["analysis_weight"] *= 100
        self.assertAlmostEqual(
            analysis.direction_beta(rows, "projection"),
            analysis.direction_beta(rescaled, "projection"),
        )


class BatchedResultTests(unittest.TestCase):
    @staticmethod
    def row(batch_id: str, item_id: str, value: float) -> dict:
        return {
            "provider": "model-a",
            "profile_id": "C-CN-001",
            "condition": "M0",
            "evidence_variant": "correct",
            "batch_id": batch_id,
            "target_ids": [item_id],
            "parsed_forecasts": {item_id: [value, 1 - value]},
        }

    def test_four_batches_for_one_profile_are_not_overwritten(self) -> None:
        rows = [
            self.row("B01", "Q1", 0.1),
            self.row("B02", "Q2", 0.2),
            self.row("B03", "Q3", 0.3),
            self.row("B04", "Q4", 0.4),
        ]

        indexed = analysis.index_result_batches(rows)

        self.assertEqual(len(indexed), 4)
        item_batches = {f"Q{index}": f"B{index:02d}" for index in range(1, 5)}
        for index in range(1, 5):
            item_id = f"Q{index}"
            result = analysis.result_for_item(
                indexed,
                "model-a",
                "C-CN-001",
                "M0",
                "correct",
                item_id,
                item_batches,
            )
            self.assertEqual(result["batch_id"], f"B{index:02d}")
            self.assertAlmostEqual(result["parsed_forecasts"][item_id][0], index / 10)

    def test_manifest_batch_mismatch_fails_instead_of_falling_back(self) -> None:
        indexed = analysis.index_result_batches([self.row("B01", "Q1", 0.1)])
        with self.assertRaisesRegex(ValueError, "Missing result"):
            analysis.result_for_item(
                indexed,
                "model-a",
                "C-CN-001",
                "M0",
                "correct",
                "Q2",
                {"Q2": "B02"},
            )

    def test_prompt_identical_repeat_does_not_replace_source_cell(self) -> None:
        source = self.row("B01", "Q1", 0.1)
        source["call_id"] = "model-a/model/C-CN-001/B01/M0/correct"
        repeated = self.row("B01", "Q1", 0.9)
        repeated["call_id"] = source["call_id"] + "/REPEAT-01"

        indexed = analysis.index_result_batches([source, repeated])

        self.assertEqual(len(indexed), 1)
        self.assertEqual(next(iter(indexed.values()))["parsed_forecasts"]["Q1"], [0.1, 0.9])

    def repeat_pair(self) -> tuple[dict, dict]:
        source = self.row("B01", "Q1", 0.2)
        source.update(
            {
                "call_id": "model-a/model/C-CN-001/B01/M0/correct/digest",
                "model_requested": "model",
                "prompt_version": "v3",
                "true_country_code": "CN",
                "displayed_country_code": None,
                "prompt_sha256": "same-prompt",
                "target_option_codes": {"Q1": ["0", "1"]},
            }
        )
        repeated = dict(source)
        repeated["call_id"] = source["call_id"] + "/REPEAT-01"
        repeated["experiment"] = "repeat"
        repeated["source_call_id"] = source["call_id"]
        repeated["parsed_forecasts"] = {"Q1": [0.5, 0.5]}
        return source, repeated

    def test_prompt_repeat_noise_pairs_source_and_reports_counts(self) -> None:
        source, repeated = self.repeat_pair()

        summary = analysis.summarize_prompt_repeat_noise([source, repeated])

        self.assertTrue(summary["available"])
        self.assertEqual(summary["overall"]["pair_count"], 1)
        self.assertEqual(summary["overall"]["item_comparison_count"], 1)
        self.assertAlmostEqual(summary["overall"]["mean_item_tv"], 0.3)
        self.assertEqual(summary["by_provider"]["model-a"]["pair_count"], 1)
        self.assertEqual(summary["by_cell"]["M0/correct"]["pair_count"], 1)
        self.assertEqual(
            summary["by_provider_cell"]["model-a"]["M0/correct"]["pair_count"],
            1,
        )

    def test_prompt_repeat_noise_works_with_suffix_only(self) -> None:
        source, repeated = self.repeat_pair()
        repeated.pop("experiment")
        repeated.pop("source_call_id")

        summary = analysis.summarize_prompt_repeat_noise([source, repeated])

        self.assertEqual(summary["overall"]["pair_count"], 1)
        self.assertAlmostEqual(summary["overall"]["mean_item_tv"], 0.3)

    def test_prompt_repeat_noise_fails_when_source_is_missing(self) -> None:
        _, repeated = self.repeat_pair()
        with self.assertRaisesRegex(ValueError, "no successful source"):
            analysis.summarize_prompt_repeat_noise([repeated])

    def test_prompt_repeat_noise_fails_on_metadata_mismatch(self) -> None:
        source, repeated = self.repeat_pair()
        repeated["batch_id"] = "B02"
        with self.assertRaisesRegex(ValueError, "mismatch for batch_id"):
            analysis.summarize_prompt_repeat_noise([source, repeated])

    def test_prompt_repeat_noise_fails_on_conflicting_source_id(self) -> None:
        source, repeated = self.repeat_pair()
        repeated["source_call_id"] = source["call_id"] + "/other"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            analysis.summarize_prompt_repeat_noise([source, repeated])

    def test_prompt_repeat_noise_empty_schema_is_explicit(self) -> None:
        source, _ = self.repeat_pair()
        summary = analysis.summarize_prompt_repeat_noise([source])
        self.assertFalse(summary["available"])
        self.assertEqual(summary["overall"]["pair_count"], 0)
        self.assertIsNone(summary["overall"]["mean_item_tv"])

    def test_analysis_design_loads_frozen_strata_and_batches(self) -> None:
        manifest = {
            "target_strata": {
                "high_country_value": ["Q1"],
                "representative": ["Q2", "Q3"],
                "equivalence_controls": ["Q4"],
            },
            "target_batches": {
                "B01": ["Q1"],
                "B02": ["Q2"],
                "B03": ["Q3"],
                "B04": ["Q4"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            item_strata, item_batches, strata, source = analysis.analysis_design(
                {"Q1", "Q2", "Q3", "Q4"}, path
            )

        self.assertEqual(item_strata["Q1"], "high_country_value")
        self.assertEqual(item_strata["Q4"], "equivalence_controls")
        self.assertEqual(item_batches["Q3"], "B03")
        self.assertEqual(
            strata,
            ("high_country_value", "representative", "equivalence_controls"),
        )
        self.assertEqual(source, str(path))


if __name__ == "__main__":
    unittest.main()
