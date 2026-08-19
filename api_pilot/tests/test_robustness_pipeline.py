from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api_pilot"))

import analyze_robustness_outputs as analyzer
import build_robustness_manifests as builder
import survey_pilot_runner as runner
import validate_robustness_suite as validator


class RobustnessManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = builder.build_suite()

    def test_expected_call_counts_and_official_roster(self) -> None:
        by_experiment = {
            manifest["experiment_id"]: manifest
            for manifest in self.suite.values()
        }
        self.assertEqual(
            {
                name: manifest["expected"]["total_calls"]
                for name, manifest in by_experiment.items()
            },
            {
                "repeat_identical": 288,
                "single_item": 1920,
                "wording_b": 1152,
                "option_reverse": 1152,
            },
        )
        for manifest in by_experiment.values():
            self.assertEqual(
                manifest["provider_models"], builder.OFFICIAL_PROVIDER_MODELS
            )
            builder.validate_manifest(manifest)

    def test_selection_is_balanced_and_deterministic(self) -> None:
        first = builder.build_suite()
        second = builder.build_suite()
        self.assertEqual(first, second)
        for manifest in first.values():
            countries = manifest["selection"]["profiles_by_true_country"]
            labels = manifest["selection"]["profiles_by_displayed_label"]
            self.assertEqual(len(set(countries.values())), 1)
            self.assertEqual(len(set(labels.values())), 1)

    def test_gate_decision_is_pending_without_metrics(self) -> None:
        manifest = {
            "analysis_contract": {
                "gate_margin": 0.05,
                "gate_direction_agreement_lower_bound": 0.75,
                "gate_forecast_tv_overall": True,
                "gate_treatment_contrasts": [],
                "gate_rule": "test",
            }
        }
        result = analyzer.gate_decision(manifest, None)
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["passes"])

    def test_gate_decision_uses_frozen_ci_upper_bounds(self) -> None:
        manifest = {
            "analysis_contract": {
                "gate_margin": 0.05,
                "gate_direction_agreement_lower_bound": 0.75,
                "gate_forecast_tv_overall": True,
                "gate_treatment_contrasts": ["MO_minus_MR"],
                "gate_rule": "test",
            }
        }
        metrics = {
            "forecast_tv": {
                "overall": {"mean": 0.01, "ci95": [0.0, 0.03]}
            },
            "treatment_contrast_stability": {
                "MO_minus_MR": {
                    "mean": 0.02,
                    "ci95": [0.0, 0.04],
                    "direction_agreement_rate": 0.9,
                    "direction_agreement_ci95": [0.8, 0.98],
                }
            },
        }
        result = analyzer.gate_decision(manifest, metrics)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["passes"])
        metrics["forecast_tv"]["overall"]["ci95"][1] = 0.05
        result = analyzer.gate_decision(manifest, metrics)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["passes"])

    def test_gate_rejects_low_direction_agreement_lower_bound(self) -> None:
        manifest = {
            "analysis_contract": {
                "gate_margin": 0.05,
                "gate_direction_agreement_lower_bound": 0.75,
                "gate_forecast_tv_overall": False,
                "gate_treatment_contrasts": ["MO_minus_MR"],
                "gate_rule": "test",
            }
        }
        metrics = {
            "forecast_tv": {"overall": {"mean": 0.0, "ci95": [0.0, 0.0]}},
            "treatment_contrast_stability": {
                "MO_minus_MR": {
                    "mean": 0.01,
                    "ci95": [0.0, 0.02],
                    "direction_agreement_rate": 0.8,
                    "direction_agreement_ci95": [0.70, 0.90],
                }
            },
        }
        result = analyzer.gate_decision(manifest, metrics)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["passes"])

    def test_repeat_units_bind_to_prompt_identical_core_calls(self) -> None:
        manifest = next(
            value
            for value in self.suite.values()
            if value["experiment_id"] == "repeat_identical"
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "api_pilot" / "tests") as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = runner.load_task_manifest(path)
            profiles = runner.load_profiles(builder.PROFILE_PATH, 12)
            profiles = runner.apply_task_manifest(profiles, loaded)
            bundle = runner.load_prompt_bundle(builder.PROMPT_A_PATH)
            tasks = runner.build_tasks(
                profiles,
                list(builder.OFFICIAL_PROVIDER_MODELS),
                builder.OFFICIAL_PROVIDER_MODELS,
                bundle=bundle,
                target_batch_size=5,
                cells=loaded["parsed_cells"],
                target_batch_ids=tuple(loaded["target_batches"]),
                experiment_id=loaded["experiment_id"],
            )
            tasks = runner.apply_task_units(tasks, loaded)
        self.assertEqual(len(tasks), 288)
        for task in tasks:
            self.assertEqual(task["source_call_id"], task["call_id"].removesuffix("/ROBUST-REPEAT-01"))
            self.assertTrue(task["call_id"].endswith("/ROBUST-REPEAT-01"))

    def test_single_item_manifest_covers_core_treatments(self) -> None:
        manifest = next(
            value
            for value in self.suite.values()
            if value["experiment_id"] == "single_item"
        )
        self.assertEqual(set(manifest["single_item_batch_to_item"]), {
            f"B{index:02d}" for index in range(1, 11)
        })
        self.assertEqual(
            {tuple(cell) for cell in manifest["cells"]},
            {
                ("M0", "correct"), ("MO", "correct"),
                ("MR", "correct"), ("MV", "correct"),
            },
        )

    def test_dry_run_validates_all_manifests(self) -> None:
        builder.main()
        report = validator.validate_suite(
            ROOT / "dataset" / "analysis" / "gate_c_robustness_suite_v2.json"
        )
        self.assertEqual(report["status"], "valid; no API request made")
        self.assertEqual(report["total_planned_calls"], 4512)
        self.assertTrue(all(row["valid"] for row in report["experiments"]))


class RobustnessAnalysisTests(unittest.TestCase):
    def test_single_item_pairing_survives_five_model_experiment_id(self) -> None:
        manifest = {
            "experiment_id": "single_item_five_model",
            "task_mode": "standard",
            "single_item_batch_to_item": {"S01": "Q1"},
            "item_to_core_batch": {"Q1": "B01"},
        }
        row = {
            "provider": "p", "profile_id": "x", "condition": "M0",
            "evidence_variant": "correct", "batch_id": "S01",
            "target_ids": ["Q1"], "target_option_codes": {"Q1": ["0", "1"]},
            "parsed_forecasts": {"Q1": [0.4, 0.6]},
        }
        core = {
            **row, "batch_id": "B01",
            "parsed_forecasts": {"Q1": [0.5, 0.5]},
        }
        pairs = analyzer.paired_rows(
            manifest,
            {("p", "x", "M0", "correct", "S01"): row},
            {},
            {("p", "x", "M0", "correct", "B01"): core},
            {"x": {"true_country_code": "US", "analysis_weight": 1.0}},
        )
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["tv"], 0.1)

    def test_macro_uses_survey_weights_with_equal_country_provider_unit(self) -> None:
        rows = []
        for provider in ("p1", "p2"):
            for country in ("CN", "US"):
                for item in ("Q1", "Q2"):
                    rows.extend(
                        [
                            {
                                "provider": provider,
                                "profile_id": f"{country}-a",
                                "true_country_code": country,
                                "item_id": item,
                                "macro_unit": item,
                                "analysis_weight": 9.0,
                                "value": 1.0,
                            },
                            {
                                "provider": provider,
                                "profile_id": f"{country}-b",
                                "true_country_code": country,
                                "item_id": item,
                                "macro_unit": item,
                                "analysis_weight": 1.0,
                                "value": 0.0,
                            },
                        ]
                    )
        self.assertAlmostEqual(analyzer._balanced_macro(rows, "value"), 0.9)

    def test_option_code_alignment_makes_reversal_invariant(self) -> None:
        first = {
            "target_option_codes": {"Q": ["0", "1"]},
            "parsed_forecasts": {"Q": [0.2, 0.8]},
        }
        reversed_row = {
            "target_option_codes": {"Q": ["1", "0"]},
            "parsed_forecasts": {"Q": [0.8, 0.2]},
        }
        self.assertEqual(
            analyzer.total_variation(
                analyzer.probability_map(first, "Q"),
                analyzer.probability_map(reversed_row, "Q"),
            ),
            0.0,
        )

    def test_treatment_contrast_discrepancy(self) -> None:
        rows = []
        for condition, core, variant in (
            ("MO", {"0": 0.8, "1": 0.2}, {"0": 0.7, "1": 0.3}),
            ("MR", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}),
        ):
            rows.append(
                {
                    "provider": "p", "profile_id": "x", "true_country_code": "CN",
                    "analysis_weight": 1.0,
                    "evidence_variant": "correct", "item_id": "Q",
                    "condition": condition, "core": core, "variant": variant,
                    "tv": analyzer.total_variation(core, variant),
                }
            )
        result = analyzer.treatment_contrasts(rows)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["contrast_discrepancy"], 0.1)
        self.assertTrue(result[0]["direction_agreement"])

    def test_pending_analysis_requires_no_output_file(self) -> None:
        builder.main()
        source_manifest = ROOT / "dataset" / "analysis" / "gate_c_repeat_identical_manifest_v2.json"
        with tempfile.TemporaryDirectory(dir=ROOT / "api_pilot" / "tests") as directory:
            missing_output = Path(directory) / "missing_robustness_output.jsonl"
            manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            manifest["immutable_output_path"] = str(missing_output)
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = analyzer.analyse(
                manifest_path,
                missing_output,
                builder.CORE_OUTPUT_PATH,
                bootstrap_replicates=20,
            )
        self.assertEqual(result["status"], "pending_inference")
        self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
