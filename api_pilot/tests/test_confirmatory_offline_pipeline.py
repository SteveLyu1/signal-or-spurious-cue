from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from api_pilot import build_auxiliary_manifests as auxiliary_builder
from api_pilot import build_robustness_manifests as robustness_builder
from api_pilot.build_confirmatory_core_manifest import build_manifest
from api_pilot.run_confirmatory_strict_inference import (
    audit_results,
    normalize_forecasts,
)
from api_pilot.survey_pilot_runner import (
    execution_gate_error,
    load_task_manifest,
    validate_v2_immutable_paths,
    validate_v2_source_bindings,
)


class ConfirmatoryCoreManifestTests(unittest.TestCase):
    def test_manifest_freezes_exact_four_provider_panel(self):
        manifest = build_manifest()
        self.assertEqual(manifest["schema_version"], "survey-task-manifest-v2")
        self.assertEqual(len(manifest["profile_ids"]), 504)
        self.assertEqual(len(manifest["task_units"]), 8_064)
        self.assertEqual(manifest["expected"]["calls_per_provider"], 8_064)
        self.assertEqual(manifest["expected"]["total_calls"], 32_256)
        self.assertEqual(
            manifest["immutable_checkpoint_path"],
            "api_pilot/restricted_outputs/confirmatory_core_v2_raw.checkpoint.sqlite3",
        )
        self.assertEqual(
            set(manifest["provider_models"]),
            {"deepseek", "gemini", "mistral", "qwen"},
        )
        self.assertEqual(
            Counter(unit["batch_id"] for unit in manifest["task_units"]),
            Counter({"B01": 2016, "B02": 2016, "B03": 2016, "B04": 2016}),
        )
        self.assertEqual(
            {tuple(cell) for cell in manifest["cells"]},
            {(condition, "correct") for condition in ("M0", "MO", "MR", "MV")},
        )
        self.assertEqual(len(manifest["logical_cells"]), 8)

    def test_formal_execution_gate_is_fail_closed_while_gate_c_is_open(self):
        root = Path(__file__).resolve().parents[2]
        manifest = load_task_manifest(
            root / "dataset/analysis/confirmatory_core_task_manifest_v1.json"
        )
        self.assertEqual(
            execution_gate_error(manifest),
            "frozen aggregate Gate C status is not open",
        )

    def test_every_version_two_manifest_freezes_checkpoint_path(self):
        manifests = [build_manifest()]
        manifests.extend(robustness_builder.build_suite().values())
        manifests.extend(auxiliary_builder.build_suite().values())
        for manifest in manifests:
            self.assertEqual(
                manifest["immutable_checkpoint_path"],
                str(
                    Path(manifest["immutable_output_path"]).with_suffix(
                        ".checkpoint.sqlite3"
                    )
                ).replace("\\", "/"),
            )

    def test_immutable_output_and_checkpoint_paths_are_fail_closed(self):
        root = Path(__file__).resolve().parents[2]
        manifest = build_manifest()
        output = root / manifest["immutable_output_path"]
        checkpoint = root / manifest["immutable_checkpoint_path"]
        validate_v2_immutable_paths(manifest, output, checkpoint)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Checkpoint path"):
                validate_v2_immutable_paths(
                    manifest, output, Path(directory) / "wrong.sqlite3"
                )

    def test_version_two_source_binding_requires_exact_analysis_manifest(self):
        root = Path(__file__).resolve().parents[2]
        manifest = build_manifest()
        with self.assertRaisesRegex(ValueError, "requires --analysis-manifest"):
            validate_v2_source_bindings(manifest, None)
        with self.assertRaisesRegex(ValueError, "Analysis manifest path"):
            validate_v2_source_bindings(
                manifest, root / "dataset/analysis/gate_c_manifest_v1.json"
            )
        validate_v2_source_bindings(
            manifest, root / manifest["source"]["analysis_manifest_file"]
        )


class ConfirmatoryAuxiliaryManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = auxiliary_builder.build_suite()

    def test_exact_call_counts_and_official_models(self):
        by_experiment = {
            manifest["experiment_id"]: manifest
            for manifest in self.suite.values()
        }
        self.assertEqual(
            {
                name: value["expected"]["total_calls"]
                for name, value in by_experiment.items()
            },
            {
                "confirmatory_mechanism_v1": 2_304,
                "confirmatory_wording_b_v1": 1_152,
            },
        )
        for manifest in by_experiment.values():
            self.assertEqual(
                manifest["provider_models"],
                auxiliary_builder.OFFICIAL_PROVIDER_MODELS,
            )
            self.assertEqual(manifest["schema_version"], "survey-task-manifest-v2")
            auxiliary_builder.validate_manifest(manifest)

    def test_selection_balances_true_country_by_displayed_label(self):
        profiles = auxiliary_builder.select_profiles(
            auxiliary_builder.load_profiles()
        )
        counts = Counter(
            (row["true_country_code"], row["balanced_random_country_label"])
            for row in profiles
        )
        self.assertEqual(len(counts), 36)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(
            Counter(row["true_country_code"] for row in profiles),
            Counter({country: 12 for country in auxiliary_builder.COUNTRIES}),
        )

    def test_auxiliary_manifests_bind_sources_and_closed_gate(self):
        root = Path(__file__).resolve().parents[2]
        for manifest in self.suite.values():
            validate_v2_source_bindings(
                manifest, root / manifest["source"]["analysis_manifest_file"]
            )
            self.assertEqual(
                execution_gate_error(manifest),
                "frozen aggregate Gate C status is not open",
            )

    def test_auxiliary_units_are_complete_and_unique(self):
        expected = {
            "confirmatory_mechanism_v1": (576, {"B01", "B02", "B03", "B04"}),
            "confirmatory_wording_b_v1": (288, {"B01", "B02"}),
        }
        for manifest in self.suite.values():
            unit_count, batches = expected[manifest["experiment_id"]]
            keys = {
                (
                    unit["profile_id"],
                    unit["condition"],
                    unit["evidence_variant"],
                    unit["batch_id"],
                )
                for unit in manifest["task_units"]
            }
            self.assertEqual(len(keys), unit_count)
            self.assertEqual({unit["batch_id"] for unit in manifest["task_units"]}, batches)


class ConfirmatoryResultAuditTests(unittest.TestCase):
    def test_offline_union_uses_declared_source_experiment_id(self):
        task_manifest = {
            "experiment_id": "core_five_model",
            "roster_amendment": {
                "source_experiment_id": "core",
                "openai_source_experiment_id": "core_openai_parity",
            },
        }
        self.assertEqual(
            (task_manifest.get("roster_amendment") or {}).get(
                "source_experiment_id", task_manifest["experiment_id"]
            ),
            "core",
        )

    def test_probability_rounding_is_normalized_only_within_one_percent(self):
        expected = {"target_ids": ["Q"]}
        row = {
            "call_id": "x",
            "parsed_forecasts": {"Q": [0.33, 0.66]},
            "target_option_codes": {"Q": ["0", "1"]},
        }
        normalized, changed = normalize_forecasts(row, expected)
        self.assertEqual(changed, 1)
        self.assertAlmostEqual(sum(normalized["parsed_forecasts"]["Q"]), 1.0)
        self.assertEqual(row["parsed_forecasts"]["Q"], [0.33, 0.66])

    def test_probability_error_above_one_percent_fails(self):
        expected = {"target_ids": ["Q"]}
        row = {
            "call_id": "x",
            "parsed_forecasts": {"Q": [0.30, 0.60]},
            "target_option_codes": {"Q": ["0", "1"]},
        }
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            normalize_forecasts(row, expected)

    def test_absent_results_are_pending_without_hypothesis_rows(self):
        rows, audit = audit_results({}, {"call-a": {}, "call-b": {}})
        self.assertEqual(rows, [])
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["missing"], 2)
        self.assertEqual(audit["analysis_ready"], 0)


if __name__ == "__main__":
    unittest.main()
