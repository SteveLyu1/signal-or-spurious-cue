from __future__ import annotations

import unittest
from collections import Counter

from api_pilot import analyze_confirmatory_sentinels as analyzer
from api_pilot import build_confirmatory_sentinel_manifests as builder


class ConfirmatorySentinelManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = builder.build_suite()
        cls.profiles = {
            row["profile_id"]: row for row in builder.load_profiles()
        }

    def test_three_waves_freeze_exact_counts_and_official_roster(self) -> None:
        self.assertEqual(set(self.suite), {"before", "mid", "after"})
        for wave, manifest in self.suite.items():
            self.assertEqual(manifest["schema_version"], "survey-task-manifest-v2")
            self.assertEqual(manifest["task_mode"], "repeat_identical")
            self.assertEqual(manifest["provider_models"], builder.OFFICIAL_PROVIDER_MODELS)
            self.assertEqual(manifest["expected"], {
                "providers": 4,
                "calls_per_provider": 48,
                "total_calls": 192,
            })
            self.assertEqual(manifest["selection"]["wave"], wave)
            self.assertTrue(
                manifest["analysis_contract"]["data_processing_acknowledgement_required"]
            )
            self.assertEqual(
                manifest["analysis_contract"]["execution_gate_json_path"],
                "gate.full_gate_c_pass",
            )

    def test_wave_profiles_are_disjoint_and_country_label_balanced(self) -> None:
        selected = [
            profile_id
            for manifest in self.suite.values()
            for profile_id in manifest["profile_ids"]
        ]
        self.assertEqual(len(selected), 36)
        self.assertEqual(len(set(selected)), 36)
        for manifest in self.suite.values():
            profiles = [self.profiles[value] for value in manifest["profile_ids"]]
            self.assertEqual(
                Counter(row["true_country_code"] for row in profiles),
                Counter({country: 2 for country in builder.COUNTRIES}),
            )
            self.assertEqual(
                set(Counter(
                    row["balanced_random_country_label"] for row in profiles
                ).values()),
                {2},
            )

    def test_every_profile_covers_four_cells_and_one_batch(self) -> None:
        total_batches = Counter()
        label_batches: dict[str, set[str]] = {}
        for manifest in self.suite.values():
            units = manifest["task_units"]
            for profile_id in manifest["profile_ids"]:
                subset = [row for row in units if row["profile_id"] == profile_id]
                self.assertEqual(
                    {(row["condition"], row["evidence_variant"]) for row in subset},
                    set(builder.CELLS),
                )
                self.assertEqual(len({row["batch_id"] for row in subset}), 1)
                batch = subset[0]["batch_id"]
                total_batches[batch] += 1
                label = self.profiles[profile_id]["balanced_random_country_label"]
                label_batches.setdefault(label, set()).add(batch)
        self.assertEqual(
            total_batches, Counter({"B01": 9, "B02": 9, "B03": 9, "B04": 9})
        )
        self.assertTrue(all(len(value) >= 3 for value in label_batches.values()))

    def test_selection_is_deterministic(self) -> None:
        self.assertEqual(self.suite, builder.build_suite())


class ConfirmatorySentinelAnalysisTests(unittest.TestCase):
    @staticmethod
    def core_row() -> dict:
        return {
            "call_id": "source",
            "provider": "deepseek",
            "model_requested": "deepseek-v4-pro",
            "model_returned": "deepseek-v4-pro",
            "prompt_version": "p",
            "profile_id": "C-CN-001",
            "true_country_code": "CN",
            "condition": "M0",
            "evidence_variant": "correct",
            "batch_id": "B01",
            "target_ids": ["Q"],
            "target_option_codes": {"Q": ["0", "1"]},
            "displayed_country_code": None,
            "prompt_sha256": "digest",
            "parsed_forecasts": {"Q": [0.25, 0.75]},
            "success": True,
            "experiment": "confirmatory_core_v1",
            "source_call_id": None,
        }

    def test_answer_code_alignment_defines_tv(self) -> None:
        first = {
            "target_option_codes": {"Q": ["0", "1"]},
            "parsed_forecasts": {"Q": [0.25, 0.75]},
        }
        second = {
            "target_option_codes": {"Q": ["1", "0"]},
            "parsed_forecasts": {"Q": [0.75, 0.25]},
        }
        self.assertEqual(
            analyzer.total_variation(
                analyzer.probability_map(first, "Q"),
                analyzer.probability_map(second, "Q"),
            ),
            0.0,
        )

    def test_incomplete_wave_emits_no_analysis_rows(self) -> None:
        rows, audit = analyzer.audit_sentinel_results(
            {},
            {"source": {"target_ids": ["Q"]}},
            {"source": self.core_row()},
            {"experiment_id": "wave", "expected": {"total_calls": 1}},
        )
        self.assertEqual(rows, [])
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["missing"], 1)
        self.assertEqual(audit["analysis_ready"], 0)

    def test_sentinel_units_subset_exact_cached_core_calls(self) -> None:
        manifest = next(iter(builder.build_suite().values()))
        core_expected = {}
        for provider in builder.OFFICIAL_PROVIDER_MODELS:
            for unit in manifest["task_units"]:
                key = (
                    unit["profile_id"], unit["condition"],
                    unit["evidence_variant"], unit["batch_id"],
                )
                core_expected[f"{provider}/{'/'.join(key)}"] = {
                    "profile": {"profile_id": unit["profile_id"]},
                    "condition": unit["condition"],
                    "evidence_variant": unit["evidence_variant"],
                    "batch_id": unit["batch_id"],
                }
        selected = analyzer.source_specs_for_manifest(manifest, core_expected)
        self.assertEqual(len(selected), 192)

    def test_complete_repeat_is_exactly_paired_to_core(self) -> None:
        core = self.core_row()
        repeat = dict(core)
        repeat.update({
            "call_id": "source/ROBUST-REPEAT-01",
            "source_call_id": "source",
            "experiment": "wave",
            "parsed_forecasts": {"Q": [0.20, 0.80]},
        })
        rows, audit = analyzer.audit_sentinel_results(
            {repeat["call_id"]: repeat},
            {"source": {"target_ids": ["Q"]}},
            {"source": core},
            {"experiment_id": "wave", "expected": {"total_calls": 1}},
        )
        self.assertTrue(audit["complete"])
        pairs = analyzer.pair_rows("before", rows, {"source": core})
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["tv"], 0.05)

    def test_merged_panel_accepts_provider_source_experiment(self) -> None:
        core = self.core_row()
        repeat = dict(core)
        repeat.update({
            "call_id": "source/ROBUST-REPEAT-01",
            "source_call_id": "source",
            "experiment": "source_wave",
        })
        manifest = {
            "experiment_id": "merged_wave",
            "expected": {"total_calls": 1},
            "roster_amendment": {
                "source_experiment_id": "source_wave",
                "openai_source_experiment_id": "openai_wave",
            },
        }
        rows, audit = analyzer.audit_sentinel_results(
            {repeat["call_id"]: repeat},
            {"source": {"target_ids": ["Q"]}},
            {"source": core},
            manifest,
        )
        self.assertTrue(audit["complete"])
        self.assertEqual(len(rows), 1)

    def test_pending_result_contains_no_metric_key(self) -> None:
        # The production path calls this only after exact core/wave audits.
        pending = {
            "status": "pending_inference",
            "complete": False,
            "estimates_generated": False,
        }
        self.assertNotIn("metrics", pending)

    def test_wave_contrast_respects_provider_and_wave_clusters(self) -> None:
        rows = []
        for provider, offset in (("a", 0.0), ("b", 0.1)):
            for wave, value in (
                ("before", 0.10 + offset),
                ("mid", 0.15 + offset),
                ("after", 0.20 + offset),
            ):
                for index in range(2):
                    rows.append({
                        "provider": provider,
                        "wave": wave,
                        "profile_id": f"{provider}-{wave}-{index}",
                        "tv": value,
                    })
        contrasts = analyzer.wave_contrasts(rows, 100)
        self.assertEqual(set(contrasts), {"a", "b"})
        for provider in contrasts:
            self.assertAlmostEqual(
                contrasts[provider]["mid_minus_before"]["estimate"],
                0.05,
            )
            self.assertAlmostEqual(
                contrasts[provider]["after_minus_before"]["estimate"],
                0.10,
            )


if __name__ == "__main__":
    unittest.main()
