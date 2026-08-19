from __future__ import annotations

import unittest

import numpy as np

from api_pilot.confirmatory_power import (
    _family_design_power,
    _holm_two_family_rejections,
    _marginal_sensitivity,
    _stratified_profile_indexes,
    simulate_confirmatory_sensitivity,
    simulate_prospective_macro_design_power,
)
from api_pilot.strict_inference import ReferenceAnchorResampler, _DirectionPanel


PROVIDERS = ("model-a", "model-b")
COUNTRIES = ("C1", "C2")
ITEMS = ("Q1",)


def reference_rows():
    rows = []
    for country, answers in {"C1": [0, 0, 0, 0], "C2": [1, 1, 1, 1]}.items():
        for index, answer in enumerate(answers):
            rows.append(
                {
                    "country": country,
                    "profile_id": f"{country}-R{index}",
                    "item_id": "Q1",
                    "answer_index": answer,
                    "n_options": 2,
                    "weight": 1.0 + index,
                }
            )
    return rows


def inference_rows():
    direction = []
    utility = []
    vectors = {"C1": np.array([0.5, -0.5]), "C2": np.array([-0.5, 0.5])}
    for provider_index, provider in enumerate(PROVIDERS):
        for country in COUNTRIES:
            for index, label in enumerate(COUNTRIES):
                profile_id = f"{country}-P{index}"
                common = {
                    "provider": provider,
                    "country": country,
                    "profile_id": profile_id,
                    "item_id": "Q1",
                    "weight": 1.0 + index,
                }
                direction.append(
                    {
                        **common,
                        "displayed_label": label,
                        "mo_delta": (
                            (0.15 + 0.01 * provider_index) * vectors[label]
                        ).tolist(),
                        "mr_delta": (0.04 * vectors[label]).tolist(),
                    }
                )
                utility.append(
                    {**common, "value": 0.01 * (index + 1 + provider_index)}
                )
    return direction, utility


class ConfirmatorySensitivityTests(unittest.TestCase):
    def setUp(self):
        self.resampler = ReferenceAnchorResampler(
            reference_rows(),
            expected_countries=COUNTRIES,
            expected_items=ITEMS,
        )
        self.anchors = self.resampler.estimate()
        self.direction, self.utility = inference_rows()

    def test_resampling_retains_original_labels_and_exact_cell_counts(self):
        panel = _DirectionPanel(self.direction, self.anchors)
        indexes = _stratified_profile_indexes(
            panel, np.random.default_rng(4), 6
        )
        for country in COUNTRIES:
            sampled_labels = panel.displayed[country][indexes[country]]
            self.assertEqual(len(indexes[country]), 6)
            self.assertEqual(np.bincount(sampled_labels).tolist(), [3, 3])

    def test_point_effect_simulation_reports_provider_and_macro_diagnostics(self):
        result = simulate_confirmatory_sensitivity(
            self.direction,
            self.utility,
            self.anchors,
            anchor_bootstrap=self.resampler,
            profiles_per_country=6,
            calibration_replicates=120,
            sensitivity_replicates=120,
            expected_providers=PROVIDERS,
            expected_countries=COUNTRIES,
            expected_items=ITEMS,
        )
        self.assertEqual(result.profiles_total, 12)
        self.assertEqual(result.planned_profiles_per_country_label, 3)
        self.assertEqual(
            set(result.results), {"model-a", "model-b", "macro_equal_provider"}
        )
        expected_lowest = min(
            result.results[provider].minimum_co_primary_power
            for provider in PROVIDERS
        )
        self.assertAlmostEqual(
            result.lowest_provider_diagnostic_sensitivity, expected_lowest
        )
        self.assertEqual(
            result.all_provider_diagnostics_reach_80_percent,
            expected_lowest >= 0.80,
        )

    def test_holm_two_family_step_down(self):
        h1, h2 = _holm_two_family_rejections(
            np.array([0.01, 0.06, 0.01, 0.20]),
            np.array([0.04, 0.01, 0.20, 0.01]),
            alpha=0.05,
        )
        np.testing.assert_array_equal(h1, [True, False, True, False])
        np.testing.assert_array_equal(h2, [True, True, False, True])

    def test_family_power_preserves_iut_and_holm_decisions(self):
        rng = np.random.default_rng(78)
        calibration = rng.normal(scale=0.01, size=(5_000, 3))
        evaluation = rng.normal(scale=0.01, size=(5_000, 3))
        result = _family_design_power(
            calibration,
            evaluation,
            np.array([0.08, 0.08, 0.05]),
            alpha=0.05,
        )
        self.assertGreater(result.joint_holm_power, 0.95)
        self.assertLessEqual(result.joint_holm_wilson_low, result.joint_holm_power)
        self.assertGreaterEqual(result.joint_holm_wilson_high, result.joint_holm_power)
        self.assertLessEqual(result.joint_holm_power, result.h1_holm_power)
        self.assertLessEqual(result.joint_holm_power, result.h2_holm_power)

    def test_prospective_power_gates_only_equal_provider_macro(self):
        result = simulate_prospective_macro_design_power(
            self.direction,
            self.utility,
            self.anchors,
            anchor_bootstrap=self.resampler,
            profiles_per_country=6,
            calibration_replicates=400,
            power_replicates=400,
            familywise_alpha=0.05,
            power_target=0.80,
            expected_providers=PROVIDERS,
            expected_countries=COUNTRIES,
            expected_items=ITEMS,
        )
        self.assertEqual(result.profiles_total, 12)
        self.assertEqual(set(result.provider_diagnostics), set(PROVIDERS))
        self.assertEqual(len(result.macro_error_covariance), 3)
        self.assertEqual(len(result.macro_error_covariance[0]), 3)
        self.assertEqual(
            set(result.variance_inflation_sensitivity),
            {"variance_x1.5", "variance_x2"},
        )
        self.assertAlmostEqual(
            result.variance_inflation_sensitivity[
                "variance_x2"
            ].standard_deviation_multiplier,
            np.sqrt(2.0),
        )
        self.assertEqual(
            set(result.single_provider_dropout_sensitivity), set(PROVIDERS)
        )
        for excluded, scenario in result.single_provider_dropout_sensitivity.items():
            self.assertEqual(scenario.excluded_provider, excluded)
            self.assertNotIn(excluded, scenario.providers)
            self.assertEqual(len(scenario.providers), len(PROVIDERS) - 1)
            self.assertEqual(
                scenario.reaches_power_target,
                scenario.macro_power.joint_holm_power >= result.power_target,
            )
        self.assertEqual(
            result.primary_gate_passes,
            result.macro_power.joint_holm_power >= result.power_target,
        )
        self.assertFalse(result.identifiable_assignment_power)
        self.assertEqual(result.null, {"beta_o": 0.0, "beta_diff": 0.0, "utility_v": 0.0})
        self.assertEqual(
            result.sesoi_alternative,
            {"beta_o": 0.05, "beta_diff": 0.05, "utility_v": 0.02},
        )

    def test_invalid_variance_inflation_factor_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "variance inflation factors"):
            simulate_prospective_macro_design_power(
                self.direction,
                self.utility,
                self.anchors,
                anchor_bootstrap=self.resampler,
                profiles_per_country=6,
                calibration_replicates=20,
                power_replicates=20,
                expected_providers=PROVIDERS,
                expected_countries=COUNTRIES,
                expected_items=ITEMS,
                variance_inflation_factors=(1.0,),
            )

    def test_null_calibration_is_close_to_one_sided_alpha(self):
        rng = np.random.default_rng(91)
        calibration = rng.normal(size=100_000)
        sensitivity = rng.normal(size=100_000)
        result, _ = _marginal_sensitivity(
            calibration,
            sensitivity,
            assumed_effect=0.05,
            null_threshold=0.05,
            alpha=0.025,
        )
        self.assertAlmostEqual(result.estimated_power, 0.025, delta=0.003)

    def test_sensitivity_is_monotone_in_assumed_effect(self):
        rng = np.random.default_rng(37)
        calibration = rng.normal(scale=0.02, size=20_000)
        sensitivity = rng.normal(scale=0.02, size=20_000)
        low, _ = _marginal_sensitivity(
            calibration,
            sensitivity,
            assumed_effect=0.05,
            null_threshold=0.05,
            alpha=0.025,
        )
        high, _ = _marginal_sensitivity(
            calibration,
            sensitivity,
            assumed_effect=0.08,
            null_threshold=0.05,
            alpha=0.025,
        )
        self.assertGreaterEqual(high.estimated_power, low.estimated_power)

    def test_profile_count_must_allow_exact_label_balance(self):
        with self.assertRaisesRegex(ValueError, "multiple of displayed labels"):
            simulate_confirmatory_sensitivity(
                self.direction,
                self.utility,
                self.anchors,
                anchor_bootstrap=self.resampler,
                profiles_per_country=5,
                calibration_replicates=100,
                sensitivity_replicates=100,
            )


if __name__ == "__main__":
    unittest.main()
