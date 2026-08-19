import unittest

import numpy as np

from api_pilot.strict_inference import (
    ReferenceAnchorResampler,
    balanced_assignment_permutation,
    direction_beta_macro,
    direction_profile_bootstrap,
    holm_adjust,
    paired_profile_bootstrap,
    run_strict_confirmatory_inference,
    scalar_macro,
)


PROVIDERS = ("model-a", "model-b")
COUNTRIES = ("C1", "C2")
ITEMS = ("Q1",)
ANCHORS = {"Q1": {"C1": [1.0, 0.0], "C2": [0.0, 1.0]}}


def scalar_rows(country_values):
    rows = []
    for provider in PROVIDERS:
        for country, profiles in country_values.items():
            for profile_id, weight, value in profiles:
                rows.append(
                    {
                        "provider": provider,
                        "country": country,
                        "profile_id": profile_id,
                        "item_id": "Q1",
                        "weight": weight,
                        "value": value,
                    }
                )
    return rows


def direction_rows(*, unbalanced=False, mismatched_provider=False):
    rows = []
    labels_by_country = {
        "C1": ["C1", "C1" if unbalanced else "C2"],
        "C2": ["C1", "C2"],
    }
    direction = {
        "C1": np.asarray([0.5, -0.5]),
        "C2": np.asarray([-0.5, 0.5]),
    }
    for provider in PROVIDERS:
        for country in COUNTRIES:
            for index, label in enumerate(labels_by_country[country]):
                actual_label = label
                if mismatched_provider and provider == "model-b" and country == "C1" and index == 0:
                    actual_label = "C2"
                rows.append(
                    {
                        "provider": provider,
                        "country": country,
                        "profile_id": f"{country}-P{index + 1}",
                        "item_id": "Q1",
                        "weight": 1.0 + index,
                        "displayed_label": actual_label,
                        "mo_delta": (0.2 * direction[label]).tolist(),
                        "mr_delta": (0.05 * direction[label]).tolist(),
                    }
                )
    return rows


class ScalarInferenceTests(unittest.TestCase):
    def test_equal_country_macro_prevents_large_country_domination(self):
        rows = scalar_rows(
            {
                "C1": [("c1-a", 1.0, 0.0)],
                "C2": [(f"c2-{index}", 1.0, 1.0) for index in range(9)],
            }
        )
        self.assertAlmostEqual(scalar_macro(rows), 0.5)

    def test_survey_weights_are_used_within_country(self):
        rows = scalar_rows(
            {
                "C1": [("a", 1.0, 0.0), ("b", 3.0, 1.0)],
                "C2": [("c", 1.0, 0.0), ("d", 3.0, 1.0)],
            }
        )
        self.assertAlmostEqual(scalar_macro(rows), 0.75)
        result = paired_profile_bootstrap(rows, n_bootstrap=100, seed=7)
        self.assertAlmostEqual(result.estimate, 0.75)
        self.assertTrue(0 <= result.p_greater <= 1)

    def test_scalar_panel_fails_on_missing_provider_cell(self):
        rows = scalar_rows(
            {
                "C1": [("a", 1.0, 0.0), ("b", 1.0, 1.0)],
                "C2": [("c", 1.0, 0.0), ("d", 1.0, 1.0)],
            }
        )
        rows.pop()
        with self.assertRaisesRegex(ValueError, "same profiles|Missing scalar"):
            scalar_macro(rows)

    def test_scalar_panel_fails_when_weight_changes_by_model(self):
        rows = scalar_rows(
            {
                "C1": [("a", 1.0, 0.0), ("b", 1.0, 1.0)],
                "C2": [("c", 1.0, 0.0), ("d", 1.0, 1.0)],
            }
        )
        rows[-1]["weight"] = 2.0
        with self.assertRaisesRegex(ValueError, "weight differs"):
            scalar_macro(rows)


class DirectionInferenceTests(unittest.TestCase):
    def test_direction_formula_returns_known_ratios(self):
        beta_o, beta_r, difference = direction_beta_macro(
            direction_rows(), ANCHORS
        )
        self.assertAlmostEqual(beta_o, 0.2)
        self.assertAlmostEqual(beta_r, 0.05)
        self.assertAlmostEqual(difference, 0.15)

    def test_country_specific_ratios_receive_equal_macro_weight(self):
        anchors = {
            "Q1": {
                "L1": [1.0, 0.0],
                "L2": [0.0, 1.0],
                "L3": [0.5, 0.5],
            }
        }
        directions = {
            label: np.asarray(values) - np.asarray([0.5, 0.5])
            for label, values in anchors["Q1"].items()
        }
        country_betas = {"C1": 0.2, "C2": 0.5, "C3": 0.8}
        country_weights = {
            "C1": [20.0, 20.0, 1.0],
            "C2": [1.0, 1.0, 20.0],
            "C3": [1.0, 1.0, 20.0],
        }
        rows = []
        for country, beta in country_betas.items():
            for index, label in enumerate(("L1", "L2", "L3")):
                rows.append(
                    {
                        "provider": "model",
                        "country": country,
                        "profile_id": f"{country}-P{index}",
                        "item_id": "Q1",
                        "weight": country_weights[country][index],
                        "displayed_label": label,
                        "mo_delta": (beta * directions[label]).tolist(),
                        "mr_delta": (0.5 * beta * directions[label]).tolist(),
                    }
                )
        beta_o, beta_r, difference = direction_beta_macro(rows, anchors)
        self.assertAlmostEqual(beta_o, 0.5)
        self.assertAlmostEqual(beta_r, 0.25)
        self.assertAlmostEqual(difference, 0.25)

    def test_balanced_permutation_outputs_iut_max(self):
        result = balanced_assignment_permutation(
            direction_rows(), ANCHORS, n_permutations=999, seed=11, chunk_size=64
        )
        self.assertAlmostEqual(
            result.h1_iut_p,
            max(result.p_beta_o_greater, result.p_beta_diff_greater),
        )
        self.assertEqual(result.n_permutations, 999)
        self.assertTrue(0 < result.p_beta_o_greater <= 1)
        self.assertTrue(0 < result.p_beta_r_greater <= 1)

    def test_unbalanced_assignment_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "not balanced"):
            balanced_assignment_permutation(
                direction_rows(unbalanced=True), ANCHORS, n_permutations=10
            )

    def test_displayed_label_must_be_shared_across_models(self):
        with self.assertRaisesRegex(ValueError, "Displayed label differs"):
            direction_beta_macro(
                direction_rows(mismatched_provider=True), ANCHORS
            )

    def test_delta_must_sum_to_zero(self):
        rows = direction_rows()
        rows[0]["mo_delta"] = [0.1, 0.1]
        with self.assertRaisesRegex(ValueError, "does not sum to zero"):
            direction_beta_macro(rows, ANCHORS)

    def test_invalid_anchor_distribution_fails(self):
        anchors = {"Q1": {"C1": [0.8, 0.8], "C2": [0.0, 1.0]}}
        with self.assertRaisesRegex(ValueError, "not a probability"):
            direction_beta_macro(direction_rows(), anchors)

    def test_confirmatory_bootstrap_requires_anchor_reestimation(self):
        with self.assertRaisesRegex(ValueError, "requires anchor re-estimation"):
            direction_profile_bootstrap(
                direction_rows(),
                ANCHORS,
                require_anchor_reestimation=True,
                n_bootstrap=5,
            )

    def test_anchor_callback_is_used_for_every_bootstrap_draw(self):
        calls = []

        def callback(_rng):
            calls.append(1)
            return ANCHORS

        result = direction_profile_bootstrap(
            direction_rows(),
            ANCHORS,
            anchor_bootstrap=callback,
            require_anchor_reestimation=True,
            n_bootstrap=17,
            seed=12,
        )
        self.assertEqual(len(calls), 17)
        self.assertTrue(result.beta_o.anchors_reestimated)


class MultiplicityTests(unittest.TestCase):
    def test_holm_known_values_and_original_key_order(self):
        adjusted = holm_adjust({"H1": 0.04, "H2": 0.01, "H3": 0.03})
        self.assertEqual(list(adjusted), ["H1", "H2", "H3"])
        self.assertAlmostEqual(adjusted["H1"], 0.06)
        self.assertAlmostEqual(adjusted["H2"], 0.03)
        self.assertAlmostEqual(adjusted["H3"], 0.06)


class ReferenceBootstrapTests(unittest.TestCase):
    def reference_rows(self):
        rows = []
        for country, answers in {"C1": [0, 0, 0], "C2": [1, 1, 1]}.items():
            for index, answer in enumerate(answers):
                rows.append(
                    {
                        "country": country,
                        "profile_id": f"{country}-R{index}",
                        "item_id": "Q1",
                        "answer_index": answer,
                        "n_options": 2,
                        "weight": index + 1.0,
                    }
                )
        return rows

    def test_reference_resampler_builds_probabilities(self):
        resampler = ReferenceAnchorResampler(
            self.reference_rows(),
            expected_countries=COUNTRIES,
            expected_items=ITEMS,
        )
        anchors = resampler.estimate()
        for country in COUNTRIES:
            self.assertAlmostEqual(sum(anchors["Q1"][country]), 1.0)
        sampled = resampler(np.random.default_rng(3))
        for country in COUNTRIES:
            self.assertAlmostEqual(sum(sampled["Q1"][country]), 1.0)

    def test_reference_resampler_fails_on_incomplete_item_panel(self):
        rows = self.reference_rows()
        rows.append(
            {
                "country": "C1",
                "profile_id": "C1-R0",
                "item_id": "Q2",
                "answer_index": 0,
                "n_options": 2,
                "weight": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "Missing reference observation"):
            ReferenceAnchorResampler(rows)

    def test_full_confirmatory_entry_runs_holm(self):
        resampler = ReferenceAnchorResampler(self.reference_rows())
        anchors = resampler.estimate()
        result = run_strict_confirmatory_inference(
            direction_rows(),
            scalar_rows(
                {
                    "C1": [("C1-P1", 1.0, 0.1), ("C1-P2", 2.0, 0.2)],
                    "C2": [("C2-P1", 1.0, 0.1), ("C2-P2", 2.0, 0.2)],
                }
            ),
            anchors,
            anchor_bootstrap=resampler,
            n_bootstrap=30,
            n_permutations=50,
            expected_providers=PROVIDERS,
            expected_countries=COUNTRIES,
            expected_items=ITEMS,
        )
        self.assertEqual(set(result.holm_adjusted_p), {"H1", "H2"})
        self.assertTrue(result.direction_bootstrap.beta_o.anchors_reestimated)
        self.assertAlmostEqual(
            result.raw_hypothesis_p["H1"],
            max(
                result.direction_bootstrap.beta_o.p_greater,
                result.direction_bootstrap.beta_diff.p_greater,
            ),
        )
        self.assertEqual(result.direction_bootstrap.beta_o.scientific_null, 0.0)
        self.assertEqual(result.direction_bootstrap.beta_diff.scientific_null, 0.05)
        self.assertEqual(result.utility_h2.scientific_null, 0.02)
        self.assertEqual(
            result.raw_hypothesis_p["H2"],
            result.utility_h2.p_greater,
        )
        self.assertEqual(
            result.raw_sesoi_exceedance_p["H1"],
            max(
                result.direction_bootstrap.beta_o.p_greater_scientific_null,
                result.direction_bootstrap.beta_diff.p_greater_scientific_null,
            ),
        )
        self.assertEqual(
            result.raw_sesoi_exceedance_p["H2"],
            result.utility_h2.p_greater_scientific_null,
        )
        self.assertIn("Fisher sharp null", result.assignment_permutation.null_hypothesis)
        self.assertFalse(result.assignment_permutation.studentized)


if __name__ == "__main__":
    unittest.main()
