from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from api_pilot.analyze_model_country_preferences import (
    COUNTRIES,
    PRIMARY_METRICS,
    PROVIDERS,
    brier_matrix,
    build_views,
    calculate_metrics,
    france_contrast_rows,
    jensen_shannon,
    stratified_samples,
    two_way_interaction,
)


@pytest.fixture(scope="module")
def analysis_inputs():
    return build_views()


def test_clean_replacement_panel_uses_no_superseded_item_vector(analysis_inputs):
    views, audit, _, _ = analysis_inputs
    hybrid = views["clean_hybrid_all20"]
    affected = set(audit["affected_items_replaced"])
    assert len(hybrid.item_ids) == 20
    assert len(affected) == 8
    assert len(audit["unaffected_items_retained"]) == 12
    assert audit["hybrid_item_vectors"] == {
        "unaffected_original": 120_960,
        "affected_original_excluded": 80_640,
        "affected_corrected_used": 11_520,
        "total_used": 132_480,
    }
    for item in hybrid.item_ids:
        panel = hybrid.panels[item]
        if item in affected:
            assert panel.source == "corrected_wording_replacement72"
            assert len(panel.profile_ids) == 72
        else:
            assert panel.source == "original_correct_wording"
            assert len(panel.profile_ids) == 504


def test_corrected_profiles_are_same_record_subset(analysis_inputs):
    views, audit, _, _ = analysis_inputs
    assert audit["profiles"] == {
        "original": 504,
        "corrected_replacement_subset": 72,
        "replacement_subset_of_original": True,
    }
    common_ids = set(
        views["clean_common72_all20"].panels[
            views["clean_common72_all20"].item_ids[0]
        ].profile_ids
    )
    for panel in views["clean_hybrid_all20"].panels.values():
        if len(panel.profile_ids) == 72:
            assert set(panel.profile_ids) == common_ids
        else:
            assert common_ids.issubset(panel.profile_ids)


def test_nested_bootstrap_preserves_item_specific_country_and_label_counts(
    analysis_inputs,
):
    views, _, _, _ = analysis_inputs
    hybrid = views["clean_hybrid_all20"]
    samples = stratified_samples(hybrid, np.random.default_rng(20260728))
    for item, panel in hybrid.panels.items():
        expected_country = 12 if len(panel.profile_ids) == 72 else 84
        expected_label = 2 if len(panel.profile_ids) == 72 else 14
        assert Counter(
            len(samples.country[item][index]) for index in range(len(COUNTRIES))
        ) == Counter({expected_country: len(COUNTRIES)})
        assert Counter(
            len(samples.label[item][(true_index, label_index)])
            for true_index in range(len(COUNTRIES))
            for label_index in range(len(COUNTRIES))
        ) == Counter({expected_label: len(COUNTRIES) ** 2})


def test_metric_shapes_ranges_and_interaction_constraints(analysis_inputs):
    views, _, _, anchors = analysis_inputs
    metrics = calculate_metrics(views["clean_hybrid_all20"], anchors)
    expected_shape = (len(PROVIDERS), len(COUNTRIES))
    for metric in PRIMARY_METRICS:
        assert metrics[metric].shape == expected_shape
        assert np.all(np.isfinite(metrics[metric]))
        interaction = two_way_interaction(metrics[metric])
        np.testing.assert_allclose(interaction.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(interaction.sum(axis=1), 0.0, atol=1e-12)
    for metric in ("m0_anchor_tv", "m0_anchor_l2", "m0_anchor_js", "mo_tv"):
        assert np.all(metrics[metric] >= 0.0)
    assert np.all(metrics["m0_anchor_tv"] <= 1.0)
    assert np.all(metrics["mo_tv"] <= 1.0)
    assert np.all((metrics["m0_brier"] >= 0.0) & (metrics["m0_brier"] <= 2.0))


def test_targeted_france_us_contrast_has_unambiguous_sign(analysis_inputs):
    views, _, _, anchors = analysis_inputs
    view = views["clean_hybrid_all20"]
    observed = calculate_metrics(view, anchors)
    samples = {
        metric: np.repeat(value[None, ...], 100, axis=0)
        for metric, value in observed.items()
        if metric in {
            "m0_anchor_l2",
            "m0_anchor_tv",
            "mo_tv",
            "mo_regret",
            "direction_o_pooled",
        }
    }
    rows = france_contrast_rows(view, observed, samples)
    row = next(
        value
        for value in rows
        if value["provider"] == "mistral"
        and value["contrast"] == "France_minus_United_States_distance"
    )
    assert row["estimate"] > 0.04
    assert row["positive_means"] == "farther_from_France_than_United_States"


def test_probability_distance_helpers_have_known_values():
    probabilities = np.asarray([[[1.0, 0.0], [0.5, 0.5]]])
    anchor = np.asarray([0.0, 1.0])
    observed = jensen_shannon(probabilities, anchor)
    np.testing.assert_allclose(observed[0, 0], np.log(2.0), atol=1e-12)
    assert 0.0 < observed[0, 1] < np.log(2.0)
    truth = np.asarray([0, 1])
    brier = brier_matrix(probabilities, truth)
    np.testing.assert_allclose(brier, [[0.0, 0.5]], atol=1e-12)
