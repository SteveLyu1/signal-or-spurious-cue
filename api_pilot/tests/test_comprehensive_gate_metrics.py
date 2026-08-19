import numpy as np

from api_pilot.comprehensive_gate_metrics import (
    _interval,
    _summarize_provider_matrix,
    comprehensive_bootstrap,
)


def test_interval_uses_percentile_limits():
    result = _interval(2.0, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 0.8)
    assert result.estimate == 2.0
    assert np.isclose(result.ci_low, 0.4)
    assert np.isclose(result.ci_high, 3.6)


def test_provider_summary_is_equal_provider_and_leave_one_out():
    observed = np.array([1.0, 3.0, 8.0])
    samples = np.array([[0.0, 2.0, 7.0], [2.0, 4.0, 9.0]])
    result = _summarize_provider_matrix(
        ("a", "b", "model_c"), observed, samples, ("a", "b"), confidence=0.95
    )
    assert result["macro"]["estimate"] == 2.0
    assert result["leave_one_provider_out_point_estimates"] == {"a": 3.0, "b": 1.0}
    assert set(result["by_provider"]) == {"a", "b"}


def test_comprehensive_bootstrap_preserves_equal_provider_macro():
    anchors = {"item": {"A": [1.0, 0.0], "B": [0.0, 1.0]}}
    direction = []
    scalar = []
    for provider, size, scalar_value in (("p1", 0.1, 1.0), ("p2", 0.2, 3.0)):
        for country in ("A", "B"):
            for index, label in enumerate(("A", "B")):
                vector = [0.5 * size, -0.5 * size]
                if label == "B":
                    vector = [-value for value in vector]
                common = {
                    "provider": provider,
                    "profile_id": f"{country}-{index}",
                    "country": country,
                    "item_id": "item",
                    "weight": 1.0 + index,
                }
                direction.append(
                    {
                        **common,
                        "displayed_label": label,
                        "mo_delta": vector,
                        "mr_delta": [0.0, 0.0],
                    }
                )
                scalar.append({**common, "value": scalar_value})

    result = comprehensive_bootstrap(
        {"only": direction},
        {"only": {"metric": scalar}},
        anchors,
        anchor_bootstrap=lambda rng: anchors,
        provider_sets={"both": ("p1", "p2")},
        expected_providers=("p1", "p2"),
        expected_countries=("A", "B"),
        items_by_stratum={"only": ("item",)},
        n_bootstrap=20,
    )
    output = result["provider_sets"]["both"]["strata"]["only"]
    assert np.isclose(output["direction"]["beta_o"]["macro"]["estimate"], 0.15)
    assert np.isclose(output["direction"]["beta_o_minus_beta_r"]["macro"]["estimate"], 0.15)
    assert np.isclose(output["prediction"]["metric"]["macro"]["estimate"], 2.0)
