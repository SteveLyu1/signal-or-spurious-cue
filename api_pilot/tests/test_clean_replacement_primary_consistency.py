import json
from pathlib import Path

import pytest

from api_pilot.analyze_clean_replacement_primary_consistency import HIGH_ITEMS, summarize


def test_high_item_contract_is_frozen():
    assert HIGH_ITEMS == (
        "A124_09", "E025", "A124_08", "A124_03", "A030", "E027", "A029"
    )


def test_summary_macro_averages_models_and_countries():
    import numpy as np

    observed = {
        "direction_o_pooled": np.full((5, 6), 0.2),
        "direction_r_pooled": np.full((5, 6), 0.1),
        "direction_o_minus_direction_r_pooled": np.full((5, 6), 0.1),
        "utility_v": np.full((5, 6), 0.04),
        "mo_regret": np.full((5, 6), 0.01),
        "mr_regret": np.full((5, 6), 0.02),
    }
    samples = {key: np.stack([value, value], axis=0) for key, value in observed.items()}
    result = summarize(observed, samples)
    assert result["beta_o"]["estimate"] == pytest.approx(0.2)
    assert result["beta_o_minus_beta_r"]["estimate"] == pytest.approx(0.1)
    assert result["utility_v"]["estimate"] == pytest.approx(0.04)


def test_frozen_output_uses_only_clean_item_vectors():
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "api_pilot/outputs/clean_replacement_primary_consistency.json")
        .read_text(encoding="utf-8")
    )
    coverage = {item: values["profiles"] for item, values in result["items"].items()}
    assert sum(coverage.values()) == 2 * 504 + 5 * 72
    assert {item for item, count in coverage.items() if count == 504} == {"E025", "E027"}
    assert result["source_audit"]["affected_old_wording_vectors_used"] == 0
    assert result["analysis_role"] == (
        "key_secondary_post_review_clean_replacement_consistency"
    )
