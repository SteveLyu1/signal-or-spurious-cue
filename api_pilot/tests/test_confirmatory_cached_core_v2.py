from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from api_pilot.materialize_confirmatory_cached_panel import (
    verify_cache_prompt_identity,
)
from api_pilot.run_confirmatory_strict_inference import batch_profile
from api_pilot.strict_inference import BootstrapResult
from api_pilot.survey_pilot_runner import load_prompt_bundle


ROOT = Path(__file__).resolve().parents[2]


def _paired_profiles() -> tuple[dict, dict]:
    rows = [
        json.loads(line)
        for line in (
            ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_id = {row["profile_id"]: row for row in rows}
    target = rows[0]
    return target, by_id[target["donor_profile_id"]]


def test_reciprocal_donor_prompt_is_byte_identical_for_all_conditions():
    target, donor = _paired_profiles()
    bundle = load_prompt_bundle(
        ROOT / "api_pilot/prompts/survey-confirmatory-v4.json"
    )
    target_ids = [row["item_id"] for row in target["targets"][:5]]
    for condition in ("M0", "MO", "MR", "MV"):
        digest, specs = verify_cache_prompt_identity(
            target, donor, condition, target_ids, bundle
        )
        assert len(digest) == 64
        assert [row["item_id"] for row in specs] == target_ids


def test_cache_prompt_identity_fails_closed_after_evidence_change():
    target, donor = _paired_profiles()
    donor = copy.deepcopy(donor)
    donor["evidence"][0]["answer_code"] = "CACHE-MISMATCH"
    bundle = load_prompt_bundle(
        ROOT / "api_pilot/prompts/survey-confirmatory-v4.json"
    )
    target_ids = [row["item_id"] for row in target["targets"][:5]]
    with pytest.raises(ValueError, match="Rendered cache prompts differ"):
        verify_cache_prompt_identity(target, donor, "M0", target_ids, bundle)


def test_batch_profile_preserves_frozen_target_order():
    target, _ = _paired_profiles()
    target_ids = [row["item_id"] for row in target["targets"][5:10]]
    selected = batch_profile(target, target_ids)
    assert [row["item_id"] for row in selected["targets"]] == target_ids


def test_beta_r_equivalence_rule_is_ci_containment_not_reverse_significance():
    margin = 0.05
    equivalent = BootstrapResult(0.0, -0.04, 0.04, None, 100)
    non_equivalent = BootstrapResult(0.0, -0.06, 0.03, None, 100)
    assert equivalent.ci_low > -margin and equivalent.ci_high < margin
    assert not (non_equivalent.ci_low > -margin and non_equivalent.ci_high < margin)
