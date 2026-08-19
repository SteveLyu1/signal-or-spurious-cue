from __future__ import annotations

import copy

import pytest

from api_pilot.run_confirmatory_comprehensive_metrics import (
    OFFICIAL_MODELS,
    REQUIRED_CELLS,
    audit_formal_item_panel,
    confirmatory_strata,
    pending_output,
)
from api_pilot.run_gate_c_comprehensive_metrics import (
    AUXILIARY_METRICS,
    build_rows,
)
from api_pilot.run_gate_c_strict_inference import item_result_index


def _analysis_manifest() -> dict:
    high = [f"H{index}" for index in range(7)]
    representative = [f"R{index}" for index in range(9)]
    controls = [f"C{index}" for index in range(4)]
    all_items = high + representative + controls
    return {
        "target_strata": {
            "high_country_value": high,
            "representative": representative,
            "equivalence_controls": controls,
        },
        "target_batches": {
            f"B{index + 1:02d}": all_items[index * 5 : (index + 1) * 5]
            for index in range(4)
        },
    }


def _minimal_complete_panel() -> tuple[list[dict], list[dict]]:
    profile = {
        "profile_id": "P1",
        "true_country_code": "CN",
        "analysis_weight": 1.0,
        "targets": [{"item_id": "Q", "answer_code": "yes"}],
        "target_options": {
            "Q": [
                {"answer_code": "no", "label": "No"},
                {"answer_code": "yes", "label": "Yes"},
            ]
        },
    }
    rows = []
    for provider in OFFICIAL_MODELS:
        for condition, evidence in REQUIRED_CELLS:
            rows.append(
                {
                    "provider": provider,
                    "profile_id": "P1",
                    "condition": condition,
                    "evidence_variant": evidence,
                    "target_ids": ["Q"],
                    "target_option_codes": {"Q": ["no", "yes"]},
                    "true_country_code": "CN",
                    "parsed_forecasts": {"Q": [0.25, 0.75]},
                }
            )
    return rows, [profile]


def test_confirmatory_strata_freezes_7_9_4_and_20_unique_targets():
    strata = confirmatory_strata(_analysis_manifest())
    assert [len(strata[name]) for name in (
        "high_country_value", "representative", "equivalence_controls"
    )] == [7, 9, 4]
    assert len(strata["all_targets"]) == 20
    assert len(set(strata["all_targets"])) == 20


def test_confirmatory_strata_rejects_overlap_across_strata():
    manifest = _analysis_manifest()
    manifest["target_strata"]["equivalence_controls"][0] = "H0"
    with pytest.raises(ValueError, match="20 unique"):
        confirmatory_strata(manifest)


def test_formal_item_panel_requires_all_cells_for_frozen_provider_roster():
    rows, profiles = _minimal_complete_panel()
    audit = audit_formal_item_panel(rows, profiles, ["Q"])
    assert audit["complete"]
    assert audit["item_cells_checked"] == len(OFFICIAL_MODELS) * len(REQUIRED_CELLS)
    assert audit["option_codes_aligned_to_frozen_profiles"]


def test_formal_item_panel_rejects_option_code_misalignment():
    rows, profiles = _minimal_complete_panel()
    broken = copy.deepcopy(rows)
    broken[0]["target_option_codes"]["Q"] = ["yes", "no"]
    with pytest.raises(ValueError, match="Option-code alignment mismatch"):
        audit_formal_item_panel(broken, profiles, ["Q"])


def test_cached_shuffled_rows_are_not_filtered_as_runtime_repeats():
    row = {
        "call_id": "derived",
        "provider": "deepseek",
        "profile_id": "P1",
        "condition": "M0",
        "evidence_variant": "shuffled",
        "target_ids": ["Q"],
        "parsed_forecasts": {"Q": [0.4, 0.6]},
        "source_call_id": "raw-donor-call",
        "derived_cached": True,
        "experiment": "confirmatory_core_v2_logical",
    }
    index = item_result_index([row])
    assert ("deepseek", "P1", "M0", "shuffled", "Q") in index


def test_pending_output_contains_no_estimate_or_analysis(monkeypatch, tmp_path):
    class Args:
        results = tmp_path / "missing.jsonl"
        profiles = tmp_path / "profiles.jsonl"
        analysis_manifest = tmp_path / "analysis.json"
        task_manifest = tmp_path / "tasks.json"
        prompt_bundle = tmp_path / "prompt.json"

    audit = {
        "complete": False,
        "planned": 48_384,
        "observed_unique": 0,
        "missing": 48_384,
        "unexpected": 0,
        "incomplete": 0,
        "analysis_ready": 0,
    }
    output = pending_output(Args(), audit, reason="not complete")
    assert output["status"] == "pending_inference"
    assert output["estimates_generated"] is False
    assert output["hypothesis_decisions_generated"] is False
    assert "analysis" not in output
    assert "estimate" not in output


def test_auxiliary_accuracy_tv_and_ordinal_metrics_are_built():
    profile = {
        "profile_id": "P1", "true_country_code": "CN",
        "analysis_weight": 1.0,
        "targets": [{"item_id": "E025", "answer_code": "2"}],
        "target_options": {"E025": [
            {"answer_code": "1"}, {"answer_code": "2"},
            {"answer_code": "3"},
        ]},
    }
    forecasts = {
        "M0": [0.2, 0.5, 0.3], "MO": [0.5, 0.3, 0.2],
        "MR": [0.2, 0.5, 0.3], "MV": [0.1, 0.8, 0.1],
    }
    rows = []
    for provider in OFFICIAL_MODELS:
        for condition, evidence in REQUIRED_CELLS:
            rows.append({
                "call_id": f"{provider}/{condition}/{evidence}",
                "provider": provider, "profile_id": "P1",
                "true_country_code": "CN", "condition": condition,
                "evidence_variant": evidence,
                "displayed_country_code": (
                    "FR" if condition in {"MO", "MR"} else
                    "CN" if condition == "MV" else None
                ),
                "target_option_codes": {"E025": ["1", "2", "3"]},
                "target_ids": ["E025"],
                "parsed_forecasts": {"E025": forecasts[condition]},
            })
    _, scalar, _ = build_rows(
        rows, [profile], {"all_targets": ["E025"]}
    )
    assert set(AUXILIARY_METRICS).issubset(scalar["all_targets"])
    values = {
        metric: metric_rows[0]["value"]
        for metric, metric_rows in scalar["all_targets"].items()
    }
    assert values["m0_accuracy"] == 1.0
    assert values["mv_accuracy_delta"] == 0.0
    assert values["mo_accuracy_delta"] == -1.0
    assert values["mv_ordinal_crps_delta"] < 0
