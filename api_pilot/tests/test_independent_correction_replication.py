from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

from api_pilot.survey_pilot_runner import build_prompt, load_prompt_bundle


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_independent_replication_contract_is_balanced_and_exact():
    manifest = load_json(
        "dataset/analysis/independent_correction_replication_manifest_v2.json"
    )
    analysis = load_json(
        "dataset/analysis/independent_correction_analysis_manifest_v2.json"
    )
    profiles = [
        json.loads(line)
        for line in (
            ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert manifest["expected"] == {
        "providers": 5,
        "calls_per_provider": 576,
        "total_calls": 2880,
    }
    assert len(profiles) == 72
    assert all(row["profile_id"].startswith("R-") for row in profiles)
    assert Counter(row["true_country_code"] for row in profiles) == Counter(
        {country: 12 for country in ("CN", "US", "JO", "FR", "GB", "IT")}
    )
    assert len(manifest["task_units"]) == 576
    assert len({tuple(row.values()) for row in manifest["task_units"]}) == 576
    assert len(analysis["target_strata"]["high_country_value"]) == 7
    assert len(analysis["target_strata"]["filler"]) == 3
    assert not set(analysis["target_strata"]["high_country_value"]).intersection(
        analysis["target_strata"]["filler"]
    )


def test_replication_batches_keep_five_items_and_no_duplicates():
    analysis = load_json(
        "dataset/analysis/independent_correction_analysis_manifest_v2.json"
    )
    batches = analysis["target_batches"]
    assert set(batches) == {"B01", "B02"}
    assert all(len(items) == 5 for items in batches.values())
    flattened = [item for items in batches.values() for item in items]
    assert len(flattened) == len(set(flattened)) == 10


def test_corrected_profile_and_rendered_prompt_use_yes_no_consistently():
    profile = json.loads(
        (ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    affected = {"A124_09", "A124_08", "A124_03", "A030", "A029"}
    targets = {row["item_id"]: row for row in profile["targets"]}
    for item_id in affected:
        assert profile["target_options"][item_id] == [
            {"answer_code": "0", "answer": "No"},
            {"answer_code": "1", "answer": "Yes"},
        ]
        assert targets[item_id]["answer"] in {"No", "Yes"}
    bundle = load_prompt_bundle(
        ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json"
    )
    prompt, _ = build_prompt(profile, "MR", "correct", bundle)
    assert "Would you prefer not to have drug addicts as neighbours?" in prompt
    assert "[0] No | [1] Yes" in prompt
    assert "[0] Not mentioned" not in prompt


def test_replication_source_records_exclude_original_504():
    sys.path.insert(0, str(ROOT / "dataset/tools"))
    import build_confirmatory_manifest as cm
    import build_pilot_gate as gate
    import csv

    source = load_json("dataset/analysis/confirmatory_manifest_v2.json")
    full_targets = tuple(
        item for batch in source["target_batches"].values() for item in batch
    )
    replication_targets = (
        "A124_09", "E025", "A124_08", "A124_03", "A030", "E027", "A029",
        "D060", "D078", "D061",
    )
    with (ROOT / "dataset/derived/country_panel_6.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if gate.respondent_split(row["uniqid"]) == "confirmatory"
        ]
    for country in gate.COUNTRIES:
        country_rows = [row for row in rows if row["cntry_AN"] == country]
        original = sorted(
            (
                row for row in country_rows
                if all(row[item] not in gate.MISSING_VALUES for item in gate.EVIDENCE_ITEMS + full_targets)
            ),
            key=lambda row: cm.stable_unit("respondent-order", row["uniqid"]),
        )[:84]
        original_ids = {row["uniqid"] for row in original}
        eligible_replication = sorted(
            (
                row for row in country_rows
                if row["uniqid"] not in original_ids
                and all(row[item] not in gate.MISSING_VALUES for item in gate.EVIDENCE_ITEMS + replication_targets)
            ),
            key=lambda row: cm.stable_unit("respondent-order", row["uniqid"]),
        )[:12]
        assert len(eligible_replication) == 12
        assert original_ids.isdisjoint(row["uniqid"] for row in eligible_replication)


def test_invalid_v1_is_explicitly_quarantined():
    invalidation = load_json(
        "api_pilot/restricted_outputs/invalidated/independent_correction_v1_20260726-213804/INVALIDATION.json"
    )
    assert invalidation["status"] == "invalidated_never_analyze"
    assert invalidation["replacement_manifest"].endswith(
        "independent_correction_replication_manifest_v2.json"
    )


def test_builder_cannot_rewrite_frozen_inputs_after_checkpoint_calls():
    paths = [
        ROOT / "dataset/analysis/independent_correction_replication_manifest_v2.json",
        ROOT / "dataset/analysis/independent_correction_analysis_manifest_v2.json",
        ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    completed = subprocess.run(
        [sys.executable, str(ROOT / "api_pilot/build_independent_correction_replication.py"), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"action": "validated_without_rewrite"' in completed.stdout
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_unattended_supervisor_treats_runner_code_one_as_retryable():
    script = (
        ROOT / "api_pilot/scripts/windows/run_independent_correction_automation.ps1"
    ).read_text(
        encoding="utf-8"
    )
    assert "$runnerCode -notin @(0, 1)" in script
    assert "$runnerCode -ne 0" not in script
