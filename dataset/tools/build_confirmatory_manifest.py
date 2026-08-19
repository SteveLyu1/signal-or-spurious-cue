#!/usr/bin/env python3
"""Freeze confirmatory targets and build restricted held-out profiles.

This script consumes the development Gate report. It does not refit models or
inspect confirmatory outcomes when selecting items. Respondent answers are read
only after the target set has been fixed, then written to a restricted profile
file for scoring and prompt construction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_pilot_gate as gate


SEED = "aaai-country-metadata-confirmatory-v2"
LABEL_RANDOMIZATION_SEED = "aaai-country-metadata-confirmatory-label-v1"
HIGH_RELATIVE_GAIN = 0.05
CONTROL_RELATIVE_UPPER = 0.01
BETA_R_EQUIVALENCE_MARGIN = 0.05
PROFILES_PER_COUNTRY = 84
GATE_C_PROFILES_PER_COUNTRY = 12
CONFIRMATORY_FROZEN_AT = "2026-07-25T10:21:53+08:00"
CACHED_DESIGN_FROZEN_AT = "2026-07-25T16:40:00+08:00"
EXCLUDED_TARGETS = {
    "C002",  # asks about priority for nationals versus immigrants
    "E012",  # "fight for your country" makes a displayed country a referent
}

DOMAINS = {
    "life": ("A001", "A002", "A003", "A004", "A005", "A008", "A009", "A170", "A173"),
    "child": ("A027", "A029", "A030", "A032", "A034", "A035", "A038", "A039", "A040", "A041", "A042"),
    "social_distance": ("A124_02", "A124_03", "A124_06", "A124_08", "A124_09"),
    "work_gender": ("C002", "C038", "C041", "D054", "D060", "D061", "D078"),
    "political_civic": ("E012", "E015", "E018", "E025", "E026", "E027", "E028"),
    "moral": ("F118", "F120", "F121"),
}


def stable_unit(*parts: str) -> float:
    payload = "\x1f".join((SEED, *parts)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / 2**64


def stable_label_randomization_unit(*parts: str) -> float:
    """Return a stable draw from the label-randomization namespace.

    Profile selection and displayed-label assignment deliberately use different
    seeds and hash domains.  This keeps the randomization independent of the
    ordering used to select the 84 complete records in each country stratum.
    """

    payload = "\x1f".join((LABEL_RANDOMIZATION_SEED, *parts)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / 2**64


def balanced_seeded_label_assignment(
    country: str,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    """Assign labels by stable-ID random rank, exactly balanced in a stratum."""

    labels = [gate.COUNTRY_LABELS[code] for code in gate.COUNTRIES]
    if len(rows) % len(labels) != 0:
        raise ValueError(
            f"{country} profile count {len(rows)} is not divisible by "
            f"{len(labels)} displayed labels"
        )
    unique_ids = [row["uniqid"] for row in rows]
    if len(unique_ids) != len(set(unique_ids)):
        raise ValueError(f"Duplicate uniqid in {country} randomization stratum")
    ranked = sorted(
        rows,
        key=lambda row: (
            stable_label_randomization_unit(
                "balanced-country-label", country, row["uniqid"]
            ),
            row["uniqid"],
        ),
    )
    per_label = len(rows) // len(labels)
    return {
        row["uniqid"]: labels[index // per_label]
        for index, row in enumerate(ranked)
    }


def legacy_cyclic_label_assignment(
    country_index: int,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    """Reproduce the deterministic assignment used by completed pilot runs."""

    labels = [gate.COUNTRY_LABELS[code] for code in gate.COUNTRIES]
    return {
        row["uniqid"]: labels[(index + country_index) % len(labels)]
        for index, row in enumerate(rows)
    }


def relative_gain(row: dict[str, Any], key: str) -> float:
    baseline = float(row["brier_evidence"])
    if baseline <= 0:
        return float("inf")
    return float(row[key]) / baseline


def select_confirmatory_targets(
    results: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    by_id = {row["variable"]: row for row in results}
    evidence_valid = {
        row["variable"]
        for row in results
        if float(row["evidence_gain_ci_low"]) > 0
    }.difference(EXCLUDED_TARGETS)
    high = sorted(
        (
            row["variable"]
            for row in results
            if row["variable"] in evidence_valid
            and float(row["country_gain_ci_low"]) > 0
            and relative_gain(row, "country_incremental_gain") >= HIGH_RELATIVE_GAIN
        ),
        key=lambda variable: (
            -relative_gain(by_id[variable], "country_incremental_gain"),
            variable,
        ),
    )
    controls = sorted(
        (
            row["variable"]
            for row in results
            if row["variable"] in evidence_valid
            and float(row["country_gain_ci_low"]) <= 0 <= float(row["country_gain_ci_high"])
            and relative_gain(row, "country_gain_ci_high") <= CONTROL_RELATIVE_UPPER
        ),
        key=lambda variable: (
            relative_gain(by_id[variable], "country_gain_ci_high"),
            variable,
        ),
    )[:4]
    excluded = set(high) | set(controls)
    representative = []
    for domain, variables in DOMAINS.items():
        eligible = sorted(
            evidence_valid.intersection(variables).difference(excluded),
            key=lambda variable: stable_unit("representative", domain, variable),
        )
        if not eligible:
            raise ValueError(f"No evidence-valid representative item for {domain}")
        representative.append(eligible[0])
    remaining = sorted(
        evidence_valid.difference(excluded).difference(representative),
        key=lambda variable: stable_unit("representative-fill", variable),
    )
    representative.extend(remaining[: 9 - len(representative)])
    if len(high) != 7 or len(representative) != 9 or len(controls) != 4:
        raise ValueError(
            "Expected 7 high, 9 representative, and 4 equivalence-control targets; "
            f"got {len(high)}, {len(representative)}, {len(controls)}"
        )
    return high, representative, controls


def build_profiles(
    rows: list[dict[str, str]],
    metadata: dict[str, Any],
    targets: list[str],
    *,
    profiles_per_country: int = PROFILES_PER_COUNTRY,
    profile_prefix: str = "C",
    label_assignment: str = "balanced_seeded",
    shuffled_pairing: str = "legacy_cyclic",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    required = gate.EVIDENCE_ITEMS + tuple(targets)
    eligible_counts: dict[str, int] = {}
    selected: dict[str, list[dict[str, str]]] = {}
    for country in gate.COUNTRIES:
        eligible = [
            row
            for row in rows
            if row["cntry_AN"] == country
            and all(row[variable] not in gate.MISSING_VALUES for variable in required)
        ]
        eligible_counts[country] = len(eligible)
        eligible.sort(key=lambda row: stable_unit("respondent-order", row["uniqid"]))
        if len(eligible) < profiles_per_country:
            raise ValueError(
                f"{country} has {len(eligible)} complete confirmatory rows; "
                f"need {profiles_per_country}"
            )
        selected[country] = eligible[:profiles_per_country]

    profiles = []
    for country_index, country in enumerate(gate.COUNTRIES):
        country_rows = selected[country]
        if label_assignment == "balanced_seeded":
            assigned_labels = balanced_seeded_label_assignment(country, country_rows)
        elif label_assignment == "legacy_cyclic":
            assigned_labels = legacy_cyclic_label_assignment(country_index, country_rows)
        else:
            raise ValueError(f"Unknown label assignment method: {label_assignment}")
        profile_id_by_uniqid = {
            row["uniqid"]: f"{profile_prefix}-{country}-{index + 1:03d}"
            for index, row in enumerate(country_rows)
        }
        donor_by_uniqid: dict[str, tuple[str, dict[str, str]]] = {}
        if shuffled_pairing == "reciprocal_within_country_label":
            label_to_code = {label: code for code, label in gate.COUNTRY_LABELS.items()}
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in country_rows:
                grouped[assigned_labels[row["uniqid"]]].append(row)
            for label in sorted(grouped, key=lambda value: label_to_code[value]):
                ranked = sorted(
                    grouped[label],
                    key=lambda row: (
                        stable_label_randomization_unit(
                            "reciprocal-donor-pair", country, label, row["uniqid"]
                        ),
                        row["uniqid"],
                    ),
                )
                if len(ranked) % 2:
                    raise ValueError(
                        f"Odd donor-pair stratum {country}/{label}: {len(ranked)}"
                    )
                code = label_to_code[label]
                for pair_index in range(0, len(ranked), 2):
                    first, second = ranked[pair_index : pair_index + 2]
                    pair_id = f"PAIR-{country}-{code}-{pair_index // 2 + 1:02d}"
                    donor_by_uniqid[first["uniqid"]] = (pair_id, second)
                    donor_by_uniqid[second["uniqid"]] = (pair_id, first)
        elif shuffled_pairing != "legacy_cyclic":
            raise ValueError(f"Unknown shuffled pairing method: {shuffled_pairing}")
        for index, row in enumerate(country_rows):
            if shuffled_pairing == "reciprocal_within_country_label":
                pair_id, shuffled = donor_by_uniqid[row["uniqid"]]
            else:
                pair_id = None
                shuffled = country_rows[(index + 1) % len(country_rows)]
            profile = {
                    "profile_id": f"{profile_prefix}-{country}-{index + 1:03d}",
                    "true_country_code": country,
                    "true_country_label": gate.COUNTRY_LABELS[country],
                    "analysis_weight": gate.survey_weight(row),
                    "balanced_random_country_label": assigned_labels[row["uniqid"]],
                    "evidence": [
                        gate.serialise_item(metadata, variable, row[variable])
                        for variable in gate.EVIDENCE_ITEMS
                    ],
                    "shuffled_evidence": [
                        gate.serialise_item(metadata, variable, shuffled[variable])
                        for variable in gate.EVIDENCE_ITEMS
                    ],
                    "targets": [
                        gate.serialise_item(metadata, variable, row[variable])
                        for variable in targets
                    ],
                    "target_options": {
                        variable: [
                            {
                                "answer_code": code,
                                "answer": gate.answer_text(metadata, variable, code),
                            }
                            for code in gate.positive_codes(metadata, variable)
                        ]
                        for variable in targets
                    },
                }
            if pair_id is not None:
                profile["pair_id"] = pair_id
                profile["donor_profile_id"] = profile_id_by_uniqid[shuffled["uniqid"]]
            profiles.append(profile)
    return profiles, eligible_counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    dataset_dir = Path(__file__).resolve().parents[1]
    gate_path = dataset_dir / "analysis" / "pilot_gate_report.json"
    panel_path = dataset_dir / "derived" / "country_panel_6.csv"
    metadata_path = dataset_dir / "metadata" / "variable_metadata.json"
    report = json.loads(gate_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["variables"]

    high, representative, controls = select_confirmatory_targets(report["results"])
    target_batches = []
    for index in range(4):
        high_slice = high[index * 2 : min(index * 2 + 2, len(high))]
        representative_count = 4 - len(high_slice)
        representative_start = sum(
            4 - len(high[prior * 2 : min(prior * 2 + 2, len(high))])
            for prior in range(index)
        )
        representative_slice = representative[
            representative_start : representative_start + representative_count
        ]
        target_batches.append(high_slice + representative_slice + [controls[index]])
    targets = [variable for batch in target_batches for variable in batch]
    required = {"uniqid", "cntry_AN", *gate.EVIDENCE_ITEMS, *targets}
    confirmatory_rows = []
    with panel_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Panel is missing variables: {sorted(missing)}")
        for row in reader:
            if gate.respondent_split(row["uniqid"]) == "confirmatory":
                confirmatory_rows.append(row)

    profiles, eligible_counts = build_profiles(
        confirmatory_rows,
        metadata,
        targets,
        label_assignment="balanced_seeded",
        shuffled_pairing="reciprocal_within_country_label",
    )
    output_path = dataset_dir / "derived" / "confirmatory_profiles_6x84.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile, ensure_ascii=False) + "\n")

    assignment_manifest = {
        "version": "confirmatory-label-assignment-v1",
        "status": "prospectively frozen before confirmatory LLM outcomes",
        "frozen_at": CONFIRMATORY_FROZEN_AT,
        "method": "stable-ID hash rank within each true-country stratum, then equal-size label blocks",
        "seed": LABEL_RANDOMIZATION_SEED,
        "hash": "SHA-256",
        "hash_domain": "balanced-country-label",
        "strata": "true_country_code",
        "profiles_per_stratum": PROFILES_PER_COUNTRY,
        "labels_per_stratum": len(gate.COUNTRIES),
        "profiles_per_displayed_label": PROFILES_PER_COUNTRY // len(gate.COUNTRIES),
        "independent_of": [
            "input row order",
            "selected-profile output order",
            "profile-selection hash namespace",
        ],
        "assignments": [
            {
                "profile_id": profile["profile_id"],
                "true_country_code": profile["true_country_code"],
                "displayed_country_label": profile["balanced_random_country_label"],
            }
            for profile in sorted(profiles, key=lambda value: value["profile_id"])
        ],
    }
    assignment_path = dataset_dir / "analysis" / "confirmatory_label_assignment_v1.json"
    assignment_path.write_text(
        json.dumps(assignment_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    donor_assignment = {
        "version": "confirmatory-reciprocal-donor-pairs-v2",
        "status": "prospectively frozen before confirmatory LLM outcomes",
        "frozen_at": CACHED_DESIGN_FROZEN_AT,
        "method": (
            "stable hash rank within each true-country x displayed-country-label "
            "stratum, followed by adjacent reciprocal pairs"
        ),
        "hash_domain": "reciprocal-donor-pair",
        "profiles_per_stratum": PROFILES_PER_COUNTRY // len(gate.COUNTRIES),
        "pairs_per_stratum": PROFILES_PER_COUNTRY // len(gate.COUNTRIES) // 2,
        "assignments": [
            {
                "profile_id": profile["profile_id"],
                "true_country_code": profile["true_country_code"],
                "displayed_country_label": profile["balanced_random_country_label"],
                "pair_id": profile["pair_id"],
                "donor_profile_id": profile["donor_profile_id"],
            }
            for profile in sorted(profiles, key=lambda value: value["profile_id"])
        ],
    }
    donor_assignment_path = (
        dataset_dir / "analysis" / "confirmatory_donor_pair_assignment_v2.json"
    )
    donor_assignment_path.write_text(
        json.dumps(donor_assignment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    by_id = {row["variable"]: row for row in report["results"]}
    manifest = {
        "status": "prospectively frozen before confirmatory LLM outcomes",
        "frozen_at": CONFIRMATORY_FROZEN_AT,
        "freeze_record": {
            "confirmatory_llm_outcomes_observed": False,
            "gate_c_measurement_pilot_observed": True,
            "basis": (
                "item selection was fixed from the development Gate; the completed "
                "Gate C panel was used only to validate measurement and model-family "
                "feasibility before this formal freeze"
            ),
        },
        "selection_source": "development Gate only; no confirmatory outcome used for item selection",
        "seed": SEED,
        "label_randomization": {
            "version": assignment_manifest["version"],
            "seed": LABEL_RANDOMIZATION_SEED,
            "method": assignment_manifest["method"],
            "assignment_file": assignment_path.relative_to(dataset_dir).as_posix(),
            "assignment_file_sha256": sha256(assignment_path),
            "profiles_per_displayed_label_within_true_country": (
                PROFILES_PER_COUNTRY // len(gate.COUNTRIES)
            ),
        },
        "cached_shuffled_evidence_design": {
            "version": donor_assignment["version"],
            "frozen_at": CACHED_DESIGN_FROZEN_AT,
            "assignment_file": donor_assignment_path.relative_to(dataset_dir).as_posix(),
            "assignment_file_sha256": sha256(donor_assignment_path),
            "method": donor_assignment["method"],
            "pairs": len({profile["pair_id"] for profile in profiles}),
            "pairs_per_true_country_and_displayed_label": 7,
            "reuse_rule": (
                "a target profile's shuffled-evidence logical cell reuses its "
                "reciprocal donor's correct-evidence response under the same "
                "condition, batch, provider and model only after rendered prompt "
                "bytes and SHA-256 are identical"
            ),
        },
        "thresholds": {
            "high_relative_country_gain": HIGH_RELATIVE_GAIN,
            "control_relative_ci_upper": CONTROL_RELATIVE_UPPER,
            "beta_r_equivalence_margin": BETA_R_EQUIVALENCE_MARGIN,
            "beta_r_equivalence_rule": (
                "the profile/reference bootstrap 95% CI must be wholly inside "
                "[-0.05, 0.05]; conservative CI-containment secondary analysis"
            ),
            "evidence_gain_ci_low": "> 0 for every target",
            "excluded_country_referential_targets": sorted(EXCLUDED_TARGETS),
        },
        "target_strata": {
            "high_country_value": high,
            "representative": representative,
            "equivalence_controls": controls,
        },
        "target_batches": {
            f"B{index + 1:02d}": batch for index, batch in enumerate(target_batches)
        },
        "target_metrics": {
            variable: {
                "evidence_gain_ci_low": by_id[variable]["evidence_gain_ci_low"],
                "country_incremental_gain": by_id[variable]["country_incremental_gain"],
                "country_gain_ci_low": by_id[variable]["country_gain_ci_low"],
                "country_gain_ci_high": by_id[variable]["country_gain_ci_high"],
                "relative_country_gain": relative_gain(by_id[variable], "country_incremental_gain"),
                "relative_country_gain_ci_upper": relative_gain(by_id[variable], "country_gain_ci_high"),
            }
            for variable in targets
        },
        "profiles": {
            "count": len(profiles),
            "per_country": PROFILES_PER_COUNTRY,
            "complete_eligible_by_country": eligible_counts,
            "restricted_file": output_path.relative_to(dataset_dir).as_posix(),
            "restricted_file_sha256": sha256(output_path),
            "contains_source_uniqid": False,
            "analysis_weight_kept_local": True,
        },
        "balanced_random_labels_by_true_country": {
            country: dict(
                Counter(
                    profile["balanced_random_country_label"]
                    for profile in profiles
                    if profile["true_country_code"] == country
                )
            )
            for country in gate.COUNTRIES
        },
    }
    manifest_path = dataset_dir / "analysis" / "confirmatory_manifest_v2.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pilot_rows = []
    with panel_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if gate.respondent_split(row["uniqid"]) == "pilot":
                pilot_rows.append(row)
    prompt_pilot_profiles, _ = build_profiles(
        pilot_rows,
        metadata,
        targets,
        profiles_per_country=2,
        profile_prefix="V2",
        label_assignment="legacy_cyclic",
    )
    prompt_pilot_path = dataset_dir / "derived" / "prompt_pilot_v2_mixed_profiles_6x2.jsonl"
    with prompt_pilot_path.open("w", encoding="utf-8", newline="\n") as handle:
        for profile in prompt_pilot_profiles:
            handle.write(json.dumps(profile, ensure_ascii=False) + "\n")

    # Gate C uses held-out pilot-split records, but its targets and batching must
    # match the frozen confirmatory design. Two mixed five-item batches give
    # 72 profiles x 6 cells x 2 batches x 5 models = 4,320 calls.
    gate_c_batches = target_batches[:2]
    gate_c_targets = [variable for batch in gate_c_batches for variable in batch]
    gate_c_profiles, gate_c_eligible_counts = build_profiles(
        pilot_rows,
        metadata,
        gate_c_targets,
        profiles_per_country=GATE_C_PROFILES_PER_COUNTRY,
        profile_prefix="GATEC",
        label_assignment="legacy_cyclic",
    )
    gate_c_path = dataset_dir / "derived" / "gate_c_profiles_6x12.jsonl"
    with gate_c_path.open("w", encoding="utf-8", newline="\n") as handle:
        for profile in gate_c_profiles:
            handle.write(json.dumps(profile, ensure_ascii=False) + "\n")
    gate_c_target_set = set(gate_c_targets)
    gate_c_manifest = {
        "status": "frozen Gate C measurement pilot; not confirmatory outcomes",
        "label_assignment": {
            "method": "legacy deterministic cyclic assignment",
            "randomized": False,
            "scope": "completed Gate C pilot only; do not reuse for confirmatory inference",
        },
        "selection_source": manifest["selection_source"],
        "parent_confirmatory_manifest_sha256": sha256(manifest_path),
        "target_strata": {
            stratum: [variable for variable in variables if variable in gate_c_target_set]
            for stratum, variables in manifest["target_strata"].items()
        },
        "target_batches": {
            f"B{index + 1:02d}": batch
            for index, batch in enumerate(gate_c_batches)
        },
        "profiles": {
            "count": len(gate_c_profiles),
            "per_country": GATE_C_PROFILES_PER_COUNTRY,
            "complete_eligible_by_country": gate_c_eligible_counts,
            "restricted_file": gate_c_path.relative_to(dataset_dir).as_posix(),
            "restricted_file_sha256": sha256(gate_c_path),
            "contains_source_uniqid": False,
            "split": "pilot",
        },
    }
    gate_c_manifest_path = dataset_dir / "analysis" / "gate_c_manifest_v1.json"
    gate_c_manifest_path.write_text(
        json.dumps(gate_c_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"High: {', '.join(high)}")
    print(f"Representative: {', '.join(representative)}")
    print(f"Controls: {', '.join(controls)}")
    print(f"Profiles: {len(profiles)} ({PROFILES_PER_COUNTRY} per country)")
    print(f"Manifest: {manifest_path}")
    print(f"Label assignment manifest: {assignment_path}")
    print(f"Restricted profiles: {output_path}")
    print(f"Prompt-pilot profiles: {prompt_pilot_path}")
    print(
        f"Gate C profiles: {len(gate_c_profiles)} "
        f"({GATE_C_PROFILES_PER_COUNTRY} per country)"
    )
    print(f"Gate C manifest: {gate_c_manifest_path}")
    print(f"Gate C restricted profiles: {gate_c_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
