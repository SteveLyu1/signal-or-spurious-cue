from __future__ import annotations

import sys
import unittest
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset" / "tools"))

import build_confirmatory_manifest as confirmatory


def row(
    variable: str,
    *,
    evidence_low: float = 0.01,
    gain: float = 0.0,
    gain_low: float = -0.001,
    gain_high: float = 0.005,
    brier: float = 0.5,
) -> dict:
    return {
        "variable": variable,
        "evidence_gain_ci_low": evidence_low,
        "country_incremental_gain": gain,
        "country_gain_ci_low": gain_low,
        "country_gain_ci_high": gain_high,
        "brier_evidence": brier,
    }


class SelectionTests(unittest.TestCase):
    def test_control_requires_equivalence_upper_bound(self) -> None:
        eligible = row("A004", gain_high=0.0049)
        too_wide = row("D078", gain_high=0.0051)
        self.assertLessEqual(
            confirmatory.relative_gain(eligible, "country_gain_ci_high"),
            confirmatory.CONTROL_RELATIVE_UPPER,
        )
        self.assertGreater(
            confirmatory.relative_gain(too_wide, "country_gain_ci_high"),
            confirmatory.CONTROL_RELATIVE_UPPER,
        )

    def test_high_requires_five_percent_relative_gain(self) -> None:
        passing = row("A029", gain=0.025, gain_low=0.01, gain_high=0.03)
        failing = row("A030", gain=0.024, gain_low=0.01, gain_high=0.03)
        self.assertGreaterEqual(
            confirmatory.relative_gain(passing, "country_incremental_gain"),
            confirmatory.HIGH_RELATIVE_GAIN,
        )
        self.assertLess(
            confirmatory.relative_gain(failing, "country_incremental_gain"),
            confirmatory.HIGH_RELATIVE_GAIN,
        )

    def test_frozen_manifest_balances_random_labels(self) -> None:
        manifest = json.loads(
            (ROOT / "dataset" / "analysis" / "confirmatory_manifest_v2.json")
            .read_text(encoding="utf-8")
        )
        expected = manifest["profiles"]["per_country"] // 6
        for counts in manifest["balanced_random_labels_by_true_country"].values():
            self.assertEqual(set(counts.values()), {expected})
        selected = {
            item_id
            for items in manifest["target_strata"].values()
            for item_id in items
        }
        self.assertTrue(confirmatory.EXCLUDED_TARGETS.isdisjoint(selected))
        assignment_path = (
            ROOT / "dataset" / manifest["label_randomization"]["assignment_file"]
        )
        self.assertEqual(
            confirmatory.sha256(assignment_path),
            manifest["label_randomization"]["assignment_file_sha256"],
        )
        assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
        self.assertEqual(len(assignments["assignments"]), 504)
        self.assertEqual(assignments["profiles_per_displayed_label"], 14)

    def test_formal_profiles_have_reciprocal_donors_within_randomization_cell(self) -> None:
        profiles = [
            json.loads(line)
            for line in (
                ROOT / "dataset" / "derived" / "confirmatory_profiles_6x84.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        by_id = {profile["profile_id"]: profile for profile in profiles}
        self.assertEqual(len(by_id), 504)
        self.assertEqual(len({profile["pair_id"] for profile in profiles}), 252)
        for profile in profiles:
            donor = by_id[profile["donor_profile_id"]]
            self.assertEqual(donor["donor_profile_id"], profile["profile_id"])
            self.assertEqual(donor["pair_id"], profile["pair_id"])
            self.assertEqual(donor["true_country_code"], profile["true_country_code"])
            self.assertEqual(
                donor["balanced_random_country_label"],
                profile["balanced_random_country_label"],
            )
            self.assertEqual(donor["evidence"], profile["shuffled_evidence"])

    def test_seeded_label_assignment_is_balanced_and_order_invariant(self) -> None:
        rows = [{"uniqid": f"id-{index:03d}"} for index in range(84)]
        forward = confirmatory.balanced_seeded_label_assignment("CN", rows)
        reverse = confirmatory.balanced_seeded_label_assignment("CN", list(reversed(rows)))
        repeated = confirmatory.balanced_seeded_label_assignment("CN", rows)

        self.assertEqual(forward, reverse)
        self.assertEqual(forward, repeated)
        self.assertEqual(set(Counter(forward.values()).values()), {14})
        self.assertEqual(
            set(forward.values()),
            set(confirmatory.gate.COUNTRY_LABELS.values()),
        )

    def test_label_randomization_uses_separate_hash_namespace(self) -> None:
        profile_selection_draw = confirmatory.stable_unit("respondent-order", "id-001")
        label_draw = confirmatory.stable_label_randomization_unit(
            "balanced-country-label", "CN", "id-001"
        )
        self.assertNotEqual(profile_selection_draw, label_draw)

    def test_seeded_assignment_rejects_unbalanced_strata(self) -> None:
        rows = [{"uniqid": f"id-{index:03d}"} for index in range(83)]
        with self.assertRaisesRegex(ValueError, "not divisible"):
            confirmatory.balanced_seeded_label_assignment("CN", rows)


if __name__ == "__main__":
    unittest.main()
