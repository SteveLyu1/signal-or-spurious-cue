#!/usr/bin/env python3
"""Provider-level descriptive estimates for the completed corrective replication."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot.run_gate_c_strict_inference import (
    COUNTRIES,
    build_inference_rows,
    load_jsonl,
    reference_rows_and_anchors,
)
from api_pilot.analyze_independent_correction_consequences import scalar_rows
from api_pilot.strict_inference import direction_profile_bootstrap, paired_profile_bootstrap


RESULTS = ROOT / "api_pilot/restricted_outputs/independent_correction_replication_v2.jsonl"
PROFILES = ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl"
PANEL = ROOT / "dataset/derived/country_panel_6.csv"
METADATA = ROOT / "dataset/metadata/variable_metadata.json"
PRIMARY = ["A124_09", "E025", "A124_08", "A124_03", "A030", "E027", "A029"]
OUT_JSON = ROOT / "api_pilot/outputs/independent_correction_provider_secondary.json"
OUT_CSV = ROOT / "api_pilot/outputs/final_tables/table_7_independent_correction_provider_secondary.csv"


def main() -> int:
    results = load_jsonl(RESULTS)
    profiles = load_jsonl(PROFILES)
    direction, utility, audit = build_inference_rows(
        results, profiles, PRIMARY, require_balanced_assignment=True
    )
    providers = sorted({row["provider"] for row in direction})
    consequences = scalar_rows(results, profiles, PRIMARY)
    resampler, anchors, reference_counts = reference_rows_and_anchors(PANEL, METADATA, PRIMARY)
    rows: list[dict[str, object]] = []
    for index, provider in enumerate(providers):
        provider_direction = [row for row in direction if row["provider"] == provider]
        provider_utility = [row for row in utility if row["provider"] == provider]
        d = direction_profile_bootstrap(
            provider_direction,
            anchors,
            anchor_bootstrap=resampler,
            require_anchor_reestimation=True,
            n_bootstrap=2_000,
            seed=2026072800 + index,
            expected_providers=[provider],
            expected_countries=COUNTRIES,
            expected_items=PRIMARY,
        )
        u = paired_profile_bootstrap(
            provider_utility,
            n_bootstrap=2_000,
            seed=2026072900 + index,
            expected_providers=[provider],
            expected_countries=COUNTRIES,
            expected_items=PRIMARY,
        )
        regret_o = paired_profile_bootstrap(
            [row for row in consequences["regret_o"] if row["provider"] == provider],
            n_bootstrap=2_000,
            seed=2026073000 + index,
            expected_providers=[provider],
            expected_countries=COUNTRIES,
            expected_items=PRIMARY,
        )
        regret_r = paired_profile_bootstrap(
            [row for row in consequences["regret_r"] if row["provider"] == provider],
            n_bootstrap=2_000,
            seed=2026073100 + index,
            expected_providers=[provider],
            expected_countries=COUNTRIES,
            expected_items=PRIMARY,
        )
        for metric, value in (
            ("beta_o", d.beta_o),
            ("beta_r", d.beta_r),
            ("beta_o_minus_beta_r", d.beta_diff),
            ("regret_o", regret_o),
            ("regret_r", regret_r),
            ("utility_v", u),
        ):
            rows.append({
                "provider": provider,
                "metric": metric,
                "estimate": value.estimate,
                "ci_low": value.ci_low,
                "ci_high": value.ci_high,
                "analysis_role": "post_review_provider_descriptive_secondary",
            })
    payload = {
        "status": "complete",
        "analysis_role": "provider-level descriptive secondary; no model-level multiplicity claims",
        "profiles": len(profiles),
        "items": PRIMARY,
        "providers": providers,
        "source_row_audit": audit,
        "reference_counts": reference_counts,
        "bootstrap_replicates": 2_000,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
