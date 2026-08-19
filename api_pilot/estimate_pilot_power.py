#!/usr/bin/env python3
"""Normal-approximation planning audit from the official-endpoint pilot.

This is a conservative planning calculation, not the final preregistered power
simulation. The final sample is frozen only after all confirmatory model
families complete their variance pilots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


Z_ALPHA_TWO_SIDED = 1.959964
Z_POWER_80 = 0.841621
PILOT_PROFILES = 72


def standard_error(metric: dict[str, float]) -> float:
    return (metric["ci_high"] - metric["ci_low"]) / (2 * Z_ALPHA_TWO_SIDED)


def required_profiles(se_pilot: float, sesoi: float) -> int:
    raw = PILOT_PROFILES * ((Z_ALPHA_TWO_SIDED + Z_POWER_80) * se_pilot / sesoi) ** 2
    return math.ceil(raw / 6) * 6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("api_pilot/outputs/survey_pilot_6x12_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("api_pilot/outputs/pilot_power_audit.json"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    metrics = summary["metrics"]["deepseek"]["high"]
    specifications = {
        "mo_minus_mr_direction_beta": {
            "metric": metrics["mo_minus_mr_direction_beta"],
            "sesoi": 0.05,
            "direction": "positive",
        },
        "verified_country_brier_utility": {
            "metric": {
                "mean": -metrics["mv_brier_delta"]["mean"],
                "ci_low": -metrics["mv_brier_delta"]["ci_high"],
                "ci_high": -metrics["mv_brier_delta"]["ci_low"],
            },
            "sesoi": 0.02,
            "direction": "positive",
        },
    }
    calculations = {}
    for name, specification in specifications.items():
        se = standard_error(specification["metric"])
        calculations[name] = {
            **specification,
            "pilot_standard_error": se,
            "required_profiles_for_80_percent_power": required_profiles(
                se, specification["sesoi"]
            ),
        }
    recommended = max(
        calculation["required_profiles_for_80_percent_power"]
        for calculation in calculations.values()
    )
    report = {
        "status": "planning approximation; replace with simulation after all-model pilot",
        "source": "DeepSeek official endpoint, 72-profile v1 development pilot",
        "method": "two-sided normal approximation; CI-derived cluster standard errors; six-country balance",
        "calculations": calculations,
        "recommended_profiles": recommended,
        "recommended_per_country": recommended // 6,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
