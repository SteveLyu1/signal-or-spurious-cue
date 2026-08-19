#!/usr/bin/env python3
"""Key-secondary H1/H2 consistency analysis for the clean replacement panel."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot import analyze_model_country_preferences as clean  # noqa: E402


OUT_JSON = ROOT / "api_pilot/outputs/clean_replacement_primary_consistency.json"
OUT_CSV = ROOT / "api_pilot/outputs/final_tables/table_17_clean_replacement_consistency.csv"
PRIMARY_JSON = ROOT / "api_pilot/outputs/independent_correction_replication_v2_analysis.json"
HIGH_ITEMS = ("A124_09", "E025", "A124_08", "A124_03", "A030", "E027", "A029")


def interval(samples: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"ci_low": float(low), "ci_high": float(high)}


def one_sided_positive_p(samples: np.ndarray) -> float:
    return float((1.0 + np.count_nonzero(samples <= 0.0)) / (len(samples) + 1.0))


def summarize(
    observed: dict[str, np.ndarray], samples: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    definitions = {
        "beta_o": ("direction_o_pooled", 1.0),
        "beta_r": ("direction_r_pooled", 1.0),
        "beta_o_minus_beta_r": ("direction_o_minus_direction_r_pooled", 1.0),
        "utility_v": ("utility_v", 1.0),
        "regret_o": ("mo_regret", 1.0),
        "regret_r": ("mr_regret", 1.0),
    }
    output: dict[str, dict[str, float]] = {}
    for name, (source, sign) in definitions.items():
        estimate = float(np.mean(observed[source]) * sign)
        draws = np.mean(samples[source], axis=(1, 2)) * sign
        output[name] = {
            "estimate": estimate,
            **interval(draws),
            "bootstrap_p_greater_zero": one_sided_positive_p(draws),
        }
    return output


def item_equal_metrics(
    view: clean.AnalysisView,
    anchors: dict[str, dict[str, list[float]]],
    *,
    samples: clean.EvaluationSamples | None = None,
) -> dict[str, np.ndarray]:
    """Average item-specific estimands, matching the manuscript macro contract."""
    per_item = []
    for item in view.item_ids:
        single = clean.AnalysisView(
            view.name,
            (item,),
            {item: view.panels[item]},
            view.role,
        )
        per_item.append(
            clean.calculate_metrics(
                single,
                anchors,
                samples=samples,
                include_nearest=False,
                include_js=False,
            )
        )
    return {
        metric: np.mean([values[metric] for values in per_item], axis=0)
        for metric in clean.BOOTSTRAP_METRICS
    }


def bootstrap_item_equal_view(
    view: clean.AnalysisView,
    anchors: dict[str, dict[str, list[float]]],
    resampler: Any,
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    observed = item_equal_metrics(view, anchors)
    draws = {
        metric: np.empty((replicates, *observed[metric].shape), dtype=float)
        for metric in clean.BOOTSTRAP_METRICS
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        evaluation_samples = clean.stratified_samples(view, rng)
        sampled_anchors = resampler(rng)
        values = item_equal_metrics(view, sampled_anchors, samples=evaluation_samples)
        for metric in clean.BOOTSTRAP_METRICS:
            draws[metric][replicate] = values[metric]
    return observed, draws


def write_csv(path: Path, estimates: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "metric", "estimate", "ci_low", "ci_high",
                "bootstrap_p_greater_zero", "analysis_role",
            ),
        )
        writer.writeheader()
        for metric, values in estimates.items():
            writer.writerow({
                "metric": metric,
                **values,
                "analysis_role": "key_secondary_post_review_clean_replacement_consistency",
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026072901)
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")

    views, source_audit, resampler, anchors = clean.build_views()
    full = views["clean_hybrid_all20"]
    view = clean.AnalysisView(
        "clean_replacement_high7",
        HIGH_ITEMS,
        {item: full.panels[item] for item in HIGH_ITEMS},
        "key_secondary_post_review_clean_replacement_consistency",
    )
    observed, samples = bootstrap_item_equal_view(
        view,
        anchors,
        resampler,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    estimates = summarize(observed, samples)

    primary = json.loads(PRIMARY_JSON.read_text(encoding="utf-8"))
    primary_inf = primary["result"]["inference"]
    primary_values = {
        "beta_o": primary_inf["direction_bootstrap"]["beta_o"]["estimate"],
        "beta_r": primary_inf["direction_bootstrap"]["beta_r"]["estimate"],
        "beta_o_minus_beta_r": primary_inf["direction_bootstrap"]["beta_diff"]["estimate"],
        "utility_v": primary_inf["utility_h2"]["estimate"],
    }
    consistency = {
        metric: {
            "primary_estimate": float(primary_values[metric]),
            "clean_replacement_estimate": estimates[metric]["estimate"],
            "same_sign": bool(np.sign(primary_values[metric]) == np.sign(estimates[metric]["estimate"])),
            "primary_estimate_inside_clean_replacement_ci": bool(
                estimates[metric]["ci_low"] <= primary_values[metric] <= estimates[metric]["ci_high"]
            ),
        }
        for metric in primary_values
    }
    item_contract = {
        item: {
            "source": view.panels[item].source,
            "profiles": len(view.panels[item].profile_ids),
        }
        for item in HIGH_ITEMS
    }
    if {item for item, value in item_contract.items() if value["profiles"] == 504} != {"E025", "E027"}:
        raise ValueError("Unexpected 504-record target coverage")
    if {item for item, value in item_contract.items() if value["profiles"] == 72} != set(HIGH_ITEMS) - {"E025", "E027"}:
        raise ValueError("Unexpected corrected-replacement target coverage")
    if source_audit["hybrid_item_vectors"]["affected_original_excluded"] <= 0:
        raise ValueError("Superseded affected-item vectors were not audited out")

    output: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "analysis_role": "key_secondary_post_review_clean_replacement_consistency",
        "claim_boundary": (
            "Two unaffected high-country-value targets use 504-record forecasts; five affected targets "
            "use corrected-wording forecasts from a balanced 72-record subset. Targets are equally "
            "weighted. This is not a uniform 504-record corrected-protocol experiment."
        ),
        "items": item_contract,
        "source_audit": {
            "affected_old_wording_vectors_used": 0,
            "affected_old_wording_vectors_excluded": source_audit["hybrid_item_vectors"]["affected_original_excluded"],
            "replacement_subset_of_original": source_audit["profiles"]["replacement_subset_of_original"],
            "human_quality_signoff": source_audit["human_quality_signoff"],
        },
        "estimand": "equal model x target x country macro-average",
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "method": (
                "nested replacement-panel cluster bootstrap: corrected-subset and remainder records "
                "resampled separately within survey country and displayed-label cells; reference "
                "anchors re-estimated on every draw"
            ),
        },
        "estimates": estimates,
        "comparison_with_uniform_primary_p72": consistency,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.csv, estimates)
    print(json.dumps({"status": "complete", "replicates": args.bootstrap_replicates, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
