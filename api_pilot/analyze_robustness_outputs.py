#!/usr/bin/env python3
"""Validate and analyse one frozen Gate C robustness output.

Only aggregate statistics are written.  Respondent-derived prompts, answers and
row-level forecasts remain in the restricted JSONL files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = ROOT / "api_pilot" / "restricted_outputs" / "gate_c_v3_complete.jsonl"
BOOTSTRAP_SEED = 2026072503


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (
        ordered[upper] - ordered[lower]
    )


def _balanced_macro(
    rows: list[dict[str, Any]],
    field: str,
    multiplicities: dict[str, Counter[str]] | None = None,
) -> float:
    """Survey-weighted country means, then equal unit/provider/country macro."""
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        unit = str(row.get("macro_unit") or row["item_id"])
        cells[(row["provider"], row["true_country_code"], unit)].append(row)
    if not cells:
        return math.nan
    cell_means: dict[tuple[str, str, str], float] = {}
    for key, values in cells.items():
        country = key[1]
        numerator = denominator = 0.0
        for row in values:
            count = (
                1
                if multiplicities is None
                else multiplicities[country][row["profile_id"]]
            )
            weight = float(row["analysis_weight"]) * count
            numerator += float(row[field]) * weight
            denominator += weight
        if denominator <= 0:
            raise ValueError(f"Empty weighted robustness cell: {key}")
        cell_means[key] = numerator / denominator
    provider_country: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (provider, country, _unit), value in cell_means.items():
        provider_country[(provider, country)].append(value)
    provider_means: dict[str, list[float]] = defaultdict(list)
    for (provider, _country), values in provider_country.items():
        provider_means[provider].append(sum(values) / len(values))
    return sum(
        sum(values) / len(values) for values in provider_means.values()
    ) / len(provider_means)


def stratified_profile_bootstrap(
    rows: list[dict[str, Any]], field: str, replicates: int
) -> dict[str, Any]:
    point = _balanced_macro(rows, field)
    profiles_by_country: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        profiles_by_country[row["true_country_code"]].append(row["profile_id"])
    profiles_by_country = {
        country: sorted(set(profile_ids))
        for country, profile_ids in profiles_by_country.items()
    }
    if not profiles_by_country:
        return {
            "mean": point,
            "ci95": [math.nan, math.nan],
            "profiles": 0,
            "bootstrap_replicates": replicates,
        }
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(replicates):
        multiplicities = {
            country: Counter(rng.choice(profile_ids) for _ in profile_ids)
            for country, profile_ids in profiles_by_country.items()
        }
        draws.append(_balanced_macro(rows, field, multiplicities))
    return {
        "mean": point,
        "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "profiles": sum(len(values) for values in profiles_by_country.values()),
        "profiles_by_true_country": {
            country: len(values) for country, values in profiles_by_country.items()
        },
        "bootstrap_replicates": replicates,
        "aggregation": "survey-weighted within country; equal unit/provider/country macro",
        "bootstrap": "profiles stratified within true country; shared deterministic draws",
    }


def summarise(
    rows: list[dict[str, Any]], field: str, replicates: int, margins: Iterable[float]
) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    bootstrap = stratified_profile_bootstrap(rows, field, replicates)
    upper = bootstrap["ci95"][1]
    return {
        **bootstrap,
        "observations": len(values),
        "median": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "maximum": max(values) if values else math.nan,
        "equivalence_by_margin": {
            str(margin): bool(math.isfinite(upper) and upper < margin)
            for margin in margins
        },
    }


def gate_decision(
    manifest: dict[str, Any], metrics: dict[str, Any] | None
) -> dict[str, Any]:
    """Apply the frozen robustness gate without manual interpretation."""
    contract = manifest["analysis_contract"]
    margin = float(contract["gate_margin"])
    direction_floor = float(contract["gate_direction_agreement_lower_bound"])
    checks: dict[str, dict[str, Any]] = {}
    if metrics is None:
        return {
            "status": "pending",
            "passes": False,
            "margin": margin,
            "direction_agreement_lower_bound": direction_floor,
            "checks": checks,
        }

    if contract.get("gate_forecast_tv_overall", True):
        summary = metrics["forecast_tv"]["overall"]
        upper = float(summary["ci95"][1])
        checks["forecast_tv_overall"] = {
            "estimate": float(summary["mean"]),
            "ci95_upper": upper,
            "required_upper_below": margin,
            "passes": bool(math.isfinite(upper) and upper < margin),
        }

    contrasts = metrics.get("treatment_contrast_stability", {})
    for name in contract.get("gate_treatment_contrasts", []):
        check_name = f"contrast_{name}"
        if name not in contrasts:
            checks[check_name] = {
                "required_upper_below": margin,
                "passes": False,
                "reason": "required contrast is absent",
            }
            continue
        summary = contrasts[name]
        upper = float(summary["ci95"][1])
        direction_low = float(summary["direction_agreement_ci95"][0])
        checks[check_name] = {
            "estimate": float(summary["mean"]),
            "ci95_upper": upper,
            "required_upper_below": margin,
            "direction_agreement_rate": float(
                summary["direction_agreement_rate"]
            ),
            "direction_agreement_ci95_lower": direction_low,
            "required_direction_ci95_lower_above": direction_floor,
            "passes": bool(
                math.isfinite(upper)
                and upper < margin
                and math.isfinite(direction_low)
                and direction_low > direction_floor
            ),
        }

    if not checks:
        raise ValueError("Frozen robustness contract contains no gate checks")
    passes = all(bool(value["passes"]) for value in checks.values())
    return {
        "status": "pass" if passes else "fail",
        "passes": passes,
        "margin": margin,
        "direction_agreement_lower_bound": direction_floor,
        "checks": checks,
        "rule": contract["gate_rule"],
    }


def probability_map(row: dict[str, Any], item_id: str) -> dict[str, float]:
    option_codes = row["target_option_codes"][item_id]
    probabilities = row["parsed_forecasts"][item_id]
    if len(option_codes) != len(probabilities):
        raise ValueError(f"Option/probability length mismatch for {item_id}")
    result = {str(code): float(value) for code, value in zip(option_codes, probabilities)}
    if len(result) != len(option_codes):
        raise ValueError(f"Duplicate option codes for {item_id}")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in result.values()):
        raise ValueError(f"Invalid probability for {item_id}")
    if abs(sum(result.values()) - 1.0) > 0.0100001:
        raise ValueError(f"Probability sum outside tolerance for {item_id}")
    return result


def total_variation(first: dict[str, float], second: dict[str, float]) -> float:
    if set(first) != set(second):
        raise ValueError("Paired forecasts use different answer-code sets")
    return 0.5 * sum(abs(first[code] - second[code]) for code in first)


def validate_new_rows(
    manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[
    dict[tuple[str, str, str, str, str], dict[str, Any]],
    Counter[tuple[str, str]],
]:
    expected_models = manifest["provider_models"]
    expected_units = {
        (
            unit["profile_id"], unit["condition"],
            unit["evidence_variant"], unit["batch_id"],
        )
        for unit in manifest["task_units"]
    }
    index = {}
    failures = Counter()
    for row in rows:
        provider = row.get("provider")
        if provider not in expected_models:
            raise ValueError(f"Unexpected provider in robustness output: {provider}")
        if row.get("model_requested") != expected_models[provider]:
            raise ValueError(f"Unexpected requested model for {provider}")
        if row.get("prompt_version") != manifest["prompt"]["version"]:
            raise ValueError("Robustness output prompt version does not match manifest")
        unit = (
            row.get("profile_id"), row.get("condition"),
            row.get("evidence_variant"), row.get("batch_id"),
        )
        if unit not in expected_units:
            raise ValueError(f"Unexpected robustness task unit: {unit}")
        key = (provider, *unit)
        if key in index:
            raise ValueError(f"Duplicate robustness result: {key}")
        if not row.get("success"):
            failures[(provider, row.get("error_type") or "unknown")] += 1
            continue
        index[key] = row
    return index, failures


def core_index(rows: list[dict[str, Any]]) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str, str, str, str], dict[str, Any]],
]:
    by_call = {}
    by_unit = {}
    for row in rows:
        if not row.get("success"):
            continue
        by_call[row["call_id"]] = row
        key = (
            row["provider"], row["profile_id"], row["condition"],
            row["evidence_variant"], row["batch_id"],
        )
        if key in by_unit:
            raise ValueError(f"Duplicate core task unit: {key}")
        by_unit[key] = row
    return by_call, by_unit


def paired_rows(
    manifest: dict[str, Any],
    new_index: dict[tuple[str, str, str, str, str], dict[str, Any]],
    core_by_call: dict[str, dict[str, Any]],
    core_by_unit: dict[tuple[str, str, str, str, str], dict[str, Any]],
    profile_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    single_map = manifest.get("single_item_batch_to_item", {})
    item_to_core = manifest.get("item_to_core_batch", {})
    for key, variant in sorted(new_index.items()):
        provider, profile_id, condition, evidence, batch_id = key
        if manifest["task_mode"] == "repeat_identical":
            source_call_id = variant.get("source_call_id")
            if not source_call_id or source_call_id not in core_by_call:
                raise ValueError("Prompt-identical repeat lacks an exact core source call")
            core = core_by_call[source_call_id]
            item_ids = variant["target_ids"]
        elif manifest.get("single_item_batch_to_item"):
            item_id = single_map[batch_id]
            core_batch = item_to_core[item_id]
            core = core_by_unit[(provider, profile_id, condition, evidence, core_batch)]
            item_ids = [item_id]
        else:
            core = core_by_unit[(provider, profile_id, condition, evidence, batch_id)]
            item_ids = variant["target_ids"]
        for item_id in item_ids:
            first = probability_map(core, item_id)
            second = probability_map(variant, item_id)
            pairs.append(
                {
                    "provider": provider,
                    "profile_id": profile_id,
                    "true_country_code": profile_by_id[profile_id]["true_country_code"],
                    "analysis_weight": float(
                        profile_by_id[profile_id]["analysis_weight"]
                    ),
                    "condition": condition,
                    "evidence_variant": evidence,
                    "item_id": item_id,
                    "macro_unit": f"{condition}|{evidence}|{item_id}",
                    "core": first,
                    "variant": second,
                    "tv": total_variation(first, second),
                }
            )
    return pairs


def treatment_contrasts(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in pairs:
        key = (
            row["provider"], row["profile_id"], row["evidence_variant"], row["item_id"]
        )
        grouped[key][row["condition"]] = row
    results = []
    for key, conditions in grouped.items():
        for contrast_name, treated, control in (
            ("MO_minus_MR", "MO", "MR"),
            ("MV_minus_M0", "MV", "M0"),
        ):
            if treated not in conditions or control not in conditions:
                continue
            treated_row = conditions[treated]
            control_row = conditions[control]
            codes = set(treated_row["core"])
            if not (
                codes == set(control_row["core"])
                == set(treated_row["variant"])
                == set(control_row["variant"])
            ):
                raise ValueError("Treatment contrast option codes do not align")
            core_delta = {
                code: treated_row["core"][code] - control_row["core"][code]
                for code in codes
            }
            variant_delta = {
                code: treated_row["variant"][code] - control_row["variant"][code]
                for code in codes
            }
            discrepancy = 0.5 * sum(
                abs(core_delta[code] - variant_delta[code]) for code in codes
            )
            inner = sum(core_delta[code] * variant_delta[code] for code in codes)
            results.append(
                {
                    "provider": key[0],
                    "profile_id": key[1],
                    "true_country_code": treated_row["true_country_code"],
                    "analysis_weight": treated_row["analysis_weight"],
                    "evidence_variant": key[2],
                    "item_id": key[3],
                    "contrast": contrast_name,
                    "macro_unit": f"{contrast_name}|{key[2]}|{key[3]}",
                    "contrast_discrepancy": discrepancy,
                    "direction_agreement": inner > 0,
                }
            )
    return results


def aggregate(
    manifest: dict[str, Any], pairs: list[dict[str, Any]], replicates: int
) -> dict[str, Any]:
    margins = manifest["analysis_contract"]["equivalence_tv_margin_grid"]
    result: dict[str, Any] = {
        "forecast_tv": {
            "overall": summarise(pairs, "tv", replicates, margins),
            "by_provider": {},
            "by_condition": {},
        }
    }
    for provider in sorted({row["provider"] for row in pairs}):
        subset = [row for row in pairs if row["provider"] == provider]
        result["forecast_tv"]["by_provider"][provider] = summarise(
            subset, "tv", replicates, margins
        )
    for condition in sorted({row["condition"] for row in pairs}):
        subset = [row for row in pairs if row["condition"] == condition]
        result["forecast_tv"]["by_condition"][condition] = summarise(
            subset, "tv", replicates, margins
        )
    contrasts = treatment_contrasts(pairs)
    if contrasts:
        result["treatment_contrast_stability"] = {}
        for contrast in sorted({row["contrast"] for row in contrasts}):
            subset = [row for row in contrasts if row["contrast"] == contrast]
            agreement = summarise(
                subset, "direction_agreement", replicates, ()
            )
            result["treatment_contrast_stability"][contrast] = {
                **summarise(subset, "contrast_discrepancy", replicates, margins),
                "direction_agreement_rate": agreement["mean"],
                "direction_agreement_ci95": agreement["ci95"],
            }
    return result


def analyse(
    manifest_path: Path,
    output_path: Path | None,
    core_path: Path,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_output = ROOT / manifest["immutable_output_path"]
    if output_path is None:
        output_path = expected_output
    if output_path.resolve() != expected_output.resolve():
        raise ValueError("Analysis output path does not match frozen manifest")
    if sha256(ROOT / manifest["source"]["profile_file"]) != manifest["source"]["profile_file_sha256"]:
        raise ValueError("Frozen profile source hash mismatch")
    if sha256(ROOT / manifest["source"]["core_manifest_file"]) != manifest["source"]["core_manifest_sha256"]:
        raise ValueError("Frozen core manifest hash mismatch")
    if sha256(core_path) != manifest["source"]["core_output_sha256"]:
        raise ValueError("Immutable core output hash mismatch")
    if sha256(ROOT / manifest["prompt"]["file"]) != manifest["prompt"]["sha256"]:
        raise ValueError("Frozen prompt hash mismatch")
    base = {
        "version": "gate-c-robustness-analysis-v2",
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(manifest_path),
        "experiment_id": manifest["experiment_id"],
        "expected": manifest["expected"],
        "immutable_output_path": manifest["immutable_output_path"],
        "scope": (
            "Gate C measurement prerequisite on the frozen 72-profile pilot panel; "
            "not a formal 504-profile outcome analysis"
        ),
        "design_estimand_stability": manifest["analysis_contract"].get(
            "design_estimand_stability"
        ),
    }
    if not output_path.exists():
        return {
            **base,
            "status": "pending_inference",
            "complete": False,
            "gate_decision": gate_decision(manifest, None),
        }
    new_rows = load_jsonl(output_path)
    new_index, failures = validate_new_rows(manifest, new_rows)
    expected_success = manifest["expected"]["total_calls"]
    complete = len(new_index) == expected_success and not failures
    result = {
        **base,
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "observed_successes": len(new_index),
        "failures": {"/".join(map(str, key)): value for key, value in failures.items()},
    }
    if not complete:
        result["gate_decision"] = gate_decision(manifest, None)
        return result
    core_rows = load_jsonl(core_path)
    core_by_call, core_by_unit = core_index(core_rows)
    profiles = load_jsonl(ROOT / manifest["source"]["profile_file"])
    profile_by_id = {row["profile_id"]: row for row in profiles}
    pairs = paired_rows(
        manifest, new_index, core_by_call, core_by_unit, profile_by_id
    )
    result["paired_forecasts"] = len(pairs)
    result["metrics"] = aggregate(manifest, pairs, bootstrap_replicates)
    result["gate_decision"] = gate_decision(manifest, result["metrics"])
    result["interpretation"] = (
        "The diagnostics use the design-aligned survey-weighted equal macro and a "
        "country-stratified shared-draw profile bootstrap. They do not directly "
        "re-estimate beta_O, beta_O-beta_R or UtilityV under each protocol variant; "
        "that remaining limitation must be reported and these gates cannot replace "
        "the formal estimand analysis."
    )
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Gate C Robustness: {result['experiment_id']}",
        "",
        f"- Status: **{result['status']}**",
        f"- Expected calls: **{result['expected']['total_calls']:,}**",
        f"- Output: `{result['immutable_output_path']}`",
    ]
    if not result.get("complete"):
        limitation = result.get("design_estimand_stability") or {}
        if limitation.get("implemented_in_this_gate") is False:
            lines.append(
                "- Limitation: this Gate does not directly re-estimate `beta_O`, "
                "`beta_O-beta_R`, or `UtilityV` under each variant."
            )
        return "\n".join(lines) + "\n"
    overall = result["metrics"]["forecast_tv"]["overall"]
    lines.extend(
        [
            f"- Paired item forecasts: **{result['paired_forecasts']:,}**",
            f"- Mean forecast TV: **{overall['mean']:.4f}** "
            f"(profile-cluster 95% CI {overall['ci95'][0]:.4f} to {overall['ci95'][1]:.4f})",
            f"- Frozen gate decision: **{result['gate_decision']['status']}**",
            "",
        ]
    )
    contrasts = result["metrics"].get("treatment_contrast_stability", {})
    if contrasts:
        lines.extend(
            [
                "| Contrast | Mean discrepancy | 95% CI | Direction agreement (95% CI) |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in contrasts.items():
            lines.append(
                f"| {name} | {values['mean']:.4f} | "
                f"[{values['ci95'][0]:.4f}, {values['ci95'][1]:.4f}] | "
                f"{values['direction_agreement_rate']:.1%} "
                f"[{values['direction_agreement_ci95'][0]:.1%}, "
                f"{values['direction_agreement_ci95'][1]:.1%}] |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--core-output", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    result = analyse(
        manifest_path,
        args.output.resolve() if args.output else None,
        args.core_output.resolve(),
        args.bootstrap_replicates,
    )
    json_path = args.json_report or (
        ROOT / "api_pilot" / "outputs" /
        f"{result['experiment_id']}_robustness_analysis.json"
    )
    markdown_path = args.markdown_report or (
        ROOT / "api_pilot" / f"{result['experiment_id'].upper()}_ROBUSTNESS.md"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"status={result['status']} json={json_path} markdown={markdown_path}")
    # Incomplete, non-terminal checkpoint rows remain resumable. The runner,
    # rather than the offline analyser, decides whether a call is terminal.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
