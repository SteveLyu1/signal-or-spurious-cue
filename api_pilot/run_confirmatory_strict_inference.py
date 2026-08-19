#!/usr/bin/env python3
"""Fail-closed strict analysis for the frozen 504-profile confirmatory run.

No API calls are made. If the immutable inference output is absent or
incomplete, an aggregate pending-status JSON is written and no hypothesis
estimate is produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot.run_gate_c_strict_inference import (
    COUNTRIES,
    build_inference_rows,
    load_jsonl,
    reference_rows_and_anchors,
    sha256,
    verify_anchor_match,
)
from api_pilot.strict_inference import run_strict_confirmatory_inference
from api_pilot.survey_pilot_runner import (
    build_prompt,
    load_prompt_bundle,
    load_task_manifest,
    make_call_id,
    sha256_text,
)


OFFICIAL_MODELS = {
    "openai": "gpt-5.4",
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.6-flash",
    "mistral": "mistral-medium-3-5",
    "qwen": "qwen3.7-max-2026-06-08",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for row in load_jsonl(path):
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Result row lacks a call_id")
        latest[call_id] = row
    return latest


def batch_profile(
    profile: dict[str, Any], target_ids: list[str]
) -> dict[str, Any]:
    targets = {row["item_id"]: row for row in profile["targets"]}
    if not all(item in targets for item in target_ids):
        raise ValueError(f"Profile lacks frozen targets: {profile['profile_id']}")
    value = dict(profile)
    value["targets"] = [targets[item] for item in target_ids]
    value["target_options"] = {
        item: profile["target_options"][item] for item in target_ids
    }
    return value


def expected_call_map(
    profiles: list[dict[str, Any]],
    analysis_manifest: dict[str, Any],
    task_manifest: dict[str, Any],
    prompt_path: Path,
) -> dict[str, dict[str, Any]]:
    bundle = load_prompt_bundle(prompt_path)
    if task_manifest["provider_models"] != OFFICIAL_MODELS:
        raise ValueError("Five-model task manifest does not match amended roster")
    batches = analysis_manifest["target_batches"]
    profile_batches = task_manifest.get("profile_target_batches") or {}
    profile_by_id = {row["profile_id"]: row for row in profiles}
    expected: dict[str, dict[str, Any]] = {}
    amendment = task_manifest.get("roster_amendment") or {}
    result_experiment_id = amendment.get(
        "source_experiment_id", task_manifest["experiment_id"]
    )
    openai_result_experiment_id = amendment.get(
        "openai_source_experiment_id", result_experiment_id
    )
    for provider, model in OFFICIAL_MODELS.items():
        for unit in task_manifest["task_units"]:
            profile = profile_by_id.get(unit["profile_id"])
            if profile is None:
                raise ValueError("Task unit references an unknown profile")
            target_ids = list(
                profile_batches.get(unit["profile_id"], batches)[unit["batch_id"]]
            )
            selected = batch_profile(profile, target_ids)
            prompt, targets = build_prompt(
                selected,
                unit["condition"],
                unit["evidence_variant"],
                bundle,
            )
            digest = sha256_text(bundle["system_prompt"] + "\n" + prompt)
            call_id = make_call_id(
                provider,
                model,
                profile["profile_id"],
                unit["condition"],
                unit["evidence_variant"],
                bundle["version"],
                unit["batch_id"],
                digest,
            )
            if call_id in expected:
                raise ValueError(f"Duplicate expected call ID: {call_id}")
            expected[call_id] = {
                "provider": provider,
                "model": model,
                "profile": profile,
                "condition": unit["condition"],
                "evidence_variant": unit["evidence_variant"],
                "batch_id": unit["batch_id"],
                "target_ids": [target["item_id"] for target in targets],
                "prompt_sha256": digest,
                "experiment": (
                    openai_result_experiment_id
                    if provider == "openai"
                    else result_experiment_id
                ),
            }
    if len(expected) != task_manifest["expected"]["total_calls"]:
        raise ValueError("Expected call map does not match frozen task count")
    return expected


def normalize_forecasts(
    row: dict[str, Any], expected: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    forecasts = row.get("parsed_forecasts")
    if not isinstance(forecasts, dict) or set(forecasts) != set(
        expected["target_ids"]
    ):
        raise ValueError(f"Forecast coverage mismatch: {row.get('call_id')}")
    normalized: dict[str, list[float]] = {}
    adjusted = 0
    for item_id in expected["target_ids"]:
        values = np.asarray(forecasts[item_id], dtype=float)
        options = row.get("target_option_codes", {}).get(item_id)
        if (
            not isinstance(options, list)
            or values.ndim != 1
            or len(values) != len(options)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ValueError(f"Invalid probability vector: {row.get('call_id')}/{item_id}")
        total = float(values.sum())
        if total <= 0 or abs(total - 1.0) > 0.01 + 1e-12:
            raise ValueError(f"Probability sum exceeds tolerance: {row.get('call_id')}/{item_id}")
        if not np.isclose(total, 1.0, atol=1e-12):
            adjusted += 1
        normalized[item_id] = (values / total).tolist()
    value = dict(row)
    value["parsed_forecasts"] = normalized
    return value, adjusted


def audit_results(
    latest: dict[str, dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_ids = set(expected)
    observed_ids = set(latest)
    missing = expected_ids - observed_ids
    extra = observed_ids - expected_ids
    incomplete = [
        call_id
        for call_id in expected_ids & observed_ids
        if not latest[call_id].get("success") or latest[call_id].get("terminal_failure")
    ]
    if missing or extra or incomplete:
        return [], {
            "complete": False,
            "planned": len(expected_ids),
            "observed_unique": len(observed_ids),
            "missing": len(missing),
            "unexpected": len(extra),
            "incomplete": len(incomplete),
            "analysis_ready": 0,
        }
    rows: list[dict[str, Any]] = []
    normalized_items = 0
    by_provider: Counter[str] = Counter()
    for call_id in sorted(expected_ids):
        row = latest[call_id]
        spec = expected[call_id]
        for field, value in (
            ("provider", spec["provider"]),
            ("model_requested", spec["model"]),
            ("model_returned", spec["model"]),
            ("profile_id", spec["profile"]["profile_id"]),
            ("condition", spec["condition"]),
            ("evidence_variant", spec["evidence_variant"]),
            ("batch_id", spec["batch_id"]),
            ("prompt_sha256", spec["prompt_sha256"]),
        ):
            if row.get(field) != value:
                raise ValueError(f"Frozen result mismatch for {field}: {call_id}")
        if row.get("experiment") != spec["experiment"] or row.get(
            "source_call_id"
        ):
            raise ValueError(f"Unexpected experiment/repeat row: {call_id}")
        if row.get("target_ids") != spec["target_ids"]:
            raise ValueError(f"Target order mismatch: {call_id}")
        profile = spec["profile"]
        if row.get("true_country_code") != profile["true_country_code"]:
            raise ValueError(f"True-country mismatch: {call_id}")
        expected_displayed = (
            profile["true_country_code"]
            if spec["condition"] == "MV"
            else next(
                code
                for code, label in {
                    "CN": "China", "US": "United States", "JO": "Jordan",
                    "FR": "France", "GB": "Great Britain", "IT": "Italy",
                }.items()
                if label == profile["balanced_random_country_label"]
            )
            if spec["condition"] in {"MO", "MR"}
            else None
        )
        if row.get("displayed_country_code") != expected_displayed:
            raise ValueError(f"Metadata-semantics mismatch: {call_id}")
        normalized, changed = normalize_forecasts(row, spec)
        normalized_items += changed
        rows.append(normalized)
        by_provider[spec["provider"]] += 1
    return rows, {
        "complete": True,
        "planned": len(expected_ids),
        "observed_unique": len(latest),
        "missing": 0,
        "unexpected": 0,
        "incomplete": 0,
        "analysis_ready": len(rows),
        "by_provider": dict(sorted(by_provider.items())),
        "locally_normalized_item_vectors": normalized_items,
        "raw_results_unchanged": True,
    }


def pending_output(
    args: argparse.Namespace, audit: dict[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "504-profile confirmatory core",
        "status": "pending_inference",
        "hypothesis_results_generated": False,
        "reason": reason,
        "source_audit": audit,
        "inputs": {
            key: sha256(path) if path.exists() else None
            for key, path in {
                "results": args.results,
                "profiles": args.profiles,
                "analysis_manifest": args.analysis_manifest,
                "task_manifest": args.task_manifest,
                "prompt_bundle": args.prompt_bundle,
            }.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_raw.jsonl",
    )
    parser.add_argument(
        "--profiles", type=Path,
        default=ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl",
    )
    parser.add_argument(
        "--analysis-manifest", type=Path,
        default=ROOT / "dataset/analysis/confirmatory_manifest_v2.json",
    )
    parser.add_argument(
        "--task-manifest", type=Path,
        default=ROOT / "dataset/analysis/confirmatory_core_task_manifest_v2.json",
    )
    parser.add_argument(
        "--gate-report", type=Path,
        default=ROOT / "dataset/analysis/pilot_gate_report.json",
    )
    parser.add_argument(
        "--panel", type=Path,
        default=ROOT / "dataset/derived/country_panel_6.csv",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "dataset/metadata/variable_metadata.json",
    )
    parser.add_argument(
        "--prompt-bundle", type=Path,
        default=ROOT / "api_pilot/prompts/survey-confirmatory-v4.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "api_pilot/outputs/confirmatory_core_v2_strict_inference.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_manifest = json.loads(args.analysis_manifest.read_text(encoding="utf-8"))
    task_manifest = load_task_manifest(args.task_manifest)
    if task_manifest is None:
        raise ValueError("Frozen task manifest is required")
    profiles = load_jsonl(args.profiles)
    if len(profiles) != 504 or sha256(args.profiles) != task_manifest["source"][
        "profile_file_sha256"
    ]:
        raise ValueError("Profile count/hash differs from frozen task manifest")
    for label, path, expected_hash in (
        ("analysis manifest", args.analysis_manifest, task_manifest["source"]["analysis_manifest_sha256"]),
        ("prompt bundle", args.prompt_bundle, task_manifest["prompt"]["sha256"]),
    ):
        if sha256(path) != expected_hash:
            raise ValueError(f"{label} hash differs from frozen task manifest")
    expected = expected_call_map(
        profiles, analysis_manifest, task_manifest, args.prompt_bundle
    )
    latest = latest_rows(args.results)
    rows, audit = audit_results(latest, expected)
    if not audit["complete"]:
        output = pending_output(
            args, audit,
            f"The immutable {task_manifest['expected']['total_calls']:,}-call raw panel is absent, incomplete, or contains extra calls.",
        )
        write_json(args.output, output)
        print(json.dumps({"status": output["status"], **audit}))
        return 0

    high_items = list(analysis_manifest["target_strata"]["high_country_value"])
    direction_rows, utility_rows, row_audit = build_inference_rows(
        rows, profiles, high_items, require_balanced_assignment=True
    )
    resampler, anchors, reference_counts = reference_rows_and_anchors(
        args.panel, args.metadata, high_items
    )
    anchor_error = verify_anchor_match(anchors, args.gate_report, high_items)
    inference = run_strict_confirmatory_inference(
        direction_rows,
        utility_rows,
        anchors,
        anchor_bootstrap=resampler,
        n_bootstrap=args.bootstrap_replicates,
        n_permutations=args.permutations,
        expected_providers=tuple(sorted(OFFICIAL_MODELS)),
        expected_countries=COUNTRIES,
        expected_items=high_items,
    )
    equivalence_margin = float(
        analysis_manifest["thresholds"]["beta_r_equivalence_margin"]
    )
    beta_r = inference.direction_bootstrap.beta_r
    beta_r_equivalence = {
        "margin": equivalence_margin,
        "confidence_interval": [beta_r.ci_low, beta_r.ci_high],
        "rule": "95% bootstrap CI wholly inside [-margin, +margin]",
        "equivalent": (
            beta_r.ci_low > -equivalence_margin
            and beta_r.ci_high < equivalence_margin
        ),
        "failure_interpretation": (
            "failure establishes only non-equivalence; it is not evidence that "
            "beta_R differs from zero"
        ),
    }
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "504-profile confirmatory core",
        "status": "strict_confirmatory_inference_complete",
        "hypothesis_results_generated": True,
        "source_audit": audit,
        "row_audit": row_audit,
        "models": OFFICIAL_MODELS,
        "items": high_items,
        "estimand": {
            "primary": "equal-country, equal-item, equal-model macro",
            "H1": "beta_O > 0 and beta_O - beta_R > 0",
            "H2": "verified-country Brier utility > 0",
            "multiplicity": "Holm across H1 and H2",
            "sesoi_alternatives": {
                "beta_O": 0.05,
                "beta_O_minus_beta_R": 0.05,
                "verified_utility": 0.02,
            },
        },
        "replicates": {
            "profile_reference_bootstrap": args.bootstrap_replicates,
            "balanced_assignment_permutations": args.permutations,
        },
        "reference": {
            "profile_counts_by_country": reference_counts,
            "rebuilt_anchor_max_abs_error_vs_gate_report": anchor_error,
            "anchors_reestimated_in_bootstrap": True,
        },
        "inputs": {
            name: sha256(path)
            for name, path in {
                "results": args.results,
                "profiles": args.profiles,
                "analysis_manifest": args.analysis_manifest,
                "task_manifest": args.task_manifest,
                "gate_report": args.gate_report,
                "panel": args.panel,
                "metadata": args.metadata,
                "prompt_bundle": args.prompt_bundle,
            }.items()
        },
        "inference": inference.to_dict(),
        "secondary_beta_r_equivalence": beta_r_equivalence,
        "interpretation_guardrail": (
            "These are confirmatory results only for the frozen model, country, item "
            "and prompt roster. Country survey labels do not identify cultural essence "
            "or model-training causes."
        ),
    }
    write_json(args.output, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "beta_o": inference.direction_bootstrap.beta_o.estimate,
                "beta_diff": inference.direction_bootstrap.beta_diff.estimate,
                "utility_v": inference.utility_h2.estimate,
                "holm": inference.holm_adjusted_p,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
