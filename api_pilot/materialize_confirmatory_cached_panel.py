#!/usr/bin/env python3
"""Materialize the eight-cell confirmatory logical panel from four raw cells.

This program makes no API calls. A shuffled-evidence logical result is copied
from its reciprocal donor's correct-evidence result only after the two fully
rendered prompts are byte-identical and have the same SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot.run_confirmatory_strict_inference import (
    OFFICIAL_MODELS,
    audit_results,
    batch_profile,
    expected_call_map,
    latest_rows,
    write_json,
)
from api_pilot.run_gate_c_strict_inference import load_jsonl, sha256
from api_pilot.survey_pilot_runner import (
    build_prompt,
    load_prompt_bundle,
    load_task_manifest,
    sha256_text,
)


RAW_CELLS = ("M0", "MO", "MR", "MV")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _logical_call_id(source_call_id: str, target_profile_id: str) -> str:
    digest = hashlib.sha256(
        f"{source_call_id}\x1f{target_profile_id}\x1fshuffled".encode("utf-8")
    ).hexdigest()[:20]
    return f"derived-cache/{target_profile_id}/{digest}"


def verify_cache_prompt_identity(
    target: dict[str, Any],
    donor: dict[str, Any],
    condition: str,
    target_ids: list[str],
    bundle: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Fail unless target-shuffled and donor-correct prompts are identical."""
    target_batch = batch_profile(target, target_ids)
    donor_batch = batch_profile(donor, target_ids)
    target_prompt, target_specs = build_prompt(
        target_batch, condition, "shuffled", bundle
    )
    donor_prompt, donor_specs = build_prompt(
        donor_batch, condition, "correct", bundle
    )
    target_bytes = (bundle["system_prompt"] + "\n" + target_prompt).encode("utf-8")
    donor_bytes = (bundle["system_prompt"] + "\n" + donor_prompt).encode("utf-8")
    if target_bytes != donor_bytes:
        raise ValueError(
            "Rendered cache prompts differ: "
            f"{target['profile_id']} <- {donor['profile_id']}/{condition}"
        )
    if [row["item_id"] for row in target_specs] != [
        row["item_id"] for row in donor_specs
    ]:
        raise ValueError("Cached target specifications differ")
    return sha256_text(bundle["system_prompt"] + "\n" + target_prompt), target_specs


def materialize_logical_rows(
    raw_rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    analysis_manifest: dict[str, Any],
    task_manifest: dict[str, Any],
    prompt_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return raw correct rows plus four fail-closed cached shuffled cells."""
    if tuple(tuple(cell) for cell in task_manifest.get("cells", [])) != tuple(
        (condition, "correct") for condition in RAW_CELLS
    ):
        raise ValueError("Cached core v2 must contain exactly four raw correct cells")
    expected_raw = int(task_manifest.get("expected", {}).get("total_calls") or 0)
    if expected_raw != 8_064 * len(OFFICIAL_MODELS):
        raise ValueError("Cached core raw count does not match five-model roster")
    profile_by_id = {row["profile_id"]: row for row in profiles}
    if len(profile_by_id) != 504:
        raise ValueError("Cached materializer requires 504 unique profiles")
    for profile in profiles:
        donor = profile_by_id.get(profile.get("donor_profile_id"))
        if (
            donor is None
            or donor.get("donor_profile_id") != profile["profile_id"]
            or donor.get("pair_id") != profile.get("pair_id")
            or donor.get("evidence") != profile.get("shuffled_evidence")
            or donor.get("true_country_code") != profile.get("true_country_code")
            or donor.get("balanced_random_country_label")
            != profile.get("balanced_random_country_label")
        ):
            raise ValueError(f"Invalid reciprocal donor for {profile['profile_id']}")

    raw_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (
            row["provider"], row["profile_id"], row["condition"], row["batch_id"]
        )
        if key in raw_index:
            raise ValueError(f"Duplicate raw logical source: {key}")
        if row.get("evidence_variant") != "correct" or row.get("source_call_id"):
            raise ValueError(f"Non-raw row in cached source panel: {row.get('call_id')}")
        raw_index[key] = row

    bundle = load_prompt_bundle(prompt_path)
    batches = analysis_manifest["target_batches"]
    profile_batches = task_manifest.get("profile_target_batches") or {}
    logical: list[dict[str, Any]] = []
    source_use: dict[str, str] = {}
    verified = 0
    for raw in sorted(raw_rows, key=lambda value: value["call_id"]):
        copied = dict(raw)
        copied["derived_cached"] = False
        copied["api_call_made"] = True
        logical.append(copied)

    for provider in sorted(OFFICIAL_MODELS):
        for target in sorted(profiles, key=lambda value: value["profile_id"]):
            donor = profile_by_id[target["donor_profile_id"]]
            for condition in RAW_CELLS:
                target_batches = profile_batches.get(target["profile_id"], batches)
                donor_batches = profile_batches.get(donor["profile_id"], batches)
                if target_batches != donor_batches:
                    raise ValueError(
                        f"Reciprocal pair batch mismatch: {target['profile_id']}"
                    )
                for batch_id, target_ids in target_batches.items():
                    source = raw_index.get(
                        (provider, donor["profile_id"], condition, batch_id)
                    )
                    if source is None:
                        raise ValueError(
                            f"Missing donor source {provider}/{donor['profile_id']}/"
                            f"{condition}/{batch_id}"
                        )
                    digest, target_specs = verify_cache_prompt_identity(
                        target, donor, condition, list(target_ids), bundle
                    )
                    if (
                        digest != source.get("prompt_sha256")
                        or source.get("target_ids")
                        != [row["item_id"] for row in target_specs]
                    ):
                        raise ValueError(
                            f"Cached prompt/source digest mismatch: {source['call_id']}"
                        )
                    if source["call_id"] in source_use:
                        raise ValueError(
                            f"Raw call reused more than once: {source['call_id']}"
                        )
                    source_use[source["call_id"]] = target["profile_id"]
                    derived = dict(source)
                    derived.update(
                        {
                            "call_id": _logical_call_id(
                                source["call_id"], target["profile_id"]
                            ),
                            "profile_id": target["profile_id"],
                            "true_country_code": target["true_country_code"],
                            "evidence_variant": "shuffled",
                            "experiment": "confirmatory_core_v2_logical",
                            "source_call_id": source["call_id"],
                            "derived_cached": True,
                            "api_call_made": False,
                            "pair_id": target["pair_id"],
                            "donor_profile_id": donor["profile_id"],
                            "rendered_prompt_byte_identical": True,
                            "rendered_prompt_sha256_verified": digest,
                            "source_latency_ms": source.get("latency_ms"),
                            "latency_ms": 0,
                            "source_input_tokens": source.get("input_tokens"),
                            "source_output_tokens": source.get("output_tokens"),
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }
                    )
                    logical.append(derived)
                    verified += 1
    expected_derived = expected_raw
    if verified != expected_derived or len(source_use) != expected_derived:
        raise ValueError("Cached source use is not a one-to-one reciprocal mapping")
    expected_logical = expected_raw * 2
    if len(logical) != expected_logical or len({row["call_id"] for row in logical}) != expected_logical:
        raise ValueError("Eight-cell logical panel is incomplete or duplicated")
    return sorted(logical, key=lambda value: value["call_id"]), {
        "complete": True,
        "raw_api_rows": len(raw_rows),
        "derived_cached_rows": verified,
        "logical_rows": len(logical),
        "reciprocal_pairs": 252,
        "rendered_prompt_byte_identity_checks": verified,
        "rendered_prompt_sha256_checks": verified,
        "raw_source_reuse_exactly_once": True,
        "api_requests_made_by_materializer": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-results", type=Path,
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
        "--prompt-bundle", type=Path,
        default=ROOT / "api_pilot/prompts/survey-confirmatory-v4.json",
    )
    parser.add_argument(
        "--donor-assignment", type=Path,
        default=ROOT / "dataset/analysis/confirmatory_donor_pair_assignment_v2.json",
    )
    parser.add_argument(
        "--logical-output", type=Path,
        default=ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_logical.jsonl",
    )
    parser.add_argument(
        "--audit-output", type=Path,
        default=ROOT / "api_pilot/outputs/confirmatory_core_v2_cache_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "raw_results", "profiles", "analysis_manifest", "task_manifest",
        "prompt_bundle", "donor_assignment", "logical_output", "audit_output",
    ):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, (ROOT / path).resolve())
    analysis = json.loads(args.analysis_manifest.read_text(encoding="utf-8"))
    manifest = load_task_manifest(args.task_manifest)
    if manifest is None:
        raise ValueError("Frozen cached-core task manifest is required")
    profiles = load_jsonl(args.profiles)
    for label, path, expected_hash in (
        ("profile", args.profiles, manifest["source"]["profile_file_sha256"]),
        (
            "analysis manifest", args.analysis_manifest,
            manifest["source"]["analysis_manifest_sha256"],
        ),
        ("prompt bundle", args.prompt_bundle, manifest["prompt"]["sha256"]),
        (
            "donor assignment", args.donor_assignment,
            manifest["source"]["donor_assignment_sha256"],
        ),
    ):
        if not path.exists() or sha256(path) != expected_hash:
            raise ValueError(f"{label} hash differs from frozen task manifest")
    expected = expected_call_map(
        profiles, analysis, manifest, args.prompt_bundle
    )
    raw, source_audit = audit_results(latest_rows(args.raw_results), expected)
    if not source_audit["complete"]:
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_inference",
            "logical_panel_generated": False,
            "reason": f"The immutable {manifest['expected']['total_calls']:,}-call raw panel is incomplete.",
            "source_audit": source_audit,
        }
        write_json(args.audit_output, output)
        print(json.dumps({"status": output["status"], **source_audit}))
        return 0
    logical, cache_audit = materialize_logical_rows(
        raw, profiles, analysis, manifest, args.prompt_bundle
    )
    atomic_write_jsonl(args.logical_output, logical)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "cached_logical_panel_complete",
        "logical_panel_generated": True,
        "source_audit": source_audit,
        "cache_audit": cache_audit,
        "inputs": {
            "raw_results_sha256": sha256(args.raw_results),
            "task_manifest_sha256": sha256(args.task_manifest),
            "profiles_sha256": sha256(args.profiles),
            "prompt_bundle_sha256": sha256(args.prompt_bundle),
        },
        "logical_output": {
            "path": args.logical_output.relative_to(ROOT).as_posix(),
            "sha256": sha256(args.logical_output),
        },
    }
    write_json(args.audit_output, output)
    print(json.dumps({"status": output["status"], **cache_audit}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
