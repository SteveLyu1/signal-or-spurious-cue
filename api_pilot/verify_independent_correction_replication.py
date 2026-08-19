#!/usr/bin/env python3
"""Fail-closed completion contract for independent corrected-wording replication v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, default=ROOT / "dataset/analysis/independent_correction_replication_manifest_v2.json")
    parser.add_argument("--analysis", type=Path, default=ROOT / "api_pilot/outputs/independent_correction_replication_v2_analysis.json")
    parser.add_argument("--output", type=Path, default=ROOT / "api_pilot/outputs/independent_correction_replication_v2_completion.json")
    args = parser.parse_args()
    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    expected = int(manifest["expected"]["total_calls"])
    expected_each = int(manifest["expected"]["calls_per_provider"])
    result_path = ROOT / manifest["immutable_output_path"]
    calls: set[str] = set()
    providers: Counter[str] = Counter()
    malformed = unsuccessful = duplicates = 0
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            call_id = row.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                malformed += 1
            elif call_id in calls:
                duplicates += 1
            else:
                calls.add(call_id)
            if not row.get("success") or row.get("terminal_failure"):
                unsuccessful += 1
            else:
                providers[str(row.get("provider"))] += 1
    result_complete = bool(
        result_path.exists()
        and len(calls) == expected
        and not malformed and not unsuccessful and not duplicates
        and providers == Counter({provider: expected_each for provider in manifest["provider_models"]})
    )
    analysis = json.loads(args.analysis.read_text(encoding="utf-8")) if args.analysis.exists() else {}
    inputs = analysis.get("inputs") or {}
    source_audit = analysis.get("source_audit") or {}
    analysis_complete = bool(
        analysis.get("status") == "complete"
        and analysis.get("analysis_type") == "post-review independent corrected-wording replication"
        and analysis.get("original_confirmatory_decisions_replaced") is False
        and source_audit.get("complete") is True
        and source_audit.get("planned") == expected
        and source_audit.get("observed_unique") == expected
        and source_audit.get("by_provider") == {provider: expected_each for provider in sorted(manifest["provider_models"])}
        and inputs.get("results") == (sha256(result_path) if result_path.exists() else None)
        and inputs.get("task_manifest") == sha256(args.task_manifest)
    )
    complete = result_complete and analysis_complete
    contract = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "pending",
        "complete": complete,
        "expected": expected,
        "observed_unique": len(calls),
        "success_by_provider": dict(sorted(providers.items())),
        "malformed": malformed,
        "unsuccessful": unsuccessful,
        "duplicates": duplicates,
        "result_complete": result_complete,
        "analysis_complete": analysis_complete,
        "task_manifest_sha256": sha256(args.task_manifest),
        "result_sha256": sha256(result_path) if result_path.exists() else None,
        "analysis_sha256": sha256(args.analysis) if args.analysis.exists() else None,
    }
    args.output.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": contract["status"], "observed": len(calls), "expected": expected}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
