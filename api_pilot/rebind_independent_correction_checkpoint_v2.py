#!/usr/bin/env python3
"""One-time audited rebind after a serialization-only v2 manifest rewrite."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "api_pilot/restricted_outputs/independent_correction_replication_v2.run.json"
BACKUP = ROOT / "api_pilot/restricted_outputs/independent_correction_replication_v2.run.pre-serialization-rebind.json"
CHECKPOINT = ROOT / "api_pilot/restricted_outputs/independent_correction_replication_v2.checkpoint.sqlite3"
CHECKPOINT_BACKUP = ROOT / "api_pilot/restricted_outputs/independent_correction_replication_v2.checkpoint.pre-serialization-rebind.sqlite3"
TASK = ROOT / "dataset/analysis/independent_correction_replication_manifest_v2.json"
ANALYSIS = ROOT / "dataset/analysis/independent_correction_analysis_manifest_v2.json"
PROFILES = ROOT / "api_pilot/restricted_outputs/independent_correction_profiles_6x12_v2.jsonl"
PROMPT = ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json"
PROVENANCE = ROOT / "dataset/analysis/independent_correction_provenance_amendment_v1.json"
AUDIT = ROOT / "dataset/analysis/independent_correction_checkpoint_rebind_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    previous_path = BACKUP if BACKUP.exists() else RUN
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    task = json.loads(TASK.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if previous["task_manifest_sha256"] != provenance["serialization_history"]["superseded_task_manifest_sha256"]:
        raise ValueError("Run metadata is not bound to the recorded superseded manifest")
    if previous["profile_file_sha256"] != sha256(PROFILES):
        raise ValueError("Profile file changed; serialization-only rebind forbidden")
    if previous["prompt_bundle_sha256"] != sha256(PROMPT):
        raise ValueError("Prompt changed; serialization-only rebind forbidden")
    if previous["profile_count"] != 72 or previous["planned_calls"] != 2880:
        raise ValueError("Previous 72-profile/2,880-call contract is not intact")
    if len(task["profile_ids"]) != 72 or len(task["task_units"]) != 576:
        raise ValueError("Current 72-profile/576-unit contract is not intact")
    if task["expected"] != {"providers": 5, "calls_per_provider": 576, "total_calls": 2880}:
        raise ValueError("Current call-count contract changed")
    if previous["models"] != task["provider_models"]:
        raise ValueError("Provider/model roster changed")
    if previous["cells"] != task["cells"]:
        raise ValueError("Experimental cells changed")

    with sqlite3.connect(f"file:{CHECKPOINT.as_posix()}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT call_id FROM results").fetchall()
        stored_row = connection.execute(
            "SELECT value FROM run_meta WHERE key='run_metadata'"
        ).fetchone()
    if stored_row is None:
        raise ValueError("Checkpoint has no frozen run metadata")
    stored_metadata = json.loads(stored_row[0])
    if stored_metadata.get("task_manifest_sha256") not in {
        previous["task_manifest_sha256"], sha256(TASK)
    }:
        raise ValueError("Checkpoint task-manifest binding is neither old nor current")
    if stored_metadata.get("profile_file_sha256") != previous["profile_file_sha256"]:
        raise ValueError("Checkpoint profile binding differs from the pre-rebind run")
    checkpoint_ids = {row[0] for row in rows}
    valid_profiles = set(task["profile_ids"])
    valid_batches = set(task["target_batches"])
    valid_cells = {tuple(cell) for cell in task["cells"]}
    def valid_call_id(call_id: str) -> bool:
        parts = call_id.split("/")
        if len(parts) != 8:
            return False
        provider, model, prompt_version, profile, batch, condition, evidence, digest = parts
        return bool(
            task["provider_models"].get(provider) == model
            and prompt_version == task["prompt"]["version"]
            and profile in valid_profiles
            and batch in valid_batches
            and (condition, evidence) in valid_cells
            and re.fullmatch(r"[0-9a-f]{12}", digest)
        )
    if not checkpoint_ids or not all(valid_call_id(call_id) for call_id in checkpoint_ids):
        raise ValueError("Checkpoint contains a call outside the frozen experiment/provider namespace")

    if not BACKUP.exists():
        shutil.copy2(RUN, BACKUP)
    elif sha256(BACKUP) != sha256(RUN):
        backup_payload = json.loads(BACKUP.read_text(encoding="utf-8"))
        if backup_payload.get("task_manifest_sha256") != previous["task_manifest_sha256"]:
            raise ValueError("Existing pre-rebind backup is inconsistent")

    if not CHECKPOINT_BACKUP.exists():
        shutil.copy2(CHECKPOINT, CHECKPOINT_BACKUP)

    updated = dict(previous)
    updated["task_manifest_sha256"] = sha256(TASK)
    updated["analysis_manifest_sha256"] = sha256(ANALYSIS)
    RUN.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stored_updated = dict(stored_metadata)
    stored_updated["task_manifest_sha256"] = updated["task_manifest_sha256"]
    stored_updated["analysis_manifest_sha256"] = updated["analysis_manifest_sha256"]
    with sqlite3.connect(CHECKPOINT) as connection:
        connection.execute(
            "UPDATE run_meta SET value=? WHERE key='run_metadata'",
            (json.dumps(stored_updated, ensure_ascii=False, sort_keys=True),),
        )
        connection.commit()
    audit = {
        "status": "authorized_serialization_only_rebind_pending_runner_signature_migration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "A dry-run rewrote generated timestamps after calls began; scientific task units remained unchanged.",
        "old_run_metadata_sha256": sha256(BACKUP),
        "checkpoint_backup_sha256": sha256(CHECKPOINT_BACKUP),
        "old_task_manifest_sha256": previous["task_manifest_sha256"],
        "new_task_manifest_sha256": updated["task_manifest_sha256"],
        "old_analysis_manifest_sha256": previous["analysis_manifest_sha256"],
        "new_analysis_manifest_sha256": updated["analysis_manifest_sha256"],
        "profile_sha256": sha256(PROFILES),
        "prompt_sha256": sha256(PROMPT),
        "checkpoint_rows_verified": len(checkpoint_ids),
        "scientific_task_set_unchanged": True,
        "changed_run_metadata_fields": ["task_manifest_sha256", "analysis_manifest_sha256"],
        "checkpoint_result_rows_changed": 0,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rebound", "checkpoint_rows": len(checkpoint_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
