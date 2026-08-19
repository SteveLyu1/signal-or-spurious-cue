#!/usr/bin/env python3
"""Create an auditable overlay for small probability-sum validation failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalized_forecasts(
    row: dict[str, Any], maximum_sum_error: float
) -> tuple[dict[str, list[float]], dict[str, float]]:
    raw = row.get("raw_response_text")
    if not raw:
        raise ValueError(f"{row['call_id']}: missing raw response")
    try:
        forecasts = json.loads(raw)["forecasts"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"{row['call_id']}: malformed raw response") from error
    expected = row.get("target_option_codes") or {}
    if set(forecasts) != set(expected):
        raise ValueError(f"{row['call_id']}: forecast keys do not match targets")
    repaired = {}
    original_sums = {}
    for item_id, option_codes in expected.items():
        values = forecasts[item_id]
        if len(values) != len(option_codes):
            raise ValueError(f"{row['call_id']}: {item_id} option count mismatch")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in values
        ):
            raise ValueError(f"{row['call_id']}: {item_id} has invalid probability")
        total = math.fsum(float(value) for value in values)
        original_sums[item_id] = total
        if total <= 0 or abs(total - 1.0) > maximum_sum_error + 1e-9:
            raise ValueError(
                f"{row['call_id']}: {item_id} sum {total:.8f} exceeds repair bound"
            )
        repaired[item_id] = [float(value) / total for value in values]
    return repaired, original_sums


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="mistral")
    parser.add_argument("--maximum-sum-error", type=float, default=0.05)
    args = parser.parse_args()

    source_uri = args.source.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(source_uri, uri=True, timeout=10) as source:
        source_metadata_row = source.execute(
            "SELECT value FROM run_meta WHERE key='run_metadata'"
        ).fetchone()
        if source_metadata_row is None:
            raise ValueError("source checkpoint has no run metadata")
        source_metadata = json.loads(source_metadata_row[0])
        rows = [
            json.loads(row[0])
            for row in source.execute(
                "SELECT result_json FROM results WHERE provider=? AND success=0",
                (args.provider,),
            )
        ]

    candidates = [
        row
        for row in rows
        if row.get("terminal_failure")
        and row.get("status_code") == 200
        and row.get("error_type") == "validation_error"
    ]
    if not candidates:
        raise ValueError("no eligible terminal validation failures")

    repaired_rows = []
    for row in candidates:
        forecasts, original_sums = normalized_forecasts(
            row, args.maximum_sum_error
        )
        repaired = dict(row)
        repaired.update(
            {
                "success": True,
                "terminal_failure": False,
                "parsed_forecasts": forecasts,
                "probability_sum_max_error": max(
                    abs(math.fsum(values) - 1.0) for values in forecasts.values()
                ),
                "error_type": None,
                "error_message": None,
                "postprocess_repair": {
                    "method": "divide_each_probability_by_item_sum",
                    "maximum_allowed_sum_error": args.maximum_sum_error,
                    "original_sums": original_sums,
                    "source_checkpoint": str(args.source),
                    "source_response_sha256": row.get("response_sha256"),
                    "preserves_original_checkpoint": True,
                },
            }
        )
        repaired_rows.append(repaired)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.exists():
        args.checkpoint.unlink()
    with sqlite3.connect(args.checkpoint) as target:
        target.execute("PRAGMA journal_mode=WAL")
        target.execute("PRAGMA synchronous=FULL")
        target.execute(
            "CREATE TABLE run_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        target.execute(
            "CREATE TABLE results (call_id TEXT PRIMARY KEY, provider TEXT NOT NULL, "
            "success INTEGER NOT NULL, result_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        target.execute(
            "CREATE TABLE result_history (history_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "call_id TEXT NOT NULL, provider TEXT NOT NULL, result_json TEXT NOT NULL, "
            "recorded_at TEXT NOT NULL)"
        )
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": {args.provider: source_metadata["models"][args.provider]},
            "repair_only": True,
            "source_checkpoint": str(args.source),
            "maximum_sum_error": args.maximum_sum_error,
            "repaired_calls": len(repaired_rows),
        }
        signature = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()
        target.execute(
            "INSERT INTO run_meta(key,value) VALUES('run_signature',?)", (signature,)
        )
        target.execute(
            "INSERT INTO run_meta(key,value) VALUES('run_metadata',?)",
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True),),
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        for row in repaired_rows:
            serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            target.execute(
                "INSERT INTO results VALUES(?,?,?,?,?)",
                (row["call_id"], row["provider"], 1, serialized, timestamp),
            )
            target.execute(
                "INSERT INTO result_history(call_id,provider,result_json,recorded_at) "
                "VALUES(?,?,?,?)",
                (row["call_id"], row["provider"], serialized, timestamp),
            )
        target.commit()

    atomic_write(
        args.output,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repaired_rows),
    )
    print(
        f"provider={args.provider} repaired={len(repaired_rows)} "
        f"bound={args.maximum_sum_error} checkpoint={args.checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
