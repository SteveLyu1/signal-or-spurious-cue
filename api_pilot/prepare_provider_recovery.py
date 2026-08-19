#!/usr/bin/env python3
"""Seed a provider-specific recovery JSONL from a multi-provider checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    uri = args.checkpoint.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as connection:
        rows = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT result_json FROM results WHERE provider=? ORDER BY call_id",
                (args.provider,),
            )
        ]
    if not rows:
        raise ValueError(f"main checkpoint has no rows for {args.provider}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(
        f"provider={args.provider} seeded={len(rows)} "
        f"successful={sum(bool(row.get('success')) for row in rows)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
