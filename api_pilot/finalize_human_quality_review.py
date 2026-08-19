#!/usr/bin/env python3
"""Validate a completed blind review sheet and create its bound sign-off."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_CSV = ROOT / "api_pilot/human_review/blind_quality_review_template.csv"
CORRECTION_REVIEW_CSV = ROOT / "api_pilot/human_review/checklist_correction_review.csv"
PACKET = ROOT / "api_pilot/human_review/BLIND_QUALITY_REVIEW.md"
SIGNOFF = ROOT / "api_pilot/human_review/human_quality_review_signoff.json"
BOOLEAN = {"yes": True, "no": False}
CORRECTED_UNITS = {
    "A029", "A030", "A039", "A042",
    "A124_03", "A124_06", "A124_08", "A124_09",
}
EXPECTED_ORIGINAL_UNITS = {
    "A006", "A165", "B008", "C001", "C039", "D059", "E023", "E035",
    "E037", "F063", "A124_09", "E025", "A005", "A039", "E039",
    "A124_08", "A124_03", "A124_06", "D060", "D078", "A030", "E027",
    "E018", "F121", "E036", "A029", "A042", "D061", "F118", "A004",
    "prompt_M0", "prompt_MO", "prompt_MR", "prompt_MV",
}


def normalized(value: str | None) -> str:
    return str(value or "").strip().lower()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_decisions(
    rows: list[dict[str, str]], required: tuple[str, ...], source: str
) -> list[dict[str, object]]:
    invalid = []
    for row in rows:
        values = {field: normalized(row.get(field)) for field in required}
        unknown = {field: value for field, value in values.items() if value not in BOOLEAN}
        if unknown:
            invalid.append({
                "source": source,
                "review_unit": row.get("review_unit") or "unknown",
                "invalid_fields": unknown,
            })
    return invalid


def validate_roster(
    rows: list[dict[str, str]], expected: set[str], source: str
) -> None:
    units = [str(row.get("review_unit") or "").strip() for row in rows]
    duplicates = sorted({unit for unit in units if units.count(unit) > 1})
    observed = set(units)
    if duplicates or observed != expected or len(units) != len(expected):
        raise ValueError(
            f"{source} review roster differs from the frozen set: "
            + json.dumps({
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
                "duplicates": duplicates,
            }, ensure_ascii=False)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    reviewer = args.reviewer.strip()
    if not reviewer:
        raise ValueError("Reviewer name must not be blank")

    rows = read_rows(REVIEW_CSV)
    if not rows:
        raise ValueError("Blind review CSV is empty")
    correction_rows = read_rows(CORRECTION_REVIEW_CSV)
    if not correction_rows:
        raise ValueError("Checklist correction review CSV is empty")

    required = (
        "question_scale_match", "country_referential",
        "ambiguous_or_mistranslated", "acceptable",
    )
    validate_roster(rows, EXPECTED_ORIGINAL_UNITS, "Original")
    validate_roster(correction_rows, CORRECTED_UNITS, "Correction")
    correction_required = (
        "question_scale_match", "ambiguous_or_mistranslated", "acceptable",
    )
    invalid = validate_decisions(rows, required, "original")
    invalid.extend(validate_decisions(correction_rows, correction_required, "correction"))
    if invalid:
        raise ValueError(
            "Every decision cell must be yes or no: "
            + json.dumps(invalid, ensure_ascii=False)
        )

    unresolved = []
    for row in rows:
        review_unit = row.get("review_unit") or "unknown"
        values = {field: normalized(row.get(field)) for field in required}
        if review_unit in CORRECTED_UNITS:
            continue
        acceptable = BOOLEAN[values["acceptable"]]
        scale_matches = BOOLEAN[values["question_scale_match"]]
        ambiguous = BOOLEAN[values["ambiguous_or_mistranslated"]]
        if not acceptable or not scale_matches or ambiguous:
            unresolved.append({
                "review_unit": review_unit,
                "question_scale_match": scale_matches,
                "country_referential": BOOLEAN[values["country_referential"]],
                "ambiguous_or_mistranslated": ambiguous,
                "acceptable": acceptable,
                "notes": str(row.get("notes") or "").strip(),
            })

    for row in correction_rows:
        review_unit = row.get("review_unit") or "unknown"
        values = {field: normalized(row.get(field)) for field in correction_required}
        acceptable = BOOLEAN[values["acceptable"]]
        scale_matches = BOOLEAN[values["question_scale_match"]]
        ambiguous = BOOLEAN[values["ambiguous_or_mistranslated"]]
        if not acceptable or not scale_matches or ambiguous:
            unresolved.append({
                "review_unit": review_unit,
                "source": "corrected wording",
                "question_scale_match": scale_matches,
                "ambiguous_or_mistranslated": ambiguous,
                "acceptable": acceptable,
                "notes": str(row.get("notes") or "").strip(),
            })

    packet_hash = hashlib.sha256(PACKET.read_bytes()).hexdigest()
    result = {
        "status": "approved" if not unresolved else "revisions_required",
        "reviewer": reviewer,
        "reviewed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "review_packet_sha256": packet_hash,
        "original_review_csv_sha256": hashlib.sha256(REVIEW_CSV.read_bytes()).hexdigest(),
        "correction_review_csv_sha256": hashlib.sha256(
            CORRECTION_REVIEW_CSV.read_bytes()
        ).hexdigest(),
        "reviewed_rows": len(rows),
        "correction_reviewed_rows": len(correction_rows),
        "superseded_original_rejections": sorted(CORRECTED_UNITS),
        "unresolved_rejections": unresolved,
        "notes": (
            "Blind review and corrected-wording review completed."
            if not unresolved
            else "Resolve or exclude remaining rejected review units before approval."
        ),
    }
    SIGNOFF.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
