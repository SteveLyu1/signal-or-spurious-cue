from __future__ import annotations

import csv
import json
import sys

import pytest

from api_pilot import finalize_human_quality_review as finalize


FIELDS = [
    "review_unit", "role", "question", "scale", "question_scale_match",
    "country_referential", "ambiguous_or_mistranslated", "acceptable", "notes",
]
CORRECTION_FIELDS = [
    "review_unit", "family", "question", "scale", "question_scale_match",
    "ambiguous_or_mistranslated", "acceptable", "notes",
]


def write_review(path, **decisions):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "review_unit": "Q1", "role": "target", "question": "Question?",
        "scale": "0/1", "question_scale_match": "yes",
        "country_referential": "no", "ambiguous_or_mistranslated": "no",
        "acceptable": "yes", "notes": "",
    }
    row.update(decisions)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


def write_correction_review(path, **decisions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORRECTION_FIELDS)
        writer.writeheader()
        for review_unit in sorted(finalize.CORRECTED_UNITS):
            row = {
                "review_unit": review_unit,
                "family": "checklist",
                "question": "Corrected question?",
                "scale": "0 = No; 1 = Yes.",
                "question_scale_match": "yes",
                "ambiguous_or_mistranslated": "no",
                "acceptable": "yes",
                "notes": "",
            }
            row.update(decisions)
            writer.writerow(row)


def test_complete_review_creates_hash_bound_approval(monkeypatch, tmp_path):
    review = tmp_path / "review.csv"
    packet = tmp_path / "packet.md"
    signoff = tmp_path / "signoff.json"
    write_review(review)
    correction = tmp_path / "correction.csv"
    write_correction_review(correction)
    packet.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(finalize, "REVIEW_CSV", review)
    monkeypatch.setattr(finalize, "EXPECTED_ORIGINAL_UNITS", {"Q1"})
    monkeypatch.setattr(finalize, "CORRECTION_REVIEW_CSV", correction)
    monkeypatch.setattr(finalize, "PACKET", packet)
    monkeypatch.setattr(finalize, "SIGNOFF", signoff)
    monkeypatch.setattr(sys, "argv", ["finalize", "--reviewer", "Reviewer A"])
    assert finalize.main() == 0
    result = json.loads(signoff.read_text(encoding="utf-8"))
    assert result["status"] == "approved"
    assert result["reviewed_rows"] == 1
    assert result["correction_reviewed_rows"] == 8
    assert result["unresolved_rejections"] == []


def test_blank_decision_is_rejected(monkeypatch, tmp_path):
    review = tmp_path / "review.csv"
    packet = tmp_path / "packet.md"
    write_review(review, acceptable="")
    correction = tmp_path / "correction.csv"
    write_correction_review(correction)
    packet.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(finalize, "REVIEW_CSV", review)
    monkeypatch.setattr(finalize, "EXPECTED_ORIGINAL_UNITS", {"Q1"})
    monkeypatch.setattr(finalize, "CORRECTION_REVIEW_CSV", correction)
    monkeypatch.setattr(finalize, "PACKET", packet)
    monkeypatch.setattr(finalize, "SIGNOFF", tmp_path / "signoff.json")
    monkeypatch.setattr(sys, "argv", ["finalize", "--reviewer", "Reviewer A"])
    with pytest.raises(ValueError, match="yes or no"):
        finalize.main()


def test_corrected_original_rejection_is_resolved_by_correction_review(
    monkeypatch, tmp_path
):
    review = tmp_path / "review.csv"
    correction = tmp_path / "correction.csv"
    packet = tmp_path / "packet.md"
    signoff = tmp_path / "signoff.json"
    write_review(
        review,
        review_unit="A029",
        question_scale_match="no",
        ambiguous_or_mistranslated="yes",
        acceptable="no",
    )
    write_correction_review(correction)
    packet.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(finalize, "REVIEW_CSV", review)
    monkeypatch.setattr(finalize, "EXPECTED_ORIGINAL_UNITS", {"A029"})
    monkeypatch.setattr(finalize, "CORRECTION_REVIEW_CSV", correction)
    monkeypatch.setattr(finalize, "PACKET", packet)
    monkeypatch.setattr(finalize, "SIGNOFF", signoff)
    monkeypatch.setattr(sys, "argv", ["finalize", "--reviewer", "Reviewer A"])
    assert finalize.main() == 0
    result = json.loads(signoff.read_text(encoding="utf-8"))
    assert result["status"] == "approved"
    assert "A029" in result["superseded_original_rejections"]


def test_missing_original_review_unit_is_rejected(monkeypatch, tmp_path):
    review = tmp_path / "review.csv"
    correction = tmp_path / "correction.csv"
    packet = tmp_path / "packet.md"
    write_review(review)
    write_correction_review(correction)
    packet.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(finalize, "REVIEW_CSV", review)
    monkeypatch.setattr(finalize, "CORRECTION_REVIEW_CSV", correction)
    monkeypatch.setattr(finalize, "PACKET", packet)
    monkeypatch.setattr(finalize, "EXPECTED_ORIGINAL_UNITS", {"Q1", "Q2"})
    monkeypatch.setattr(sys, "argv", ["finalize", "--reviewer", "Reviewer A"])
    with pytest.raises(ValueError, match="roster differs"):
        finalize.main()
