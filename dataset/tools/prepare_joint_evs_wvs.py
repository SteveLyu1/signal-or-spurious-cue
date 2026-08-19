#!/usr/bin/env python3
"""Validate Joint EVS/WVS v5.0 CSV and build machine-readable metadata."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


EXPECTED_COLUMNS = 231
EXPECTED_ROWS = 156_658
EXPECTED_VERSION = "5-0-0 (2024-06-24)"
EXPECTED_DOI_GESIS = "doi:10.4232/1.14320"
EXPECTED_STUDIES = {"EVS5": 59_438, "WVS7": 97_220}
EXPECTED_COUNTRIES = 92

MISSING_CODES = {
    "-5": "Missing: Other",
    "-4": "Not asked in survey",
    "-3": "Not applicable",
    "-2": "No answer",
    "-1": "Don't know",
}

HEADER_FOOTER_PATTERNS = (
    re.compile(r"^Joint EVS/WVS 2017-2022 Dataset$"),
    re.compile(r"^Version 5\.0\.0 .*"),
    re.compile(r"^page \d+$", re.IGNORECASE),
)

VALUE_LINE = re.compile(
    r"^(?P<code>-?\d+(?:\.\d+)?|[A-Z][A-Z0-9_.:-]{0,23})\s+(?P<label>\S.*)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_pdf_lines(section: str) -> list[str]:
    lines: list[str] = []
    for raw in section.splitlines():
        line = " ".join(raw.replace("\u00a0", " ").split())
        if not line:
            continue
        if any(pattern.match(line) for pattern in HEADER_FOOTER_PATTERNS):
            continue
        lines.append(line)
    return lines


def parse_codebook(
    codebook: Path,
    variables: list[str],
    observed_values: dict[str, set[str]],
) -> dict[str, dict]:
    reader = PdfReader(str(codebook))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    starts: dict[str, tuple[int, str]] = {}
    for variable in variables:
        match = re.search(
            rf"(?mi)^[\-‐–—]?{re.escape(variable)}\s+-\s+(.+)$",
            text,
        )
        if not match:
            raise ValueError(f"Variable {variable!r} not found in codebook")
        starts[variable] = (match.start(), match.group(1).strip())

    ordered = sorted((start, variable) for variable, (start, _) in starts.items())
    end_by_variable = {
        variable: ordered[index + 1][0] if index + 1 < len(ordered) else len(text)
        for index, (_, variable) in enumerate(ordered)
    }

    metadata: dict[str, dict] = {}
    for variable in variables:
        start, label = starts[variable]
        lines = clean_pdf_lines(text[start : end_by_variable[variable]])
        if lines and lines[0].startswith(f"{variable} - "):
            lines = lines[1:]

        value_labels: dict[str, str] = {}
        first_value_index = len(lines)
        for index, line in enumerate(lines):
            match = VALUE_LINE.match(line)
            if not match:
                continue
            code = match.group("code")
            value_label = match.group("label").strip()
            if code not in observed_values[variable]:
                continue
            if code.lower() in {
                "data",
                "evs5",
                "wvs7",
                "master",
                "note",
                "version",
            }:
                continue
            value_labels.setdefault(code, value_label)
            first_value_index = min(first_value_index, index)

        description_lines = []
        for line in lines[:first_value_index]:
            if line in {"WEIGHTS", "Missing values:", "M49 standard:", "ISO 3166-1:"}:
                continue
            description_lines.append(line)

        metadata[variable] = {
            "variable": variable,
            "label": label,
            "description": " ".join(description_lines).strip(),
            "value_labels": value_labels,
        }

    return metadata


def validate_csv(
    csv_path: Path,
) -> tuple[dict, list[str], dict[tuple[str, str], dict], dict[str, set[str]]]:
    versions: collections.Counter[str] = collections.Counter()
    dois: collections.Counter[str] = collections.Counter()
    studies: collections.Counter[str] = collections.Counter()
    waves: collections.Counter[str] = collections.Counter()
    country_stats: dict[tuple[str, str], dict] = {}
    ids: set[str] = set()
    duplicate_ids = 0
    bad_width_rows = 0
    rows = 0
    observed_values: dict[str, set[str]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        observed_values = {name: set() for name in header}
        indexes = {
            name: header.index(name)
            for name in (
                "version",
                "doi_gesis",
                "studytit",
                "wave",
                "uniqid",
                "cntry",
                "cntry_AN",
                "year",
            )
        }
        for row in reader:
            rows += 1
            if len(row) != len(header):
                bad_width_rows += 1
                continue
            for name, value in zip(header, row):
                observed_values[name].add(value)
            versions[row[indexes["version"]]] += 1
            dois[row[indexes["doi_gesis"]]] += 1
            studies[row[indexes["studytit"]]] += 1
            waves[row[indexes["wave"]]] += 1
            unique_id = row[indexes["uniqid"]]
            if unique_id in ids:
                duplicate_ids += 1
            ids.add(unique_id)

            key = (row[indexes["cntry"]], row[indexes["cntry_AN"]])
            stats = country_stats.setdefault(
                key,
                {"n": 0, "studies": collections.Counter(), "years": set()},
            )
            stats["n"] += 1
            stats["studies"][row[indexes["studytit"]]] += 1
            stats["years"].add(row[indexes["year"]])

    checks = {
        "row_count": {"actual": rows, "expected": EXPECTED_ROWS, "pass": rows == EXPECTED_ROWS},
        "column_count": {
            "actual": len(header),
            "expected": EXPECTED_COLUMNS,
            "pass": len(header) == EXPECTED_COLUMNS,
        },
        "bad_width_rows": {"actual": bad_width_rows, "expected": 0, "pass": bad_width_rows == 0},
        "duplicate_uniqid": {"actual": duplicate_ids, "expected": 0, "pass": duplicate_ids == 0},
        "version": {
            "actual": dict(versions),
            "expected": {EXPECTED_VERSION: EXPECTED_ROWS},
            "pass": dict(versions) == {EXPECTED_VERSION: EXPECTED_ROWS},
        },
        "doi_gesis": {
            "actual": dict(dois),
            "expected": {EXPECTED_DOI_GESIS: EXPECTED_ROWS},
            "pass": dict(dois) == {EXPECTED_DOI_GESIS: EXPECTED_ROWS},
        },
        "study_counts": {
            "actual": dict(studies),
            "expected": EXPECTED_STUDIES,
            "pass": dict(studies) == EXPECTED_STUDIES,
        },
        "country_count": {
            "actual": len(country_stats),
            "expected": EXPECTED_COUNTRIES,
            "pass": len(country_stats) == EXPECTED_COUNTRIES,
        },
    }
    report = {
        "file": str(csv_path),
        "sha256": sha256(csv_path),
        "encoding": "UTF-8 (BOM accepted)",
        "checks": checks,
        "all_checks_pass": all(check["pass"] for check in checks.values()),
        "waves": dict(waves),
    }
    return report, header, country_stats, observed_values


def write_metadata(output_dir: Path, metadata: dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "variable_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "Joint EVS/WVS 2017-2022",
                "version": "5.0.0",
                "source": "ZA7505_Codebook_v5.0.pdf",
                "variables": metadata,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "variable_labels.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("variable", "label", "description", "value_label_count"))
        for item in metadata.values():
            writer.writerow(
                (
                    item["variable"],
                    item["label"],
                    item["description"],
                    len(item["value_labels"]),
                )
            )

    with (output_dir / "value_labels.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("variable", "value", "value_label"))
        for item in metadata.values():
            for value, value_label in item["value_labels"].items():
                writer.writerow((item["variable"], value, value_label))

    with (output_dir / "missing_codes.json").open("w", encoding="utf-8") as handle:
        json.dump(MISSING_CODES, handle, ensure_ascii=False, indent=2)


def write_countries(
    output_dir: Path,
    country_stats: dict[tuple[str, str], dict],
    metadata: dict[str, dict],
) -> None:
    country_labels = metadata["cntry"]["value_labels"]
    with (output_dir / "country_codes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("cntry", "cntry_AN", "country_label", "n", "study_counts", "survey_years")
        )
        for (numeric, alpha), stats in sorted(
            country_stats.items(), key=lambda item: int(item[0][0])
        ):
            writer.writerow(
                (
                    numeric,
                    alpha,
                    country_labels.get(numeric, ""),
                    stats["n"],
                    json.dumps(dict(stats["studies"]), sort_keys=True),
                    ";".join(sorted(stats["years"])),
                )
            )


def write_report(output_dir: Path, report: dict) -> None:
    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    status = "PASS" if report["all_checks_pass"] else "FAIL"
    lines = [
        "# Joint EVS/WVS CSV Validation",
        "",
        f"- Overall: **{status}**",
        f"- SHA-256: `{report['sha256']}`",
        f"- Encoding: {report['encoding']}",
        "",
        "| Check | Actual | Expected | Result |",
        "|---|---:|---:|---|",
    ]
    for name, check in report["checks"].items():
        actual = json.dumps(check["actual"], ensure_ascii=False, sort_keys=True)
        expected = json.dumps(check["expected"], ensure_ascii=False, sort_keys=True)
        result = "PASS" if check["pass"] else "FAIL"
        lines.append(f"| `{name}` | `{actual}` | `{expected}` | {result} |")
    lines.extend(
        (
            "",
            "The checks establish that this is the complete Joint EVS/WVS v5.0 CSV,",
            "not that every variable is suitable for the planned experiment. Item coverage,",
            "missingness, survey source, fieldwork year, and measurement comparability still",
            "require separate screening before country or target selection.",
            "",
        )
    )
    (output_dir / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")


def write_checksums(dataset_dir: Path) -> None:
    paths = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    lines = [f"{sha256(path)}  {path.relative_to(dataset_dir).as_posix()}" for path in sorted(paths)]
    (dataset_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    csv_path = dataset_dir / "EVS_WVS_Joint_Csv_v5_0.csv"
    codebook = dataset_dir / "documentation" / "ZA7505_Codebook_v5.0.pdf"
    output_dir = dataset_dir / "metadata"

    report, variables, country_stats, observed_values = validate_csv(csv_path)
    metadata = parse_codebook(codebook, variables, observed_values)
    write_metadata(output_dir, metadata)
    write_countries(output_dir, country_stats, metadata)
    write_report(output_dir, report)
    write_checksums(dataset_dir)

    print(f"Validation: {'PASS' if report['all_checks_pass'] else 'FAIL'}")
    print(f"Variables documented: {len(metadata)}")
    print(f"Countries documented: {len(country_stats)}")
    print(f"Output: {output_dir}")
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
