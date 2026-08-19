#!/usr/bin/env python3
"""Build the frozen six-country analysis panel from Joint EVS/WVS v5.0."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PANEL = (
    {"cntry_AN": "CN", "studytit": "WVS7", "year": "2018", "label": "China", "expected_n": 3036},
    {"cntry_AN": "US", "studytit": "WVS7", "year": "2017", "label": "United States", "expected_n": 2596},
    {"cntry_AN": "JO", "studytit": "WVS7", "year": "2018", "label": "Jordan", "expected_n": 1203},
    {"cntry_AN": "FR", "studytit": "EVS5", "year": "2018", "label": "France", "expected_n": 1870},
    {"cntry_AN": "GB", "studytit": "EVS5", "year": "2018", "label": "Great Britain", "expected_n": 1788},
    {"cntry_AN": "IT", "studytit": "EVS5", "year": "2018", "label": "Italy", "expected_n": 2277},
)

PRELIMINARY_ITEMS = (
    "A001", "A002", "A003", "A004", "A005", "A006", "A008", "A009",
    "A170", "A173", "A027", "A029", "A030", "A032", "A034", "A035",
    "A038", "A039", "A040", "A041", "A042", "A124_02", "A124_03",
    "A124_06", "A124_08", "A124_09", "A165", "B008", "C001", "C002",
    "D081", "D026_03", "D026_05", "C038", "C039", "C041", "D001_B",
    "G007_18_B", "G007_33_B", "G007_34_B", "G007_35_B", "G007_36_B",
    "D054", "D059", "D060", "D061", "D078", "E012", "E015", "E018",
    "E023", "E025", "E026", "E027", "E028", "E033", "E035", "E036",
    "E037", "E039", "F118", "F120", "F121",
)

MISSING_VALUES = {"", "-5", "-4", "-3", "-2", "-1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    dataset_dir = Path(__file__).resolve().parents[1]
    source = dataset_dir / "EVS_WVS_Joint_Csv_v5_0.csv"
    output_dir = dataset_dir / "derived"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "country_panel_6.csv"
    manifest_path = output_dir / "country_panel_6.manifest.json"
    coverage_path = output_dir / "country_panel_6.coverage.csv"

    selectors = {
        (item["cntry_AN"], item["studytit"], item["year"]): item for item in PANEL
    }
    counts: Counter[tuple[str, str, str]] = Counter()
    valid_counts = {
        key: Counter() for key in selectors
    }
    rows_written = 0

    with source.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError("Source CSV has no header")
        required = {"cntry_AN", "studytit", "year", "uniqid"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Source CSV is missing required columns: {sorted(missing)}")

        with output.open("w", encoding="utf-8-sig", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = (row["cntry_AN"], row["studytit"], row["year"])
                if key not in selectors:
                    continue
                writer.writerow(row)
                counts[key] += 1
                for variable in PRELIMINARY_ITEMS:
                    if row[variable] not in MISSING_VALUES:
                        valid_counts[key][variable] += 1
                rows_written += 1

    expected_total = sum(item["expected_n"] for item in PANEL)
    if rows_written != expected_total:
        raise ValueError(f"Expected {expected_total} rows, wrote {rows_written}")
    for key, item in selectors.items():
        if counts[key] != item["expected_n"]:
            raise ValueError(
                f"{key} expected {item['expected_n']} rows, wrote {counts[key]}"
            )

    with coverage_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "cntry_AN",
                "country_label",
                "studytit",
                "year",
                "n",
                "preliminary_item_count",
                "items_at_least_90pct_valid",
                "items_below_90pct",
            )
        )
        for key, item in selectors.items():
            below = [
                variable
                for variable in PRELIMINARY_ITEMS
                if valid_counts[key][variable] / counts[key] < 0.9
            ]
            writer.writerow(
                (
                    item["cntry_AN"],
                    item["label"],
                    item["studytit"],
                    item["year"],
                    counts[key],
                    len(PRELIMINARY_ITEMS),
                    len(PRELIMINARY_ITEMS) - len(below),
                    ";".join(below),
                )
            )

    manifest = {
        "dataset": "Joint EVS/WVS 2017-2022",
        "version": "5.0.0",
        "source_file": source.name,
        "source_sha256": sha256(source),
        "output_file": output.relative_to(dataset_dir).as_posix(),
        "output_sha256": sha256(output),
        "coverage_file": coverage_path.relative_to(dataset_dir).as_posix(),
        "coverage_sha256": sha256(coverage_path),
        "row_count": rows_written,
        "column_count": len(reader.fieldnames),
        "selection_key": ["cntry_AN", "studytit", "year"],
        "preliminary_coverage": {
            "item_count": len(PRELIMINARY_ITEMS),
            "validity_threshold": 0.9,
            "missing_values": sorted(MISSING_VALUES),
            "purpose": "Coverage screen only; not the frozen target set.",
        },
        "panel": [
            {
                **item,
                "actual_n": counts[(item["cntry_AN"], item["studytit"], item["year"])],
            }
            for item in PANEL
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {rows_written} rows to {output}")
    print(f"Coverage: {coverage_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
