#!/usr/bin/env python3
"""Screen survey items and build a restricted balanced pilot sample.

The script reads the local six-country respondent panel, uses deterministic
respondent-level splits, and evaluates whether country improves held-out
response forecasts beyond ten prior answers. It never calls an LLM.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


SEED = "aaai-country-metadata-v1"
COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")
COUNTRY_LABELS = {
    "CN": "China",
    "US": "United States",
    "JO": "Jordan",
    "FR": "France",
    "GB": "Great Britain",
    "IT": "Italy",
}
MISSING_VALUES = {"", "-5", "-4", "-3", "-2", "-1"}

# Broad prior responses that do not directly name a country, government, or
# local institution. These remain fixed while candidate targets are screened.
EVIDENCE_ITEMS = (
    "A006",
    "A165",
    "B008",
    "C001",
    "C039",
    "D059",
    "E023",
    "E035",
    "E037",
    "F063",
)

# Candidate social-value targets with common coding in the frozen six-country
# panel. Direct evaluations of national institutions are excluded.
TARGET_CANDIDATES = (
    "A001",
    "A002",
    "A003",
    "A004",
    "A005",
    "A008",
    "A009",
    "A170",
    "A173",
    "A027",
    "A029",
    "A030",
    "A032",
    "A034",
    "A035",
    "A038",
    "A039",
    "A040",
    "A041",
    "A042",
    "A124_02",
    "A124_03",
    "A124_06",
    "A124_08",
    "A124_09",
    "C002",
    "C038",
    "C041",
    "D054",
    "D060",
    "D061",
    "D078",
    "D081",
    "E012",
    "E015",
    "E018",
    "E025",
    "E026",
    "E027",
    "E028",
    "E033",
    "E036",
    "E039",
    "F118",
    "F120",
    "F121",
)

SPLIT_BOUNDS = (
    ("reference", 0.50),
    ("development", 0.80),
    ("pilot", 0.90),
    ("confirmatory", 1.00),
)
N_FOLDS = 5
N_BOOTSTRAP = 1_000
PILOT_PROFILES_PER_COUNTRY = 12
PILOT_LABEL_ASSIGNMENT_METHOD = "legacy deterministic cyclic assignment"
PILOT_HIGH_ITEMS = 4
PILOT_CONTROL_ITEMS = 2
MIN_DEVELOPMENT_PER_COUNTRY = 100


def stable_unit(*parts: str) -> float:
    payload = "\x1f".join((SEED, *parts)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / 2**64


def respondent_split(unique_id: str) -> str:
    value = stable_unit("split", unique_id)
    for name, upper in SPLIT_BOUNDS:
        if value < upper:
            return name
    raise AssertionError("split bounds do not cover [0, 1)")


def respondent_fold(unique_id: str) -> int:
    return min(N_FOLDS - 1, int(stable_unit("fold", unique_id) * N_FOLDS))


def positive_codes(metadata: dict[str, Any], variable: str) -> list[str]:
    labels = metadata[variable]["value_labels"]
    return sorted(
        (code for code in labels if code not in MISSING_VALUES and float(code) >= 0),
        key=float,
    )


def answer_text(metadata: dict[str, Any], variable: str, code: str) -> str:
    return metadata[variable]["value_labels"].get(code, code)


def encode_rows(
    rows: list[dict[str, str]], include_country: bool
) -> list[list[str]]:
    encoded = [
        [row[item] if row[item] not in MISSING_VALUES else "<MISSING>" for item in EVIDENCE_ITEMS]
        for row in rows
    ]
    if include_country:
        for values, row in zip(encoded, rows):
            values.append(row["cntry_AN"])
    return encoded


def equal_country_weights(rows: list[dict[str, str]]) -> np.ndarray:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[row["cntry_AN"]] += survey_weight(row)
    scale = len(rows) / len(totals)
    return np.asarray(
        [scale * survey_weight(row) / totals[row["cntry_AN"]] for row in rows],
        dtype=float,
    )


def survey_weight(row: dict[str, str]) -> float:
    try:
        value = float(row["gwght"])
    except (KeyError, TypeError, ValueError):
        return 1.0
    return value if math.isfinite(value) and value > 0 else 1.0


def align_probabilities(
    probabilities: np.ndarray, learned: Iterable[str], classes: list[str]
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(classes)), dtype=float)
    indexes = {value: index for index, value in enumerate(classes)}
    for source, value in enumerate(learned):
        if value in indexes:
            aligned[:, indexes[value]] = probabilities[:, source]
    return aligned


def brier_rows(probabilities: np.ndarray, y: list[str], classes: list[str]) -> np.ndarray:
    indexes = {value: index for index, value in enumerate(classes)}
    truth = np.zeros_like(probabilities)
    truth[np.arange(len(y)), [indexes[value] for value in y]] = 1.0
    return np.sum((probabilities - truth) ** 2, axis=1)


def macro_country_mean(values: np.ndarray, rows: list[dict[str, str]]) -> float:
    by_country: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for value, row in zip(values, rows):
        by_country[row["cntry_AN"]].append((float(value), survey_weight(row)))
    return float(
        np.mean(
            [
                np.average(
                    [value for value, _ in by_country[country]],
                    weights=[weight for _, weight in by_country[country]],
                )
                for country in COUNTRIES
            ]
        )
    )


def marginal_probabilities(
    train_rows: list[dict[str, str]], target: str, classes: list[str]
) -> np.ndarray:
    by_country: dict[str, defaultdict[str, float]] = {
        country: defaultdict(float) for country in COUNTRIES
    }
    for row in train_rows:
        by_country[row["cntry_AN"]][row[target]] += survey_weight(row)
    vectors = []
    for country in COUNTRIES:
        total = sum(by_country[country].values())
        vectors.append(np.asarray([by_country[country][code] / total for code in classes]))
    return np.mean(vectors, axis=0)


def fit_probabilities(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    target: str,
    classes: list[str],
    include_country: bool,
) -> np.ndarray:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    train_x = encoder.fit_transform(encode_rows(train_rows, include_country))
    test_x = encoder.transform(encode_rows(test_rows, include_country))
    train_y = [row[target] for row in train_rows]
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        model.fit(train_x, train_y, sample_weight=equal_country_weights(train_rows))
    return align_probabilities(model.predict_proba(test_x), model.classes_, classes)


def cross_fitted_scores(
    rows: list[dict[str, str]], target: str, classes: list[str]
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    eligible = [row for row in rows if row[target] not in MISSING_VALUES]
    marginal_scores = np.empty(len(eligible), dtype=float)
    evidence_scores = np.empty(len(eligible), dtype=float)
    country_scores = np.empty(len(eligible), dtype=float)
    for fold in range(N_FOLDS):
        train_indexes = [i for i, row in enumerate(eligible) if respondent_fold(row["uniqid"]) != fold]
        test_indexes = [i for i, row in enumerate(eligible) if respondent_fold(row["uniqid"]) == fold]
        train = [eligible[i] for i in train_indexes]
        test = [eligible[i] for i in test_indexes]
        if not test or set(row["cntry_AN"] for row in train) != set(COUNTRIES):
            raise ValueError(f"Invalid fold {fold} for {target}")
        y = [row[target] for row in test]
        marginal = marginal_probabilities(train, target, classes)
        marginal_p = np.repeat(marginal[None, :], len(test), axis=0)
        evidence_p = fit_probabilities(train, test, target, classes, False)
        country_p = fit_probabilities(train, test, target, classes, True)
        marginal_scores[test_indexes] = brier_rows(marginal_p, y, classes)
        evidence_scores[test_indexes] = brier_rows(evidence_p, y, classes)
        country_scores[test_indexes] = brier_rows(country_p, y, classes)
    return eligible, marginal_scores, evidence_scores, country_scores


def stratified_bootstrap_ci(
    values: np.ndarray, rows: list[dict[str, str]], seed: str
) -> tuple[float, float]:
    grouped = {}
    for country in COUNTRIES:
        indexes = [index for index, row in enumerate(rows) if row["cntry_AN"] == country]
        grouped[country] = (
            np.asarray([values[index] for index in indexes]),
            np.asarray([survey_weight(rows[index]) for index in indexes]),
        )
    rng_seed = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    estimates = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        country_estimates = []
        for country in COUNTRIES:
            country_values, country_weights = grouped[country]
            sampled = rng.integers(0, len(country_values), len(country_values))
            country_estimates.append(
                np.average(country_values[sampled], weights=country_weights[sampled])
            )
        estimates[index] = np.mean(country_estimates)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        mask = first > 0
        return float(np.sum(first[mask] * np.log2(first[mask] / second[mask])))

    return (kl(left, midpoint) + kl(right, midpoint)) / 2


def country_association(
    rows: list[dict[str, str]], target: str, classes: list[str]
) -> float:
    distributions = []
    for country in COUNTRIES:
        counts: defaultdict[str, float] = defaultdict(float)
        for row in rows:
            if row["cntry_AN"] == country and row[target] not in MISSING_VALUES:
                counts[row[target]] += survey_weight(row)
        total = sum(counts.values())
        distributions.append(np.asarray([counts[code] / total for code in classes]))
    pooled = np.mean(distributions, axis=0)
    return float(np.mean([js_divergence(distribution, pooled) for distribution in distributions]))


def reference_distributions(
    rows: list[dict[str, str]], target: str, classes: list[str]
) -> dict[str, list[float]]:
    result = {}
    for country in COUNTRIES:
        counts: defaultdict[str, float] = defaultdict(float)
        for row in rows:
            if row["cntry_AN"] == country and row[target] not in MISSING_VALUES:
                counts[row[target]] += survey_weight(row)
        total = sum(counts.values())
        result[country] = [counts[code] / total for code in classes]
    return result


def evaluate_target(
    development_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    metadata: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    classes = positive_codes(metadata, target)
    valid_counts = Counter(
        row["cntry_AN"]
        for row in development_rows
        if row[target] not in MISSING_VALUES
    )
    sparse = {
        country: valid_counts[country]
        for country in COUNTRIES
        if valid_counts[country] < MIN_DEVELOPMENT_PER_COUNTRY
    }
    if sparse:
        raise ValueError(f"insufficient development coverage: {sparse}")
    rows, marginal, evidence, country = cross_fitted_scores(
        development_rows, target, classes
    )
    evidence_gain_rows = marginal - evidence
    country_gain_rows = evidence - country
    evidence_gain = macro_country_mean(evidence_gain_rows, rows)
    country_gain = macro_country_mean(country_gain_rows, rows)
    evidence_ci = stratified_bootstrap_ci(evidence_gain_rows, rows, target + ":evidence")
    country_ci = stratified_bootstrap_ci(country_gain_rows, rows, target + ":country")
    return {
        "variable": target,
        "label": metadata[target]["label"],
        "option_count": len(classes),
        "n_development": len(rows),
        "n_by_country": dict(Counter(row["cntry_AN"] for row in rows)),
        "option_codes": classes,
        "reference_distributions": reference_distributions(
            reference_rows, target, classes
        ),
        "reference_country_js_bits": country_association(reference_rows, target, classes),
        "brier_marginal": macro_country_mean(marginal, rows),
        "brier_evidence": macro_country_mean(evidence, rows),
        "brier_evidence_country": macro_country_mean(country, rows),
        "evidence_gain": evidence_gain,
        "evidence_gain_ci_low": evidence_ci[0],
        "evidence_gain_ci_high": evidence_ci[1],
        "country_incremental_gain": country_gain,
        "country_gain_ci_low": country_ci[0],
        "country_gain_ci_high": country_ci[1],
    }


def select_targets(results: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    evidence_valid = [row for row in results if row["evidence_gain_ci_low"] > 0]
    high = sorted(
        evidence_valid,
        key=lambda row: (row["country_gain_ci_low"], row["country_incremental_gain"]),
        reverse=True,
    )[:PILOT_HIGH_ITEMS]
    used = {row["variable"] for row in high}
    null_compatible = [
        row
        for row in evidence_valid
        if row["variable"] not in used
        and row["country_gain_ci_low"] <= 0 <= row["country_gain_ci_high"]
    ]
    controls = sorted(
        null_compatible,
        key=lambda row: (
            row["reference_country_js_bits"],
            abs(row["country_incremental_gain"]),
        ),
    )[:PILOT_CONTROL_ITEMS]
    if len(high) < PILOT_HIGH_ITEMS or len(controls) < PILOT_CONTROL_ITEMS:
        raise ValueError("Not enough evidence-valid high and control targets")
    return [row["variable"] for row in high], [row["variable"] for row in controls]


def serialise_item(metadata: dict[str, Any], variable: str, code: str) -> dict[str, str]:
    return {
        "item_id": variable,
        "question": metadata[variable]["label"],
        "answer_code": code,
        "answer": answer_text(metadata, variable, code),
    }


def build_pilot_profiles(
    pilot_rows: list[dict[str, str]],
    metadata: dict[str, Any],
    targets: list[str],
) -> list[dict[str, Any]]:
    selected: dict[str, list[dict[str, str]]] = {}
    required = EVIDENCE_ITEMS + tuple(targets)
    for country in COUNTRIES:
        complete = [
            row
            for row in pilot_rows
            if row["cntry_AN"] == country
            and all(row[variable] not in MISSING_VALUES for variable in required)
        ]
        complete.sort(key=lambda row: stable_unit("pilot-order", row["uniqid"]))
        if len(complete) < PILOT_PROFILES_PER_COUNTRY:
            raise ValueError(
                f"{country} has only {len(complete)} complete pilot rows; "
                f"need {PILOT_PROFILES_PER_COUNTRY}"
            )
        selected[country] = complete[:PILOT_PROFILES_PER_COUNTRY]

    profiles = []
    labels = [COUNTRY_LABELS[country] for country in COUNTRIES]
    for country_index, country in enumerate(COUNTRIES):
        rows = selected[country]
        for index, row in enumerate(rows):
            shuffled = rows[(index + 1) % len(rows)]
            profiles.append(
                {
                    "profile_id": f"P-{country}-{index + 1:03d}",
                    "true_country_code": country,
                    "true_country_label": COUNTRY_LABELS[country],
                    "balanced_random_country_label": labels[(index + country_index) % len(labels)],
                    "evidence": [
                        serialise_item(metadata, variable, row[variable])
                        for variable in EVIDENCE_ITEMS
                    ],
                    "shuffled_evidence": [
                        serialise_item(metadata, variable, shuffled[variable])
                        for variable in EVIDENCE_ITEMS
                    ],
                    "targets": [
                        serialise_item(metadata, variable, row[variable])
                        for variable in targets
                    ],
                    "target_options": {
                        variable: [
                            {
                                "answer_code": code,
                                "answer": answer_text(metadata, variable, code),
                            }
                            for code in positive_codes(metadata, variable)
                        ]
                        for variable in targets
                    },
                }
            )
    return profiles


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    dataset_dir = Path(__file__).resolve().parents[1]
    panel_path = dataset_dir / "derived" / "country_panel_6.csv"
    metadata_path = dataset_dir / "metadata" / "variable_metadata.json"
    analysis_dir = dataset_dir / "analysis"
    derived_dir = dataset_dir / "derived"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["variables"]
    required = {"uniqid", "cntry_AN", *EVIDENCE_ITEMS, *TARGET_CANDIDATES}
    with panel_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Panel is missing variables: {sorted(missing)}")
        rows = list(reader)

    split_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        split_rows[respondent_split(row["uniqid"])].append(row)

    results = []
    skipped = []
    for index, target in enumerate(TARGET_CANDIDATES, start=1):
        print(f"[{index:02d}/{len(TARGET_CANDIDATES)}] {target}", flush=True)
        try:
            result = evaluate_target(
                split_rows["development"],
                split_rows["reference"],
                metadata,
                target,
            )
        except ValueError as error:
            skipped.append({"variable": target, "reason": str(error)})
            print(f"  skipped: {error}", flush=True)
            continue
        results.append(result)

    high_targets, control_targets = select_targets(results)
    pilot_targets = high_targets + control_targets
    profiles = build_pilot_profiles(split_rows["pilot"], metadata, pilot_targets)

    report_csv = analysis_dir / "pilot_gate_results.csv"
    flat_fields = [key for key in results[0] if key != "n_by_country"]
    with report_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in flat_fields})

    restricted_profiles = derived_dir / "pilot_profiles_6x12.jsonl"
    with restricted_profiles.open("w", encoding="utf-8", newline="\n") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile, ensure_ascii=False) + "\n")

    split_counts = {
        split: dict(Counter(row["cntry_AN"] for row in values))
        for split, values in split_rows.items()
    }
    report = {
        "dataset": "Joint EVS/WVS 2017-2022 v5.0",
        "panel_sha256": sha256(panel_path),
        "method": {
            "seed": SEED,
            "split_bounds": SPLIT_BOUNDS,
            "folds": N_FOLDS,
            "bootstrap_replicates": N_BOOTSTRAP,
            "minimum_development_valid_per_country": MIN_DEVELOPMENT_PER_COUNTRY,
            "score": "multiclass Brier; equal-country macro mean",
            "model": "L2 multinomial logistic regression with one-hot prior answers",
        },
        "split_counts": split_counts,
        "evidence_items": list(EVIDENCE_ITEMS),
        "candidate_count": len(TARGET_CANDIDATES),
        "evaluated_candidate_count": len(results),
        "skipped_candidates": skipped,
        "high_country_value_targets": high_targets,
        "low_country_value_controls": control_targets,
        "pilot_targets_in_order": pilot_targets,
        "pilot_profiles": {
            "count": len(profiles),
            "per_country": PILOT_PROFILES_PER_COUNTRY,
            "restricted_file": restricted_profiles.relative_to(dataset_dir).as_posix(),
            "restricted_file_sha256": sha256(restricted_profiles),
            "contains_source_uniqid": False,
            "label_assignment": {
                "method": PILOT_LABEL_ASSIGNMENT_METHOD,
                "randomized": False,
                "scope": "screening/measurement pilot only; not valid for confirmatory randomization",
            },
        },
        "results": results,
    }
    report_path = analysis_dir / "pilot_gate_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"High-value targets: {', '.join(high_targets)}")
    print(f"Negative controls: {', '.join(control_targets)}")
    print(f"Aggregate report: {report_path}")
    print(f"Restricted profiles: {restricted_profiles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
