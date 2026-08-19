#!/usr/bin/env python3
"""Explore model-specific country affinity using the clean replacement panel.

This is a post-review secondary analysis. It uses no external API and never
modifies the immutable result JSONL files. Eight checklist items are taken only
from the corrected 72-profile rerun; the other twelve items retain the original
504-profile results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import xlogy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pilot.run_confirmatory_strict_inference import (  # noqa: E402
    OFFICIAL_MODELS,
    audit_results,
    expected_call_map,
    latest_rows,
)
from api_pilot.run_gate_c_strict_inference import (  # noqa: E402
    COUNTRIES,
    item_result_index,
    load_jsonl,
    reference_rows_and_anchors,
)
from api_pilot.strict_inference import holm_adjust  # noqa: E402
from api_pilot.survey_pilot_runner import load_task_manifest  # noqa: E402


COUNTRY_LABELS = {
    "CN": "China",
    "FR": "France",
    "GB": "Great Britain",
    "IT": "Italy",
    "JO": "Jordan",
    "US": "United States",
}
LABEL_TO_CODE = {label: code for code, label in COUNTRY_LABELS.items()}
PROVIDERS = tuple(sorted(OFFICIAL_MODELS))
CONDITIONS = ("M0", "MO", "MR", "MV")

ORIGINAL_RESULTS = ROOT / "api_pilot/restricted_outputs/confirmatory_core_v2_raw_five_model.jsonl"
ORIGINAL_PROFILES = ROOT / "dataset/derived/confirmatory_profiles_6x84.jsonl"
ORIGINAL_TASK = ROOT / "dataset/analysis/confirmatory_core_v2_raw_five_model_manifest_v1.json"
ORIGINAL_PROMPT = ROOT / "api_pilot/prompts/survey-confirmatory-v4.json"
CORRECTED_RESULTS = ROOT / "api_pilot/restricted_outputs/checklist_correction_supplement_v1.jsonl"
CORRECTED_PROFILES = ROOT / "api_pilot/restricted_outputs/checklist_correction_profiles_6x12.jsonl"
CORRECTED_TASK = ROOT / "dataset/analysis/checklist_correction_supplement_manifest_v1.json"
CORRECTED_PROMPT = ROOT / "api_pilot/prompts/survey-checklist-correction-v1.json"
ANALYSIS_MANIFEST = ROOT / "dataset/analysis/confirmatory_manifest_v2.json"
AMENDMENT = ROOT / "dataset/analysis/checklist_conversion_amendment_v1.json"
HUMAN_SIGNOFF = ROOT / "api_pilot/human_review/human_quality_review_signoff.json"
PANEL = ROOT / "dataset/derived/country_panel_6.csv"
METADATA = ROOT / "dataset/metadata/variable_metadata.json"

OUT_JSON = ROOT / "api_pilot/outputs/model_country_preference_analysis.json"
OUT_METRICS = ROOT / "api_pilot/outputs/final_tables/table_9_model_country_preference_metrics.csv"
OUT_TESTS = ROOT / "api_pilot/outputs/final_tables/table_10_model_country_preference_global_tests.csv"
OUT_RESEMBLANCE = ROOT / "api_pilot/outputs/final_tables/table_11_derived_anchor_resemblance.csv"
OUT_NEAREST_DIAGNOSTIC = ROOT / "api_pilot/outputs/final_tables/table_12_profile_nearest_anchor_diagnostic.csv"
OUT_LOO = ROOT / "api_pilot/outputs/final_tables/table_13_model_country_leave_one_item_out.csv"
OUT_AUDIT = ROOT / "api_pilot/outputs/final_tables/table_14_clean_replacement_audit.csv"
OUT_CORRELATIONS = ROOT / "api_pilot/outputs/final_tables/table_15_model_country_profile_correlations.csv"
OUT_CONTRASTS = ROOT / "api_pilot/outputs/final_tables/table_16_targeted_country_contrasts.csv"

PRIMARY_METRICS = (
    "m0_anchor_l2",
    "mo_tv",
    "mo_regret",
    "m0_brier",
)


@dataclass(frozen=True)
class ItemPanel:
    item_id: str
    source: str
    profile_ids: tuple[str, ...]
    true_country: np.ndarray
    displayed_label: np.ndarray
    weights: np.ndarray
    truth_index: np.ndarray
    forecasts: dict[str, np.ndarray]


@dataclass(frozen=True)
class AnalysisView:
    name: str
    item_ids: tuple[str, ...]
    panels: dict[str, ItemPanel]
    role: str


@dataclass(frozen=True)
class EvaluationSamples:
    """Per-item profile positions for one clustered bootstrap replicate."""

    country: dict[str, dict[int, np.ndarray]]
    label: dict[str, dict[tuple[int, int], np.ndarray]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_items(manifest: dict[str, Any]) -> list[str]:
    values = [
        item
        for name in ("high_country_value", "representative", "equivalence_controls")
        for item in manifest["target_strata"][name]
    ]
    if len(values) != 20 or len(set(values)) != 20:
        raise ValueError("The frozen analysis manifest must contain 20 unique targets")
    return values


def audited_source(
    *,
    results_path: Path,
    profiles_path: Path,
    task_path: Path,
    prompt_path: Path,
    analysis: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    profiles = load_jsonl(profiles_path)
    task = load_task_manifest(task_path)
    if task is None:
        raise ValueError(f"Frozen task manifest is required: {task_path}")
    expected = expected_call_map(profiles, analysis, task, prompt_path)
    rows, audit = audit_results(latest_rows(results_path), expected)
    if not audit["complete"]:
        raise ValueError(f"Incomplete immutable source panel: {results_path}: {audit}")
    if any(row.get("evidence_variant") != "correct" for row in rows):
        raise ValueError("Raw source audit unexpectedly returned shuffled evidence")
    return rows, profiles, audit


def target_map(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["item_id"]: row for row in profile["targets"]}


def audit_profile_compatibility(
    original: list[dict[str, Any]], corrected: list[dict[str, Any]], items: list[str]
) -> None:
    source = {row["profile_id"]: row for row in original}
    for replacement in corrected:
        prior = source.get(replacement["profile_id"])
        if prior is None:
            raise ValueError("Corrected profile is not a subset of the original panel")
        for field in (
            "true_country_code",
            "balanced_random_country_label",
            "analysis_weight",
        ):
            if prior[field] != replacement[field]:
                raise ValueError(f"Replacement profile changed {field}")
        old_targets = target_map(prior)
        new_targets = target_map(replacement)
        for item in items:
            if old_targets[item]["answer_code"] != new_targets[item]["answer_code"]:
                raise ValueError(f"Replacement changed the held-out answer for {item}")
            old_codes = [value["answer_code"] for value in prior["target_options"][item]]
            new_codes = [value["answer_code"] for value in replacement["target_options"][item]]
            if old_codes != new_codes:
                raise ValueError(f"Replacement changed option codes for {item}")


def build_item_panel(
    *,
    item: str,
    source: str,
    rows_index: dict[tuple[str, str, str, str, str], dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> ItemPanel:
    country_order = {country: index for index, country in enumerate(COUNTRIES)}
    ordered = sorted(
        profiles,
        key=lambda row: (country_order[row["true_country_code"]], row["profile_id"]),
    )
    ids = tuple(row["profile_id"] for row in ordered)
    true_country = np.asarray([COUNTRIES.index(row["true_country_code"]) for row in ordered])
    displayed = np.asarray(
        [COUNTRIES.index(LABEL_TO_CODE[row["balanced_random_country_label"]]) for row in ordered]
    )
    weights = np.asarray([float(row["analysis_weight"]) for row in ordered], dtype=float)
    options = [value["answer_code"] for value in ordered[0]["target_options"][item]]
    if any(
        [value["answer_code"] for value in profile["target_options"][item]] != options
        for profile in ordered
    ):
        raise ValueError(f"Profile option-code order differs within target {item}")
    truth = np.asarray(
        [
            options.index(target_map(profile)[item]["answer_code"])
            for profile in ordered
        ],
        dtype=int,
    )
    forecasts: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        values: list[list[list[float]]] = []
        for provider in PROVIDERS:
            provider_values = []
            for profile in ordered:
                key = (provider, profile["profile_id"], condition, "correct", item)
                row = rows_index.get(key)
                if row is None:
                    raise ValueError(f"Missing clean item cell: {key}")
                observed_codes = row["target_option_codes"][item]
                expected_codes = [
                    value["answer_code"] for value in profile["target_options"][item]
                ]
                if observed_codes != expected_codes:
                    raise ValueError(f"Option-code mismatch: {key}")
                expected_displayed = (
                    None
                    if condition == "M0"
                    else profile["true_country_code"]
                    if condition == "MV"
                    else LABEL_TO_CODE[profile["balanced_random_country_label"]]
                )
                if row.get("displayed_country_code") != expected_displayed:
                    raise ValueError(f"Displayed-country mismatch: {key}")
                vector = np.asarray(row["parsed_forecasts"][item], dtype=float)
                if (
                    vector.shape != (len(expected_codes),)
                    or not np.all(np.isfinite(vector))
                    or np.any(vector < 0)
                    or np.any(vector > 1)
                    or not np.isclose(vector.sum(), 1.0, atol=1e-12)
                ):
                    raise ValueError(f"Invalid audited forecast: {key}")
                provider_values.append(vector.tolist())
            values.append(provider_values)
        forecasts[condition] = np.asarray(values, dtype=float)
    return ItemPanel(
        item_id=item,
        source=source,
        profile_ids=ids,
        true_country=true_country,
        displayed_label=displayed,
        weights=weights,
        truth_index=truth,
        forecasts=forecasts,
    )


def build_views() -> tuple[
    dict[str, AnalysisView],
    dict[str, Any],
    Any,
    dict[str, dict[str, list[float]]],
]:
    analysis = load_json(ANALYSIS_MANIFEST)
    all_items = flatten_items(analysis)
    amendment = load_json(AMENDMENT)
    affected = list(amendment["affected_items"])
    if len(affected) != 8 or not set(affected).issubset(all_items):
        raise ValueError("Checklist amendment does not identify the expected eight targets")
    signoff = load_json(HUMAN_SIGNOFF)
    if (
        signoff.get("status") != "approved"
        or signoff.get("unresolved_rejections")
        or set(signoff.get("superseded_original_rejections", [])) != set(affected)
    ):
        raise ValueError("Human question-quality audit is not cleanly signed off")
    unaffected = [item for item in all_items if item not in affected]

    original_rows, original_profiles, original_audit = audited_source(
        results_path=ORIGINAL_RESULTS,
        profiles_path=ORIGINAL_PROFILES,
        task_path=ORIGINAL_TASK,
        prompt_path=ORIGINAL_PROMPT,
        analysis=analysis,
    )
    corrected_rows, corrected_profiles, corrected_audit = audited_source(
        results_path=CORRECTED_RESULTS,
        profiles_path=CORRECTED_PROFILES,
        task_path=CORRECTED_TASK,
        prompt_path=CORRECTED_PROMPT,
        analysis=analysis,
    )
    if len(original_profiles) != 504 or len(corrected_profiles) != 72:
        raise ValueError("Unexpected original/corrected profile counts")
    audit_profile_compatibility(original_profiles, corrected_profiles, all_items)
    corrected_ids = {row["profile_id"] for row in corrected_profiles}
    common_original_profiles = [
        row for row in original_profiles if row["profile_id"] in corrected_ids
    ]
    if len(common_original_profiles) != 72:
        raise ValueError("The corrected profiles are not an exact original-panel subset")

    original_index = item_result_index(original_rows)
    corrected_index = item_result_index(corrected_rows)
    original_panels = {
        item: build_item_panel(
            item=item,
            source="original_correct_wording",
            rows_index=original_index,
            profiles=original_profiles,
        )
        for item in unaffected
    }
    original_common_panels = {
        item: build_item_panel(
            item=item,
            source="original_correct_wording_common72",
            rows_index=original_index,
            profiles=common_original_profiles,
        )
        for item in unaffected
    }
    corrected_panels = {
        item: build_item_panel(
            item=item,
            source="corrected_wording_replacement72",
            rows_index=corrected_index,
            profiles=corrected_profiles,
        )
        for item in affected
    }
    hybrid_panels = {
        item: corrected_panels[item] if item in affected else original_panels[item]
        for item in all_items
    }
    common_panels = {
        item: corrected_panels[item]
        if item in affected
        else original_common_panels[item]
        for item in all_items
    }
    views = {
        "clean_hybrid_all20": AnalysisView(
            "clean_hybrid_all20",
            tuple(all_items),
            hybrid_panels,
            "post_review_exploratory_clean_replacement_panel",
        ),
        "clean_common72_all20": AnalysisView(
            "clean_common72_all20",
            tuple(all_items),
            common_panels,
            "post_review_inferential_common_profile_sensitivity",
        ),
        "clean_unaffected504_12": AnalysisView(
            "clean_unaffected504_12",
            tuple(unaffected),
            original_panels,
            "post_review_inferential_unaffected_item_sensitivity",
        ),
    }
    resampler, anchors, reference_counts = reference_rows_and_anchors(
        PANEL, METADATA, all_items
    )
    source_audit = {
        "human_quality_signoff": {
            "status": signoff["status"],
            "reviewed_rows": signoff["reviewed_rows"],
            "correction_reviewed_rows": signoff["correction_reviewed_rows"],
            "unresolved_rejections": signoff["unresolved_rejections"],
        },
        "original_source": original_audit,
        "corrected_source": corrected_audit,
        "affected_items_replaced": affected,
        "unaffected_items_retained": unaffected,
        "hybrid_item_vectors": {
            "unaffected_original": len(unaffected) * 504 * len(PROVIDERS) * len(CONDITIONS),
            "affected_original_excluded": len(affected) * 504 * len(PROVIDERS) * len(CONDITIONS),
            "affected_corrected_used": len(affected) * 72 * len(PROVIDERS) * len(CONDITIONS),
            "total_used": (
                len(unaffected) * 504 + len(affected) * 72
            )
            * len(PROVIDERS)
            * len(CONDITIONS),
        },
        "profiles": {
            "original": len(original_profiles),
            "corrected_replacement_subset": len(corrected_profiles),
            "replacement_subset_of_original": True,
        },
        "reference_counts": reference_counts,
    }
    return views, source_audit, resampler, anchors


def weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sum(values * weights, axis=axis) / np.sum(weights)


def brier_matrix(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    result = np.sum(probabilities * probabilities, axis=2) + 1.0
    selected = np.take_along_axis(
        probabilities, truth[None, :, None], axis=2
    )[:, :, 0]
    return result - 2.0 * selected


def aligned_profiles(view: AnalysisView) -> bool:
    first = view.panels[view.item_ids[0]].profile_ids
    return all(view.panels[item].profile_ids == first for item in view.item_ids)


def two_way_interaction(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Two-way interaction requires a matrix")
    return (
        matrix
        - np.mean(matrix, axis=1, keepdims=True)
        - np.mean(matrix, axis=0, keepdims=True)
        + np.mean(matrix)
    )


def profile_centered(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.mean(matrix, axis=1, keepdims=True)


def jensen_shannon(probabilities: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence over the last axis with NumPy broadcasting."""
    midpoint = 0.5 * (probabilities + anchor)
    model_term = np.sum(
        xlogy(probabilities, probabilities) - xlogy(probabilities, midpoint),
        axis=-1,
    )
    anchor_values = np.broadcast_to(anchor, midpoint.shape)
    anchor_term = np.sum(
        xlogy(anchor_values, anchor_values) - xlogy(anchor_values, midpoint),
        axis=-1,
    )
    return 0.5 * (model_term + anchor_term)


def sampled_positions(
    panel: ItemPanel,
    item: str,
    stratum: int | tuple[int, int],
    samples: EvaluationSamples | None,
    *,
    label: bool,
) -> np.ndarray:
    if samples is not None:
        source = samples.label if label else samples.country
        return np.asarray(source[item][stratum], dtype=int)
    if label:
        true_index, label_index = stratum
        return np.flatnonzero(
            (panel.true_country == true_index)
            & (panel.displayed_label == label_index)
        )
    return np.flatnonzero(panel.true_country == stratum)


def calculate_metrics(
    view: AnalysisView,
    anchors: dict[str, dict[str, list[float]]],
    *,
    samples: EvaluationSamples | None = None,
    include_nearest: bool = True,
    include_js: bool = True,
) -> dict[str, np.ndarray]:
    n_models = len(PROVIDERS)
    n_countries = len(COUNTRIES)
    default_tv_distance = np.zeros((n_models, n_countries), dtype=float)
    default_l2_distance = np.zeros_like(default_tv_distance)
    default_js_distance = np.zeros_like(default_tv_distance)
    mo_tv = np.zeros_like(default_tv_distance)
    mr_tv = np.zeros_like(default_tv_distance)
    mo_regret = np.zeros_like(default_tv_distance)
    mr_regret = np.zeros_like(default_tv_distance)
    mo_tv_mismatch = np.zeros_like(default_tv_distance)
    mo_regret_mismatch = np.zeros_like(default_tv_distance)
    direction_num_o = np.zeros((n_models, n_countries, n_countries), dtype=float)
    direction_num_r = np.zeros_like(direction_num_o)
    direction_den = np.zeros((n_countries, n_countries), dtype=float)
    m0_brier = np.zeros((n_models, n_countries), dtype=float)
    mv_brier = np.zeros_like(m0_brier)
    item_count = len(view.item_ids)

    for item in view.item_ids:
        panel = view.panels[item]
        item_anchors = np.asarray([anchors[item][country] for country in COUNTRIES])
        pooled = np.mean(item_anchors, axis=0)
        directions = item_anchors - pooled
        norms = np.sum(directions * directions, axis=1)
        p0 = panel.forecasts["M0"]
        po = panel.forecasts["MO"]
        pr = panel.forecasts["MR"]
        pv = panel.forecasts["MV"]
        loss0 = brier_matrix(p0, panel.truth_index)
        losso = brier_matrix(po, panel.truth_index)
        lossr = brier_matrix(pr, panel.truth_index)
        lossv = brier_matrix(pv, panel.truth_index)

        country_positions = [
            sampled_positions(panel, item, index, samples, label=False)
            for index in range(n_countries)
        ]
        if len({len(value) for value in country_positions}) != 1:
            raise ValueError("Country strata must have a common profile count")
        country_index = np.stack(country_positions)
        country_weights = panel.weights[country_index]
        country_weights = country_weights / np.sum(
            country_weights, axis=1, keepdims=True
        )
        country_p0 = p0[:, country_index, :]
        differences = (
            country_p0[:, :, :, None, :]
            - item_anchors[None, None, None, :, :]
        )
        weights = country_weights[None, :, :, None]
        default_tv_distance += (
            np.sum(0.5 * np.sum(np.abs(differences), axis=4) * weights, axis=(1, 2))
            / (item_count * n_countries)
        )
        default_l2_distance += (
            np.sum(np.sum(differences * differences, axis=4) * weights, axis=(1, 2))
            / (item_count * n_countries)
        )
        if include_js:
            default_js_distance += (
                np.sum(
                    jensen_shannon(
                        country_p0[:, :, :, None, :],
                        item_anchors[None, None, None, :, :],
                    )
                    * weights,
                    axis=(1, 2),
                )
                / (item_count * n_countries)
            )
        m0_brier += (
            np.sum(loss0[:, country_index] * country_weights[None, :, :], axis=2)
            / item_count
        )
        mv_brier += (
            np.sum(lossv[:, country_index] * country_weights[None, :, :], axis=2)
            / item_count
        )

        label_positions = [
            [
                sampled_positions(
                    panel, item, (true_index, label_index), samples, label=True
                )
                for label_index in range(n_countries)
            ]
            for true_index in range(n_countries)
        ]
        if any(len(value) == 0 for row in label_positions for value in row):
            raise ValueError("Displayed-label cell is empty")
        if len({len(value) for row in label_positions for value in row}) != 1:
            raise ValueError("Displayed-label cells must have a common profile count")
        label_index = np.asarray(label_positions, dtype=int)
        label_weights = panel.weights[label_index]
        label_weights = label_weights / np.sum(
            label_weights, axis=2, keepdims=True
        )
        delta_o = po[:, label_index, :] - p0[:, label_index, :]
        delta_r = pr[:, label_index, :] - p0[:, label_index, :]
        projection_o = np.einsum("ptlno,lo->ptln", delta_o, directions)
        projection_r = np.einsum("ptlno,lo->ptln", delta_r, directions)
        weighted = label_weights[None, :, :, :]
        direction_num_o += (
            np.sum(projection_o * weighted, axis=3) / item_count
        )
        direction_num_r += (
            np.sum(projection_r * weighted, axis=3) / item_count
        )
        direction_den += np.broadcast_to(norms[None, :], direction_den.shape) / item_count
        cell_tv_o = np.sum(
            0.5 * np.sum(np.abs(delta_o), axis=4) * weighted, axis=3
        )
        cell_tv_r = np.sum(
            0.5 * np.sum(np.abs(delta_r), axis=4) * weighted, axis=3
        )
        cell_regret_o = np.sum(
            (losso[:, label_index] - loss0[:, label_index]) * weighted,
            axis=3,
        )
        cell_regret_r = np.sum(
            (lossr[:, label_index] - loss0[:, label_index]) * weighted,
            axis=3,
        )
        scale = item_count * n_countries
        mo_tv += np.sum(cell_tv_o, axis=1) / scale
        mr_tv += np.sum(cell_tv_r, axis=1) / scale
        mo_regret += np.sum(cell_regret_o, axis=1) / scale
        mr_regret += np.sum(cell_regret_r, axis=1) / scale
        keep = ~np.eye(n_countries, dtype=bool)
        mismatch_scale = item_count * (n_countries - 1)
        mo_tv_mismatch += np.sum(
            np.where(keep[None, :, :], cell_tv_o, 0.0), axis=1
        ) / mismatch_scale
        mo_regret_mismatch += np.sum(
            np.where(keep[None, :, :], cell_regret_o, 0.0), axis=1
        ) / mismatch_scale

    if np.any(direction_den <= 1e-15):
        raise ValueError("Pooled country-direction energy is zero")
    direction_o_cells = direction_num_o / direction_den[None, :, :]
    direction_r_cells = direction_num_r / direction_den[None, :, :]
    direction_o = np.mean(direction_o_cells, axis=1)
    direction_r = np.mean(direction_r_cells, axis=1)
    direction_o_mismatch = np.empty_like(direction_o)
    for label_index in range(n_countries):
        keep = np.arange(n_countries) != label_index
        direction_o_mismatch[:, label_index] = np.mean(
            direction_o_cells[:, keep, label_index], axis=1
        )
    result = {
        "m0_anchor_tv": default_tv_distance,
        "m0_anchor_l2": default_l2_distance,
        "m0_anchor_js": default_js_distance,
        "mo_tv": mo_tv,
        "mr_tv": mr_tv,
        "mo_tv_minus_mr_tv": mo_tv - mr_tv,
        "mo_regret": mo_regret,
        "mr_regret": mr_regret,
        "mo_regret_minus_mr_regret": mo_regret - mr_regret,
        "mo_tv_mismatch": mo_tv_mismatch,
        "mo_regret_mismatch": mo_regret_mismatch,
        "direction_o_pooled": direction_o,
        "direction_r_pooled": direction_r,
        "direction_o_minus_direction_r_pooled": direction_o - direction_r,
        "direction_o_mismatch_pooled": direction_o_mismatch,
        "m0_brier": m0_brier,
        "mv_brier": mv_brier,
        "utility_v": m0_brier - mv_brier,
    }

    if include_nearest:
        if not aligned_profiles(view):
            return result
        first = view.panels[view.item_ids[0]]
        nearest_share = np.zeros((n_models, n_countries), dtype=float)
        confusion = np.zeros((n_models, n_countries, n_countries), dtype=float)
        mean_margin = np.zeros(n_models, dtype=float)
        raw_counts = np.zeros((n_models, n_countries), dtype=int)
        for true_index in range(n_countries):
            positions = sampled_positions(
                first, view.item_ids[0], true_index, samples, label=False
            )
            distances = np.zeros((n_models, len(positions), n_countries), dtype=float)
            for item in view.item_ids:
                panel = view.panels[item]
                item_anchors = np.asarray(
                    [anchors[item][country] for country in COUNTRIES]
                )
                difference = (
                    panel.forecasts["M0"][:, positions, None, :]
                    - item_anchors[None, None, :, :]
                ) / item_count
                distances += np.sum(difference * difference, axis=3) * item_count
            predicted = np.argmin(distances, axis=2)
            ordered = np.sort(distances, axis=2)
            weights = first.weights[positions]
            normalized = weights / weights.sum()
            for provider_index in range(n_models):
                for label_index in range(n_countries):
                    share = np.sum(
                        normalized[predicted[provider_index] == label_index]
                    )
                    confusion[provider_index, true_index, label_index] = share
                    nearest_share[provider_index, label_index] += share / n_countries
                    if samples is None:
                        raw_counts[provider_index, label_index] += int(
                            np.count_nonzero(predicted[provider_index] == label_index)
                        )
                mean_margin[provider_index] += (
                    np.sum((ordered[provider_index, :, 1] - ordered[provider_index, :, 0]) * normalized)
                    / n_countries
                )
        result.update(
            {
                "nearest_share": nearest_share,
                "nearest_confusion": confusion,
                "nearest_margin": mean_margin,
                "nearest_raw_counts": raw_counts,
            }
        )
    return result


def stratified_samples(
    view: AnalysisView, rng: np.random.Generator
) -> EvaluationSamples:
    """Resample linked profiles once, including the nested 72-in-504 panel."""
    panels = [view.panels[item] for item in view.item_ids]
    profile_sets = [set(panel.profile_ids) for panel in panels]
    common_ids = set.intersection(*profile_sets)
    full_ids = set.union(*profile_sets)
    if not common_ids:
        raise ValueError("Analysis items have no shared profile cluster")
    if any(ids not in (common_ids, full_ids) for ids in profile_sets):
        raise ValueError("Unsupported partially overlapping item panels")
    canonical = max(panels, key=lambda panel: len(panel.profile_ids))
    canonical_index = {
        profile_id: index for index, profile_id in enumerate(canonical.profile_ids)
    }
    attributes = {
        profile_id: (
            int(canonical.true_country[index]),
            int(canonical.displayed_label[index]),
        )
        for profile_id, index in canonical_index.items()
    }
    item_indexes = {
        item: {
            profile_id: index
            for index, profile_id in enumerate(view.panels[item].profile_ids)
        }
        for item in view.item_ids
    }
    country_samples: dict[str, dict[int, np.ndarray]] = {
        item: {} for item in view.item_ids
    }
    label_samples: dict[str, dict[tuple[int, int], np.ndarray]] = {
        item: {} for item in view.item_ids
    }

    def draw_ids(pool: list[str]) -> list[str]:
        if not pool:
            return []
        indexes = rng.integers(0, len(pool), len(pool))
        return [pool[index] for index in indexes]

    for true_index in range(len(COUNTRIES)):
        common_pool = sorted(
            profile_id
            for profile_id in common_ids
            if attributes[profile_id][0] == true_index
        )
        remainder_pool = sorted(
            profile_id
            for profile_id in full_ids - common_ids
            if attributes[profile_id][0] == true_index
        )
        common_draw = draw_ids(common_pool)
        remainder_draw = draw_ids(remainder_pool)
        for item in view.item_ids:
            source = common_draw if profile_sets[view.item_ids.index(item)] == common_ids else common_draw + remainder_draw
            country_samples[item][true_index] = np.asarray(
                [item_indexes[item][profile_id] for profile_id in source], dtype=int
            )
        for label_index in range(len(COUNTRIES)):
            common_cell = [
                profile_id
                for profile_id in common_pool
                if attributes[profile_id][1] == label_index
            ]
            remainder_cell = [
                profile_id
                for profile_id in remainder_pool
                if attributes[profile_id][1] == label_index
            ]
            if not common_cell or (remainder_pool and not remainder_cell):
                raise ValueError("Nested displayed-label bootstrap cell is empty")
            common_cell_draw = draw_ids(common_cell)
            remainder_cell_draw = draw_ids(remainder_cell)
            for item_index, item in enumerate(view.item_ids):
                source = (
                    common_cell_draw
                    if profile_sets[item_index] == common_ids
                    else common_cell_draw + remainder_cell_draw
                )
                label_samples[item][(true_index, label_index)] = np.asarray(
                    [item_indexes[item][profile_id] for profile_id in source], dtype=int
                )
    return EvaluationSamples(country_samples, label_samples)


BOOTSTRAP_METRICS = (
    "m0_anchor_tv",
    "m0_anchor_l2",
    "mo_tv",
    "mr_tv",
    "mo_tv_minus_mr_tv",
    "mo_regret",
    "mr_regret",
    "mo_regret_minus_mr_regret",
    "mo_tv_mismatch",
    "mo_regret_mismatch",
    "direction_o_pooled",
    "direction_r_pooled",
    "direction_o_minus_direction_r_pooled",
    "direction_o_mismatch_pooled",
    "m0_brier",
    "utility_v",
)


def bootstrap_view(
    view: AnalysisView,
    anchors: dict[str, dict[str, list[float]]],
    resampler: Any,
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    observed = calculate_metrics(view, anchors)
    samples = {
        metric: np.empty(
            (replicates, *observed[metric].shape), dtype=float
        )
        for metric in BOOTSTRAP_METRICS
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        evaluation_samples = stratified_samples(view, rng)
        sampled_anchors = resampler(rng)
        values = calculate_metrics(
            view,
            sampled_anchors,
            samples=evaluation_samples,
            include_nearest=False,
            include_js=False,
        )
        for metric in BOOTSTRAP_METRICS:
            samples[metric][replicate] = values[metric]
    return observed, samples


def percentile_interval(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    return low, high


def centred_bootstrap_p(estimate: np.ndarray, samples: np.ndarray) -> np.ndarray:
    centred = samples - estimate[None, ...]
    return (1.0 + np.sum(np.abs(centred) >= np.abs(estimate)[None, ...], axis=0)) / (
        len(samples) + 1.0
    )


def max_t_interaction_test(
    estimate: np.ndarray, samples: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    interaction = two_way_interaction(estimate)
    interaction_samples = np.asarray(
        [two_way_interaction(value) for value in samples]
    )
    standard_error = np.std(interaction_samples, axis=0, ddof=1)
    standard_error = np.maximum(standard_error, np.finfo(float).eps)
    observed_t = np.abs(interaction / standard_error)
    null_t = np.abs((interaction_samples - interaction[None, ...]) / standard_error)
    max_null = np.max(null_t, axis=(1, 2))
    adjusted = (
        1.0
        + np.sum(max_null[:, None, None] >= observed_t[None, ...], axis=0)
    ) / (len(samples) + 1.0)
    observed_max = float(np.max(observed_t))
    global_p = float((1.0 + np.count_nonzero(max_null >= observed_max)) / (len(samples) + 1.0))
    return (
        {
            "statistic": observed_max,
            "p_value": global_p,
            "resampling_method": "centred_cluster_bootstrap_max_t",
        },
        adjusted,
        standard_error,
    )


def summarize_bootstrap(
    observed: dict[str, np.ndarray], samples: dict[str, np.ndarray]
) -> dict[str, Any]:
    summary: dict[str, Any] = {"metrics": {}, "global_tests": {}}
    for metric in BOOTSTRAP_METRICS:
        estimate = observed[metric]
        sample = samples[metric]
        low, high = percentile_interval(sample)
        interaction = two_way_interaction(estimate)
        interaction_samples = np.asarray(
            [two_way_interaction(value) for value in sample]
        )
        interaction_low, interaction_high = percentile_interval(interaction_samples)
        p_values = centred_bootstrap_p(interaction, interaction_samples)
        global_test, max_t_p, interaction_se = max_t_interaction_test(
            estimate, sample
        )
        summary["metrics"][metric] = {
            "estimate": estimate.tolist(),
            "ci_low": low.tolist(),
            "ci_high": high.tolist(),
            "provider_centered": profile_centered(estimate).tolist(),
            "interaction": interaction.tolist(),
            "interaction_ci_low": interaction_low.tolist(),
            "interaction_ci_high": interaction_high.tolist(),
            "interaction_p": p_values.tolist(),
            "interaction_max_t_p": max_t_p.tolist(),
            "interaction_se": interaction_se.tolist(),
        }
        summary["global_tests"][f"{metric}_model_by_country_interaction"] = {
            **global_test,
            "family": "model_by_country_interaction",
            "metric": metric,
        }

    provider_tv = np.mean(observed["mo_tv"], axis=1)
    provider_tv_samples = np.mean(samples["mo_tv"], axis=2)
    tv_low, tv_high = percentile_interval(provider_tv_samples)
    summary["provider_mo_tv"] = {
        provider: {
            "estimate": float(provider_tv[index]),
            "ci_low": float(tv_low[index]),
            "ci_high": float(tv_high[index]),
        }
        for index, provider in enumerate(PROVIDERS)
    }

    primary_names = [
        f"{metric}_model_by_country_interaction" for metric in PRIMARY_METRICS
    ]
    adjusted = holm_adjust(
        {
            name: summary["global_tests"][name]["p_value"]
            for name in primary_names
        }
    )
    for name in primary_names:
        summary["global_tests"][name]["holm_p_across_four_primary_families"] = adjusted[
            name
        ]
    return summary


def descriptive_summary(metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        metric: {
            "estimate": value.tolist(),
            "provider_centered": profile_centered(value).tolist()
            if value.ndim == 2
            else None,
            "interaction": two_way_interaction(value).tolist()
            if value.ndim == 2
            else None,
        }
        for metric, value in metrics.items()
        if metric not in {"nearest_confusion", "nearest_raw_counts"}
    }


def leave_one_item_out(
    view: AnalysisView, anchors: dict[str, dict[str, list[float]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for omitted in (None, *view.item_ids):
        kept = tuple(item for item in view.item_ids if item != omitted)
        subset = AnalysisView(
            view.name,
            kept,
            {item: view.panels[item] for item in kept},
            view.role,
        )
        values = calculate_metrics(subset, anchors)
        for provider_index, provider in enumerate(PROVIDERS):
            distance = values["m0_anchor_l2"][provider_index]
            top_index = int(np.argmin(distance))
            france = COUNTRIES.index("FR")
            cue_tv = values["mo_tv"][provider_index]
            direction = values["direction_o_pooled"][provider_index]
            rows.append(
                {
                    "view": view.name,
                    "omitted_item": omitted or "none",
                    "provider": provider,
                    "top_default_anchor": COUNTRIES[top_index],
                    "france_default_rank": int(np.argsort(distance).tolist().index(france) + 1),
                    "france_centered_affinity": float(
                        (-distance[france]) - np.mean(-distance)
                    ),
                    "mo_tv_france": float(cue_tv[france]),
                    "mo_tv_france_minus_other_labels": float(
                        cue_tv[france] - np.mean(np.delete(cue_tv, france))
                    ),
                    "direction_o_pooled_france": float(direction[france]),
                    "direction_o_pooled_france_minus_other_labels": float(
                        direction[france] - np.mean(np.delete(direction, france))
                    ),
                }
            )
    return rows


def matrix_rows(
    *,
    view: AnalysisView,
    metric: str,
    estimate: np.ndarray,
    bootstrap: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    domain = (
        "default_output_affinity"
        if metric.startswith("m0_anchor")
        else "random_label_response"
        if metric.startswith(("mo_", "mr_", "direction_"))
        else "true_country_prediction"
    )
    output = []
    for provider_index, provider in enumerate(PROVIDERS):
        for country_index, country in enumerate(COUNTRIES):
            row: dict[str, Any] = {
                "view": view.name,
                "analysis_role": view.role,
                "domain": domain,
                "metric": metric,
                "provider": provider,
                "model": OFFICIAL_MODELS[provider],
                "country_code": country,
                "country_label": COUNTRY_LABELS[country],
                "estimate": float(estimate[provider_index, country_index]),
                "ci_low": "",
                "ci_high": "",
                "provider_centered_estimate": float(
                    profile_centered(estimate)[provider_index, country_index]
                ),
                "interaction_estimate": float(
                    two_way_interaction(estimate)[provider_index, country_index]
                ),
                "interaction_ci_low": "",
                "interaction_ci_high": "",
                "interaction_p": "",
                "interaction_max_t_p": "",
            }
            if bootstrap is not None and metric in bootstrap["metrics"]:
                value = bootstrap["metrics"][metric]
                row.update(
                    {
                        "ci_low": value["ci_low"][provider_index][country_index],
                        "ci_high": value["ci_high"][provider_index][country_index],
                        "interaction_ci_low": value["interaction_ci_low"][provider_index][country_index],
                        "interaction_ci_high": value["interaction_ci_high"][provider_index][country_index],
                        "interaction_p": value["interaction_p"][provider_index][country_index],
                        "interaction_max_t_p": value["interaction_max_t_p"][provider_index][country_index],
                    }
                )
            output.append(row)
    return output


def correlation_rows(
    view: AnalysisView, metrics: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        interaction = two_way_interaction(metrics[metric])
        provider_centred = profile_centered(metrics[metric])
        for left_index, left in enumerate(PROVIDERS):
            for right_index in range(left_index + 1, len(PROVIDERS)):
                right = PROVIDERS[right_index]
                rows.append(
                    {
                        "view": view.name,
                        "metric": metric,
                        "provider_a": left,
                        "provider_b": right,
                        "country_profile_correlation": float(
                            np.corrcoef(
                                provider_centred[left_index],
                                provider_centred[right_index],
                            )[0, 1]
                        ),
                        "interaction_profile_correlation": float(
                            np.corrcoef(
                                interaction[left_index], interaction[right_index]
                            )[0, 1]
                        ),
                    }
                )
    return rows


def france_contrast_rows(
    view: AnalysisView,
    observed: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    france = COUNTRIES.index("FR")
    for metric in (
        "m0_anchor_l2",
        "m0_anchor_tv",
        "mo_tv",
        "mo_regret",
        "direction_o_pooled",
    ):
        estimate = observed[metric]
        bootstrap = samples[metric]
        for provider_index, provider in enumerate(PROVIDERS):
            others = [index for index in range(len(COUNTRIES)) if index != france]
            difference = float(
                estimate[provider_index, france]
                - np.mean(estimate[provider_index, others])
            )
            draws = (
                bootstrap[:, provider_index, france]
                - np.mean(bootstrap[:, provider_index, others], axis=1)
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "view": view.name,
                    "provider": provider,
                    "metric": metric,
                    "contrast": "France_minus_mean_other_countries",
                    "estimate": difference,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "positive_means": (
                        "farther_from_France" if metric.startswith("m0_anchor")
                        else "larger_France_label_response"
                    ),
                    "status": "post_hoc_exploratory",
                }
            )
    us = COUNTRIES.index("US")
    metric = "m0_anchor_l2"
    estimate = observed[metric]
    bootstrap = samples[metric]
    for provider_index, provider in enumerate(PROVIDERS):
        difference = float(
            estimate[provider_index, france] - estimate[provider_index, us]
        )
        draws = (
            bootstrap[:, provider_index, france]
            - bootstrap[:, provider_index, us]
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "view": view.name,
                "provider": provider,
                "metric": metric,
                "contrast": "France_minus_United_States_distance",
                "estimate": difference,
                "ci_low": float(low),
                "ci_high": float(high),
                "positive_means": "farther_from_France_than_United_States",
                "status": "post_hoc_exploratory",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2026072801)
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    views, source_audit, resampler, anchors = build_views()
    point = {
        name: calculate_metrics(view, anchors)
        for name, view in views.items()
    }
    inferential: dict[str, Any] = {}
    bootstrap_arrays: dict[str, dict[str, np.ndarray]] = {}
    inferential_views = (
        "clean_hybrid_all20",
        "clean_common72_all20",
        "clean_unaffected504_12",
    )
    for offset, name in enumerate(inferential_views):
        observed, samples = bootstrap_view(
            views[name],
            anchors,
            resampler,
            replicates=args.bootstrap_replicates,
            seed=args.seed + offset * 1000,
        )
        point[name] = observed
        bootstrap_arrays[name] = samples
        inferential[name] = summarize_bootstrap(observed, samples)

    loo_rows = []
    for name in ("clean_common72_all20", "clean_unaffected504_12"):
        loo_rows.extend(leave_one_item_out(views[name], anchors))

    correlation_output = []
    contrast_output = []
    for name in inferential_views:
        correlation_output.extend(correlation_rows(views[name], point[name]))
        contrast_output.extend(
            france_contrast_rows(
                views[name], point[name], bootstrap_arrays[name]
            )
        )

    metrics_rows: list[dict[str, Any]] = []
    exported_metrics = (
        "m0_anchor_tv",
        "m0_anchor_l2",
        "m0_anchor_js",
        "mo_tv",
        "mr_tv",
        "mo_tv_minus_mr_tv",
        "mo_regret",
        "mr_regret",
        "mo_regret_minus_mr_regret",
        "mo_tv_mismatch",
        "mo_regret_mismatch",
        "direction_o_pooled",
        "direction_r_pooled",
        "direction_o_minus_direction_r_pooled",
        "direction_o_mismatch_pooled",
        "m0_brier",
        "utility_v",
    )
    for name, view in views.items():
        for metric in exported_metrics:
            metrics_rows.extend(
                matrix_rows(
                    view=view,
                    metric=metric,
                    estimate=point[name][metric],
                    bootstrap=inferential.get(name),
                )
            )

    test_rows = []
    for name, value in inferential.items():
        for test_name, test in value["global_tests"].items():
            test_rows.append(
                {
                    "view": name,
                    "test": test_name,
                    "family": test["family"],
                    "metric": test["metric"],
                    "statistic": test["statistic"],
                    "p_value": test["p_value"],
                    "holm_p_across_four_primary_families": test.get(
                        "holm_p_across_four_primary_families", ""
                    ),
                    "resampling_method": test["resampling_method"],
                    "analysis_role": views[name].role,
                }
            )

    resemblance_rows = []
    nearest_diagnostic_rows = []
    for name in inferential_views:
        observed = point[name]
        for provider_index, provider in enumerate(PROVIDERS):
            aggregate_distance = observed["m0_anchor_l2"][provider_index]
            ranks = np.argsort(np.argsort(aggregate_distance)) + 1
            for country_index, country in enumerate(COUNTRIES):
                resemblance_rows.append(
                    {
                        "view": name,
                        "provider": provider,
                        "model": OFFICIAL_MODELS[provider],
                        "country_anchor_code": country,
                        "country_anchor_label": COUNTRY_LABELS[country],
                        "aggregate_squared_l2_distance": float(
                            aggregate_distance[country_index]
                        ),
                        "aggregate_distance_rank": int(ranks[country_index]),
                        "nearest_anchor": bool(ranks[country_index] == 1),
                        "interpretation": "derived aggregate resemblance to a human-country response anchor; hard nearest-anchor assignment is descriptive and is not a model-emitted country label",
                    }
                )
        if aligned_profiles(views[name]):
            for provider_index, provider in enumerate(PROVIDERS):
                for country_index, country in enumerate(COUNTRIES):
                    for true_index, true_country in enumerate(COUNTRIES):
                        nearest_diagnostic_rows.append(
                            {
                                "view": name,
                                "provider": provider,
                                "model": OFFICIAL_MODELS[provider],
                                "true_survey_country": true_country,
                                "nearest_anchor_code": country,
                                "equal_weight_within_true_country_share": float(
                                    observed["nearest_confusion"][provider_index, true_index, country_index]
                                ),
                                "interpretation": "profile-level hard nearest squared-L2 human-country anchor diagnostic; not a model-emitted country label",
                            }
                        )

    audit_rows = []
    for item in views["clean_hybrid_all20"].item_ids:
        panel = views["clean_hybrid_all20"].panels[item]
        audit_rows.append(
            {
                "item_id": item,
                "source": panel.source,
                "profiles": len(panel.profile_ids),
                "countries": len(set(panel.true_country.tolist())),
                "models": len(PROVIDERS),
                "conditions": len(CONDITIONS),
                "item_vectors_used": len(panel.profile_ids) * len(PROVIDERS) * len(CONDITIONS),
                "old_affected_vectors_used": 0,
                "question_quality_status": "approved_no_unresolved_rejections",
            }
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_post_review_exploratory_analysis",
        "question": "Do models exhibit different country-related output affinities or cue responses?",
        "construct_guardrail": (
            "The endpoints output survey-answer probability vectors, not country labels. "
            "Nearest-country results measure resemblance to external human-country anchors; "
            "they do not identify nationality, internal belief, or social bias by themselves."
        ),
        "data_contract": (
            "Original 504-profile results for 12 already-correct items plus corrected-wording "
            "72-profile replacements for all eight affected checklist items; no affected old-wording "
            "forecast enters any clean view. The 20-item hybrid estimand weights items equally, "
            "then true countries equally within item; it does not treat every item as having 504 profiles."
        ),
        "analysis_version": "model-country-preference-v1",
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
        "source_audit": source_audit,
        "models": OFFICIAL_MODELS,
        "countries": COUNTRY_LABELS,
        "views": {
            name: {
                "role": view.role,
                "items": list(view.item_ids),
                "item_profile_counts": {
                    item: len(view.panels[item].profile_ids) for item in view.item_ids
                },
                "descriptive": descriptive_summary(point[name]),
                "inference": inferential.get(name),
            }
            for name, view in views.items()
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "profile_resampling": (
                "nested replacement-panel cluster bootstrap: corrected-subset and remainder profiles "
                "are resampled separately within true country; displayed-label analyses use the "
                "corresponding true-country x displayed-label strata; draws are shared across models/items"
            ),
            "scope": (
                "conditional on the fixed 20 survey items, five model endpoints, six-country protocol, "
                "and fixed same-record 72-profile correction subset; this is not population-level or "
                "prospectively confirmatory inference"
            ),
            "reference_anchors_reestimated_each_replicate": True,
            "interval": "percentile 95%",
            "cell_tests": (
                "exploratory centred cluster-bootstrap max-T adjusted over 30 model-country cells; "
                "not a randomisation test or finite-sample exact test"
            ),
            "global_tests": (
                "exploratory centred cluster-bootstrap max-T interaction tests; four designated primary "
                "metric families Holm-adjusted within each view only"
            ),
            "multiplicity_boundary": (
                "the Holm family excludes comparisons across the three dependent views, secondary "
                "metrics, leave-one-item-out diagnostics, and targeted France contrasts"
            ),
            "view_dependence": (
                "clean_common72_all20 and clean_unaffected504_12 are non-independent sensitivity views "
                "of clean_hybrid_all20, not replication samples"
            ),
        },
        "leave_one_item_out": loo_rows,
        "input_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in (
                ORIGINAL_RESULTS,
                ORIGINAL_PROFILES,
                ORIGINAL_TASK,
                ORIGINAL_PROMPT,
                CORRECTED_RESULTS,
                CORRECTED_PROFILES,
                CORRECTED_TASK,
                CORRECTED_PROMPT,
                ANALYSIS_MANIFEST,
                AMENDMENT,
                HUMAN_SIGNOFF,
                PANEL,
                METADATA,
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUT_METRICS, metrics_rows)
    write_csv(OUT_TESTS, test_rows)
    write_csv(OUT_RESEMBLANCE, resemblance_rows)
    write_csv(OUT_NEAREST_DIAGNOSTIC, nearest_diagnostic_rows)
    write_csv(OUT_LOO, loo_rows)
    write_csv(OUT_AUDIT, audit_rows)
    write_csv(OUT_CORRELATIONS, correlation_output)
    write_csv(OUT_CONTRASTS, contrast_output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "hybrid_items": len(views["clean_hybrid_all20"].item_ids),
                "bootstrap": args.bootstrap_replicates,
                "outputs": 9,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
