"""Comprehensive aggregate metrics for the frozen Gate C measurement pilot.

This module deliberately contains no API clients. It consumes paired item-level
rows and returns survey-weighted estimates under equal country, item, and
provider aggregation. Profile bootstrap draws are shared across every metric;
directional metrics additionally rebuild the reference anchors on every draw.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from api_pilot.strict_inference import (
    AnchorBootstrap,
    AnchorMap,
    DirectionObservation,
    ScalarObservation,
    _DirectionPanel,
    _ScalarPanel,
)


@dataclass(frozen=True)
class Interval:
    estimate: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


def _interval(estimate: float, samples: np.ndarray, confidence: float) -> Interval:
    if samples.ndim != 1 or samples.size < 1 or not np.all(np.isfinite(samples)):
        raise ValueError("Bootstrap samples must be a non-empty finite vector")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    alpha = 1.0 - confidence
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(float(estimate), float(low), float(high), int(samples.size))


def _summarize_provider_matrix(
    provider_names: Sequence[str],
    observed: np.ndarray,
    samples: np.ndarray,
    included: Sequence[str],
    *,
    confidence: float,
) -> dict[str, Any]:
    """Summarize provider estimates, equal-provider macro, and leave-one-out."""
    names = tuple(provider_names)
    if observed.shape != (len(names),):
        raise ValueError("Observed provider vector has the wrong shape")
    if samples.ndim != 2 or samples.shape[1] != len(names):
        raise ValueError("Bootstrap provider matrix has the wrong shape")
    indexes = [names.index(name) for name in included]
    macro_observed = float(np.mean(observed[indexes]))
    macro_samples = np.mean(samples[:, indexes], axis=1)
    provider_results = {
        name: asdict(_interval(observed[names.index(name)], samples[:, names.index(name)], confidence))
        for name in included
    }
    leave_one_out = {}
    if len(indexes) > 1:
        for omitted in included:
            retained = [names.index(name) for name in included if name != omitted]
            leave_one_out[omitted] = float(np.mean(observed[retained]))
    return {
        "macro": asdict(_interval(macro_observed, macro_samples, confidence)),
        "by_provider": provider_results,
        "leave_one_provider_out_point_estimates": leave_one_out,
    }


def comprehensive_bootstrap(
    direction_rows_by_stratum: Mapping[str, Sequence[DirectionObservation | Mapping[str, Any]]],
    scalar_rows_by_stratum_metric: Mapping[
        str, Mapping[str, Sequence[ScalarObservation | Mapping[str, Any]]]
    ],
    anchors: AnchorMap,
    *,
    anchor_bootstrap: AnchorBootstrap,
    provider_sets: Mapping[str, Sequence[str]],
    expected_providers: Sequence[str],
    expected_countries: Sequence[str],
    items_by_stratum: Mapping[str, Sequence[str]],
    n_bootstrap: int = 10_000,
    seed: int = 2026072517,
    confidence: float = 0.95,
    pair_ids_by_profile: Mapping[str, str] | None = None,
    pair_cluster_metrics: Sequence[str] = (),
) -> dict[str, Any]:
    """Run one shared profile/reference bootstrap for all reported metrics."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not direction_rows_by_stratum or not scalar_rows_by_stratum_metric:
        raise ValueError("Direction and scalar panels are required")
    providers = tuple(sorted(expected_providers))
    countries = tuple(sorted(expected_countries))
    if any(not values for values in provider_sets.values()):
        raise ValueError("Provider sets must be non-empty")
    if any(not set(values).issubset(providers) for values in provider_sets.values()):
        raise ValueError("Provider set contains an unexpected provider")

    direction_panels = {
        stratum: _DirectionPanel(
            rows,
            {item: anchors[item] for item in items_by_stratum[stratum]},
            expected_providers=providers,
            expected_countries=countries,
            expected_items=items_by_stratum[stratum],
        )
        for stratum, rows in direction_rows_by_stratum.items()
    }
    scalar_panels = {}
    for stratum, metrics in scalar_rows_by_stratum_metric.items():
        stratum_items = set(items_by_stratum[stratum])
        for metric, rows in metrics.items():
            observed_items = sorted({
                str(row.item_id if isinstance(row, ScalarObservation) else row["item_id"])
                for row in rows
            })
            if not observed_items or not set(observed_items).issubset(stratum_items):
                raise ValueError(
                    f"Scalar panel {stratum}/{metric} has invalid target items"
                )
            scalar_panels[(stratum, metric)] = _ScalarPanel(
                rows,
                expected_providers=providers,
                expected_countries=countries,
                expected_items=observed_items,
            )
    first_panel = next(iter(direction_panels.values()))
    profile_ids = first_panel.profile_ids
    for panel in [*direction_panels.values(), *scalar_panels.values()]:
        if panel.profile_ids != profile_ids:
            raise ValueError("All panels must share the same country/profile clusters")
    pair_metric_set = set(pair_cluster_metrics)
    unknown_pair_metrics = pair_metric_set.difference(
        metric for _, metric in scalar_panels
    )
    if unknown_pair_metrics:
        raise ValueError(
            f"Unknown pair-cluster metrics: {sorted(unknown_pair_metrics)}"
        )
    pair_groups: dict[str, tuple[np.ndarray, ...]] = {}
    if pair_metric_set:
        if pair_ids_by_profile is None:
            raise ValueError("Pair-cluster metrics require pair_ids_by_profile")
        observed_profiles = {
            profile_id for values in profile_ids.values() for profile_id in values
        }
        if set(pair_ids_by_profile) != observed_profiles:
            raise ValueError("Pair-ID map does not match the complete profile panel")
        for country in countries:
            grouped: dict[str, list[int]] = {}
            for index, profile_id in enumerate(profile_ids[country]):
                grouped.setdefault(pair_ids_by_profile[profile_id], []).append(index)
            if not grouped or any(len(indexes) != 2 for indexes in grouped.values()):
                raise ValueError(f"Country {country} does not contain reciprocal pairs")
            pair_groups[country] = tuple(
                np.asarray(grouped[pair_id], dtype=int)
                for pair_id in sorted(grouped)
            )

    direction_observed = {
        stratum: panel.estimate_by_provider()
        for stratum, panel in direction_panels.items()
    }
    scalar_observed = {
        key: panel.estimate_by_provider() for key, panel in scalar_panels.items()
    }
    direction_samples = {
        stratum: np.empty((n_bootstrap, len(providers), 3), dtype=float)
        for stratum in direction_panels
    }
    scalar_samples = {
        key: np.empty((n_bootstrap, len(providers)), dtype=float)
        for key in scalar_panels
    }

    rng = np.random.default_rng(seed)
    for replicate in range(n_bootstrap):
        sampled = {
            country: rng.integers(0, len(profile_ids[country]), len(profile_ids[country]))
            for country in countries
        }
        sampled_pairs = None
        if pair_metric_set:
            sampled_pairs = {}
            for country in countries:
                groups = pair_groups[country]
                draws = rng.integers(0, len(groups), len(groups))
                sampled_pairs[country] = np.concatenate(
                    [groups[int(draw)] for draw in draws]
                )
        sampled_anchors = anchor_bootstrap(rng)
        for stratum, panel in direction_panels.items():
            projections = panel.projection_arrays(
                {item: sampled_anchors[item] for item in items_by_stratum[stratum]}
            )
            direction_samples[stratum][replicate] = panel.estimate_by_provider(
                projections=projections, sample_indexes=sampled
            )
        for key, panel in scalar_panels.items():
            sample_indexes = (
                sampled_pairs if key[1] in pair_metric_set else sampled
            )
            scalar_samples[key][replicate] = panel.estimate_by_provider(sample_indexes)

    direction_names = ("beta_o", "beta_r", "beta_o_minus_beta_r")
    output: dict[str, Any] = {
        "bootstrap": {
            "replicates": n_bootstrap,
            "seed": seed,
            "confidence": confidence,
            "profile_clusters_resampled_within_true_country": True,
            "reference_anchors_reestimated_each_replicate": True,
            "pair_clusters_resampled_for_metrics": sorted(pair_metric_set),
            "reciprocal_pair_cluster_count": (
                sum(len(values) for values in pair_groups.values())
                if pair_metric_set else 0
            ),
        },
        "provider_sets": {},
    }
    for set_name, included in provider_sets.items():
        result: dict[str, Any] = {"strata": {}, "interactions": {}}
        for stratum in direction_panels:
            direction_result = {}
            for column, name in enumerate(direction_names):
                direction_result[name] = _summarize_provider_matrix(
                    providers,
                    direction_observed[stratum][:, column],
                    direction_samples[stratum][:, :, column],
                    included,
                    confidence=confidence,
                )
            scalar_result = {
                metric: _summarize_provider_matrix(
                    providers,
                    scalar_observed[(stratum, metric)],
                    scalar_samples[(stratum, metric)],
                    included,
                    confidence=confidence,
                )
                for metric in scalar_rows_by_stratum_metric[stratum]
            }
            result["strata"][stratum] = {
                "items": list(items_by_stratum[stratum]),
                "direction": direction_result,
                "prediction": scalar_result,
            }

        if "high_country_value" in direction_panels and "equivalence_controls" in direction_panels:
            interaction: dict[str, Any] = {"direction": {}, "prediction": {}}
            high = "high_country_value"
            control = "equivalence_controls"
            for column, name in enumerate(direction_names):
                interaction["direction"][name] = _summarize_provider_matrix(
                    providers,
                    direction_observed[high][:, column] - direction_observed[control][:, column],
                    direction_samples[high][:, :, column] - direction_samples[control][:, :, column],
                    included,
                    confidence=confidence,
                )
            common_metrics = sorted(
                set(scalar_rows_by_stratum_metric[high])
                & set(scalar_rows_by_stratum_metric[control])
            )
            for metric in common_metrics:
                interaction["prediction"][metric] = _summarize_provider_matrix(
                    providers,
                    scalar_observed[(high, metric)] - scalar_observed[(control, metric)],
                    scalar_samples[(high, metric)] - scalar_samples[(control, metric)],
                    included,
                    confidence=confidence,
                )
            result["interactions"]["high_minus_equivalence"] = interaction
        output["provider_sets"][set_name] = result
    return output
