"""Strict, design-aligned inference for the country-metadata experiment.

The module works on item-level paired contrasts. It deliberately does not read
LLM logs or choose an analysis stratum. Callers must construct the frozen rows
and pass only the preregistered items. All estimators use survey weights within
country, then give equal weight to countries, items, and model providers.

The confirmatory entry point requires a reference-anchor bootstrap callback.
This prevents a fixed-reference analysis from being mistaken for the planned
confirmatory analysis, which must include uncertainty in the human anchors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_PERMUTATIONS = 100_000
PROBABILITY_TOLERANCE = 1e-8
ZERO_TOLERANCE = 1e-12

AnchorMap = Mapping[str, Mapping[str, Sequence[float]]]
AnchorBootstrap = Callable[[np.random.Generator], AnchorMap]


@dataclass(frozen=True)
class ScalarObservation:
    provider: str
    profile_id: str
    country: str
    item_id: str
    weight: float
    value: float


@dataclass(frozen=True)
class DirectionObservation:
    provider: str
    profile_id: str
    country: str
    item_id: str
    weight: float
    displayed_label: str
    mo_delta: tuple[float, ...]
    mr_delta: tuple[float, ...]


@dataclass(frozen=True)
class ReferenceObservation:
    country: str
    profile_id: str
    item_id: str
    answer_index: int
    n_options: int
    weight: float


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    p_greater: float | None
    n_bootstrap: int
    anchors_reestimated: bool = False
    scientific_null: float | None = None
    p_greater_scientific_null: float | None = None


@dataclass(frozen=True)
class DirectionBootstrapResult:
    beta_o: BootstrapResult
    beta_r: BootstrapResult
    beta_diff: BootstrapResult


@dataclass(frozen=True)
class PermutationResult:
    beta_o: float
    beta_r: float
    beta_diff: float
    p_beta_o_greater: float
    p_beta_r_greater: float
    p_beta_diff_greater: float
    h1_iut_p: float
    n_permutations: int
    monte_carlo_se_h1_upper_bound: float
    null_hypothesis: str = (
        "Fisher sharp null that displayed-label assignment changes no profile's "
        "forecast vector"
    )
    studentized: bool = False


@dataclass(frozen=True)
class StrictInferenceResult:
    direction_bootstrap: DirectionBootstrapResult
    assignment_permutation: PermutationResult
    utility_h2: BootstrapResult
    raw_hypothesis_p: dict[str, float]
    holm_adjusted_p: dict[str, float]
    raw_sesoi_exceedance_p: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"Missing required field: {name}")
    return mapping[name]


def _as_scalar(row: ScalarObservation | Mapping[str, Any]) -> ScalarObservation:
    if isinstance(row, ScalarObservation):
        return row
    return ScalarObservation(
        provider=str(_required(row, "provider")),
        profile_id=str(_required(row, "profile_id")),
        country=str(row.get("country", row.get("true_country_code", ""))),
        item_id=str(_required(row, "item_id")),
        weight=float(row.get("weight", row.get("analysis_weight", np.nan))),
        value=float(_required(row, "value")),
    )


def _as_direction(
    row: DirectionObservation | Mapping[str, Any],
) -> DirectionObservation:
    if isinstance(row, DirectionObservation):
        return row
    return DirectionObservation(
        provider=str(_required(row, "provider")),
        profile_id=str(_required(row, "profile_id")),
        country=str(row.get("country", row.get("true_country_code", ""))),
        item_id=str(_required(row, "item_id")),
        weight=float(row.get("weight", row.get("analysis_weight", np.nan))),
        displayed_label=str(_required(row, "displayed_label")),
        mo_delta=tuple(float(value) for value in _required(row, "mo_delta")),
        mr_delta=tuple(float(value) for value in _required(row, "mr_delta")),
    )


def _check_expected(
    observed: Sequence[str], expected: Sequence[str] | None, name: str
) -> tuple[str, ...]:
    values = tuple(sorted(set(observed)))
    if not values or any(not value for value in values):
        raise ValueError(f"No valid {name} values")
    if expected is not None and set(values) != set(expected):
        raise ValueError(
            f"{name} mismatch: observed={list(values)}, expected={sorted(expected)}"
        )
    return values


class _ScalarPanel:
    def __init__(
        self,
        rows: Iterable[ScalarObservation | Mapping[str, Any]],
        *,
        expected_providers: Sequence[str] | None = None,
        expected_countries: Sequence[str] | None = None,
        expected_items: Sequence[str] | None = None,
    ) -> None:
        observations = [_as_scalar(row) for row in rows]
        if not observations:
            raise ValueError("Scalar panel is empty")
        self.providers = _check_expected(
            [row.provider for row in observations], expected_providers, "providers"
        )
        self.countries = _check_expected(
            [row.country for row in observations], expected_countries, "countries"
        )
        self.items = _check_expected(
            [row.item_id for row in observations], expected_items, "items"
        )
        records: dict[tuple[str, str, str, str], ScalarObservation] = {}
        for row in observations:
            key = (row.provider, row.country, row.profile_id, row.item_id)
            if key in records:
                raise ValueError(f"Duplicate scalar observation: {key}")
            if not np.isfinite(row.weight) or row.weight <= 0:
                raise ValueError(f"Invalid survey weight for {key}: {row.weight}")
            if not np.isfinite(row.value):
                raise ValueError(f"Non-finite scalar value for {key}")
            records[key] = row

        self.profile_ids: dict[str, tuple[str, ...]] = {}
        self.values: dict[str, np.ndarray] = {}
        self.weights: dict[str, np.ndarray] = {}
        for country in self.countries:
            provider_profiles = {
                provider: {
                    row.profile_id
                    for row in observations
                    if row.provider == provider and row.country == country
                }
                for provider in self.providers
            }
            first = provider_profiles[self.providers[0]]
            if not first or any(ids != first for ids in provider_profiles.values()):
                raise ValueError(
                    f"Providers do not share the same profiles in country {country}"
                )
            profiles = tuple(sorted(first))
            self.profile_ids[country] = profiles
            matrix = np.empty(
                (len(self.providers), len(self.items), len(profiles)), dtype=float
            )
            weights = np.empty(len(profiles), dtype=float)
            for p_index, provider in enumerate(self.providers):
                for i_index, item in enumerate(self.items):
                    for r_index, profile_id in enumerate(profiles):
                        key = (provider, country, profile_id, item)
                        if key not in records:
                            raise ValueError(f"Missing scalar observation: {key}")
                        row = records[key]
                        matrix[p_index, i_index, r_index] = row.value
                        if p_index == 0 and i_index == 0:
                            weights[r_index] = row.weight
                        elif not np.isclose(weights[r_index], row.weight):
                            raise ValueError(
                                "Survey weight differs across provider/item for "
                                f"profile {country}/{profile_id}"
                            )
            self.values[country] = matrix
            self.weights[country] = weights
        expected_count = sum(
            len(self.providers) * len(self.items) * len(self.profile_ids[country])
            for country in self.countries
        )
        if len(records) != expected_count:
            raise ValueError("Scalar panel contains observations outside its full panel")

    def estimate_by_provider(
        self, sample_indexes: Mapping[str, np.ndarray] | None = None
    ) -> np.ndarray:
        country_means = []
        for country in self.countries:
            indexes = (
                np.arange(len(self.profile_ids[country]))
                if sample_indexes is None
                else np.asarray(sample_indexes[country], dtype=int)
            )
            values = self.values[country][:, :, indexes]
            weights = self.weights[country][indexes]
            country_means.append(
                np.sum(values * weights[None, None, :], axis=2) / weights.sum()
            )
        # Countries and items receive equal weight within each provider.
        return np.mean(np.asarray(country_means, dtype=float), axis=(0, 2))

    def estimate(self, sample_indexes: Mapping[str, np.ndarray] | None = None) -> float:
        # Providers receive equal weight in the macro estimate.
        return float(np.mean(self.estimate_by_provider(sample_indexes)))


def scalar_macro(
    rows: Iterable[ScalarObservation | Mapping[str, Any]],
    **panel_expectations: Any,
) -> float:
    """Survey-weighted, equal-country/item/provider scalar macro."""
    return _ScalarPanel(rows, **panel_expectations).estimate()


def paired_profile_bootstrap(
    rows: Iterable[ScalarObservation | Mapping[str, Any]],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    seed: int = 20260724,
    confidence: float = 0.95,
    scientific_null: float | None = None,
    **panel_expectations: Any,
) -> BootstrapResult:
    """Bootstrap profiles within country using shared draws for every model/item."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    panel = _ScalarPanel(rows, **panel_expectations)
    observed = panel.estimate()
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = {
            country: rng.integers(0, len(panel.profile_ids[country]), len(panel.profile_ids[country]))
            for country in panel.countries
        }
        samples[index] = panel.estimate(sampled)
    alpha = 1.0 - confidence
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    # One-sided centred bootstrap test of H0: theta <= 0 against theta > 0.
    p_greater = (1.0 + np.count_nonzero(samples - observed >= observed)) / (
        n_bootstrap + 1.0
    )
    p_scientific = None
    if scientific_null is not None:
        if not np.isfinite(scientific_null):
            raise ValueError("scientific_null must be finite")
        p_scientific = (
            1.0
            + np.count_nonzero(
                samples - observed >= observed - scientific_null
            )
        ) / (n_bootstrap + 1.0)
    return BootstrapResult(
        estimate=observed,
        ci_low=float(low),
        ci_high=float(high),
        p_greater=float(p_greater),
        n_bootstrap=n_bootstrap,
        scientific_null=scientific_null,
        p_greater_scientific_null=(
            float(p_scientific) if p_scientific is not None else None
        ),
    )


def _validate_anchors(
    anchors: AnchorMap,
    items: Sequence[str],
    labels: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], tuple[str, ...]]:
    if set(anchors) != set(items):
        raise ValueError(
            f"Anchor items mismatch: observed={sorted(anchors)}, expected={sorted(items)}"
        )
    observed_labels = set(next(iter(anchors.values())))
    if labels is not None and observed_labels != set(labels):
        raise ValueError("Anchor labels do not match the assignment labels")
    if len(observed_labels) < 2:
        raise ValueError("At least two anchor labels are required")
    normalized: dict[str, dict[str, np.ndarray]] = {}
    option_counts: dict[str, int] = {}
    for item in items:
        if set(anchors[item]) != observed_labels:
            raise ValueError(f"Anchor labels differ across items at {item}")
        normalized[item] = {}
        for label, raw in anchors[item].items():
            vector = np.asarray(raw, dtype=float)
            if vector.ndim != 1 or len(vector) < 2 or not np.all(np.isfinite(vector)):
                raise ValueError(f"Invalid anchor vector for {item}/{label}")
            if np.any(vector < -PROBABILITY_TOLERANCE) or not np.isclose(
                vector.sum(), 1.0, atol=PROBABILITY_TOLERANCE
            ):
                raise ValueError(f"Anchor is not a probability distribution: {item}/{label}")
            option_counts.setdefault(item, len(vector))
            if len(vector) != option_counts[item]:
                raise ValueError(f"Anchor option count differs for item {item}")
            normalized[item][label] = vector
    return normalized, tuple(sorted(observed_labels))


class _DirectionPanel:
    def __init__(
        self,
        rows: Iterable[DirectionObservation | Mapping[str, Any]],
        anchors: AnchorMap,
        *,
        expected_providers: Sequence[str] | None = None,
        expected_countries: Sequence[str] | None = None,
        expected_items: Sequence[str] | None = None,
        require_balanced_assignment: bool = True,
    ) -> None:
        observations = [_as_direction(row) for row in rows]
        if not observations:
            raise ValueError("Direction panel is empty")
        self.providers = _check_expected(
            [row.provider for row in observations], expected_providers, "providers"
        )
        self.countries = _check_expected(
            [row.country for row in observations], expected_countries, "countries"
        )
        self.items = _check_expected(
            [row.item_id for row in observations], expected_items, "items"
        )
        self.anchors, self.labels = _validate_anchors(anchors, self.items)
        records: dict[tuple[str, str, str, str], DirectionObservation] = {}
        for row in observations:
            key = (row.provider, row.country, row.profile_id, row.item_id)
            if key in records:
                raise ValueError(f"Duplicate direction observation: {key}")
            if not np.isfinite(row.weight) or row.weight <= 0:
                raise ValueError(f"Invalid survey weight for {key}: {row.weight}")
            if row.displayed_label not in self.labels:
                raise ValueError(f"Unknown displayed label for {key}: {row.displayed_label}")
            for name, delta in (("MO", row.mo_delta), ("MR", row.mr_delta)):
                vector = np.asarray(delta, dtype=float)
                expected_options = len(self.anchors[row.item_id][self.labels[0]])
                if vector.shape != (expected_options,) or not np.all(np.isfinite(vector)):
                    raise ValueError(f"Invalid {name} delta vector for {key}")
                if not np.isclose(vector.sum(), 0.0, atol=PROBABILITY_TOLERANCE):
                    raise ValueError(f"{name} delta does not sum to zero for {key}")
            records[key] = row

        self.profile_ids: dict[str, tuple[str, ...]] = {}
        self.weights: dict[str, np.ndarray] = {}
        self.displayed: dict[str, np.ndarray] = {}
        self.mo_delta: dict[str, list[np.ndarray]] = {}
        self.mr_delta: dict[str, list[np.ndarray]] = {}
        label_index = {label: index for index, label in enumerate(self.labels)}
        for country in self.countries:
            provider_profiles = {
                provider: {
                    row.profile_id
                    for row in observations
                    if row.provider == provider and row.country == country
                }
                for provider in self.providers
            }
            first = provider_profiles[self.providers[0]]
            if not first or any(ids != first for ids in provider_profiles.values()):
                raise ValueError(
                    f"Providers do not share the same profiles in country {country}"
                )
            profiles = tuple(sorted(first))
            self.profile_ids[country] = profiles
            weights = np.empty(len(profiles), dtype=float)
            displayed = np.empty(len(profiles), dtype=int)
            mo = [
                np.empty(
                    (
                        len(self.providers),
                        len(profiles),
                        len(self.anchors[item][self.labels[0]]),
                    ),
                    dtype=float,
                )
                for item in self.items
            ]
            mr = [np.empty_like(values) for values in mo]
            for p_index, provider in enumerate(self.providers):
                for i_index, item in enumerate(self.items):
                    for r_index, profile_id in enumerate(profiles):
                        key = (provider, country, profile_id, item)
                        if key not in records:
                            raise ValueError(f"Missing direction observation: {key}")
                        row = records[key]
                        mo[i_index][p_index, r_index] = np.asarray(
                            row.mo_delta, dtype=float
                        )
                        mr[i_index][p_index, r_index] = np.asarray(
                            row.mr_delta, dtype=float
                        )
                        if p_index == 0 and i_index == 0:
                            weights[r_index] = row.weight
                            displayed[r_index] = label_index[row.displayed_label]
                        else:
                            if not np.isclose(weights[r_index], row.weight):
                                raise ValueError(
                                    "Survey weight differs across provider/item for "
                                    f"profile {country}/{profile_id}"
                                )
                            if displayed[r_index] != label_index[row.displayed_label]:
                                raise ValueError(
                                    "Displayed label differs across provider/item for "
                                    f"profile {country}/{profile_id}"
                                )
            counts = np.bincount(displayed, minlength=len(self.labels))
            if require_balanced_assignment and (
                np.any(counts == 0) or np.any(counts != counts[0])
            ):
                raise ValueError(
                    f"Displayed-label assignment is not balanced in {country}: {counts.tolist()}"
                )
            self.weights[country] = weights
            self.displayed[country] = displayed
            self.mo_delta[country] = mo
            self.mr_delta[country] = mr
        expected_count = sum(
            len(self.providers) * len(self.items) * len(self.profile_ids[country])
            for country in self.countries
        )
        if len(records) != expected_count:
            raise ValueError("Direction panel contains observations outside its full panel")
        self._projection_cache = self.projection_arrays(self.anchors)

    def projection_arrays(
        self, anchors: AnchorMap
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        normalized, labels = _validate_anchors(anchors, self.items, self.labels)
        if labels != self.labels:
            raise ValueError("Anchor label order changed")
        directions: dict[str, list[np.ndarray]] = {}
        norms = np.empty((len(self.items), len(self.labels)), dtype=float)
        for i_index, item in enumerate(self.items):
            pooled = np.mean([normalized[item][label] for label in self.labels], axis=0)
            directions[item] = []
            for l_index, label in enumerate(self.labels):
                vector = normalized[item][label] - pooled
                directions[item].append(vector)
                norms[i_index, l_index] = float(np.dot(vector, vector))
        if not np.any(norms > ZERO_TOLERANCE):
            raise ValueError("All reference directions are zero")

        arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for country in self.countries:
            shape = (
                len(self.providers),
                len(self.items),
                len(self.profile_ids[country]),
                len(self.labels),
            )
            mo_projection = np.empty(shape, dtype=float)
            mr_projection = np.empty(shape, dtype=float)
            for i_index, item in enumerate(self.items):
                direction_matrix = np.asarray(directions[item], dtype=float)
                mo_projection[:, i_index] = np.einsum(
                    "pro,lo->prl",
                    self.mo_delta[country][i_index],
                    direction_matrix,
                    optimize=True,
                )
                mr_projection[:, i_index] = np.einsum(
                    "pro,lo->prl",
                    self.mr_delta[country][i_index],
                    direction_matrix,
                    optimize=True,
                )
            country_norms = np.broadcast_to(
                norms[:, None, :],
                (len(self.items), len(self.profile_ids[country]), len(self.labels)),
            ).copy()
            arrays[country] = (mo_projection, mr_projection, country_norms)
        return arrays

    def estimate_by_provider(
        self,
        *,
        projections: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        labels: Mapping[str, np.ndarray] | None = None,
        sample_indexes: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        arrays = self._projection_cache if projections is None else projections
        country_beta_o: list[np.ndarray] = []
        country_beta_r: list[np.ndarray] = []
        for country in self.countries:
            indexes = (
                np.arange(len(self.profile_ids[country]))
                if sample_indexes is None
                else np.asarray(sample_indexes[country], dtype=int)
            )
            if labels is None:
                assigned = self.displayed[country][indexes]
            else:
                supplied = np.asarray(labels[country], dtype=int)
                assigned = supplied[indexes]
            weights = self.weights[country][indexes]
            weights = weights / weights.sum()
            mo, mr, norms = arrays[country]
            mo = mo[:, :, indexes, :]
            mr = mr[:, :, indexes, :]
            norms = norms[:, indexes, :]
            select = assigned[None, None, :, None]
            selected_mo = np.take_along_axis(mo, select, axis=3)[..., 0]
            selected_mr = np.take_along_axis(mr, select, axis=3)[..., 0]
            selected_norm = np.take_along_axis(
                norms, assigned[None, :, None], axis=2
            )[..., 0]
            numerator_o = np.sum(
                selected_mo * weights[None, None, :], axis=2
            )
            numerator_r = np.sum(
                selected_mr * weights[None, None, :], axis=2
            )
            denominator = np.sum(selected_norm * weights[None, :], axis=1)
            if np.any(denominator <= ZERO_TOLERANCE):
                bad = [
                    self.items[index]
                    for index in np.flatnonzero(denominator <= ZERO_TOLERANCE)
                ]
                raise ValueError(
                    f"Zero country-direction denominator in {country} for items: {bad}"
                )
            country_beta_o.append(numerator_o / denominator[None, :])
            country_beta_r.append(numerator_r / denominator[None, :])
        # Country-specific ratios are averaged so countries, items, and providers
        # each receive equal macro weight. Pooling denominators across countries
        # would implicitly upweight countries with larger direction norms.
        beta_o = np.mean(np.asarray(country_beta_o), axis=(0, 2))
        beta_r = np.mean(np.asarray(country_beta_r), axis=(0, 2))
        return np.column_stack((beta_o, beta_r, beta_o - beta_r))

    def estimate(
        self,
        *,
        projections: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        labels: Mapping[str, np.ndarray] | None = None,
        sample_indexes: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[float, float, float]:
        # Providers and items receive equal weight in the macro estimate.
        values = self.estimate_by_provider(
            projections=projections,
            labels=labels,
            sample_indexes=sample_indexes,
        )
        return tuple(float(value) for value in np.mean(values, axis=0))


def direction_beta_macro(
    rows: Iterable[DirectionObservation | Mapping[str, Any]],
    anchors: AnchorMap,
    **panel_expectations: Any,
) -> tuple[float, float, float]:
    """Return beta_O, beta_R, and beta_O-beta_R under the frozen weighting."""
    return _DirectionPanel(rows, anchors, **panel_expectations).estimate()


def direction_profile_bootstrap(
    rows: Iterable[DirectionObservation | Mapping[str, Any]],
    anchors: AnchorMap,
    *,
    anchor_bootstrap: AnchorBootstrap | None = None,
    require_anchor_reestimation: bool = False,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    seed: int = 20260725,
    confidence: float = 0.95,
    scientific_nulls: Mapping[str, float] | None = None,
    **panel_expectations: Any,
) -> DirectionBootstrapResult:
    """Bootstrap profiles and, when supplied, re-estimate reference anchors."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if require_anchor_reestimation and anchor_bootstrap is None:
        raise ValueError("Confirmatory direction bootstrap requires anchor re-estimation")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    panel = _DirectionPanel(rows, anchors, **panel_expectations)
    observed = panel.estimate()
    rng = np.random.default_rng(seed)
    samples = np.empty((n_bootstrap, 3), dtype=float)
    for index in range(n_bootstrap):
        sampled = {
            country: rng.integers(0, len(panel.profile_ids[country]), len(panel.profile_ids[country]))
            for country in panel.countries
        }
        projections = (
            panel._projection_cache
            if anchor_bootstrap is None
            else panel.projection_arrays(anchor_bootstrap(rng))
        )
        samples[index] = panel.estimate(
            projections=projections, sample_indexes=sampled
        )
    alpha = 1.0 - confidence
    limits = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)

    margins = {"beta_o": 0.0, "beta_r": 0.0, "beta_diff": 0.0}
    if scientific_nulls is not None:
        unknown = set(scientific_nulls).difference(margins)
        if unknown:
            raise ValueError(f"Unknown direction scientific nulls: {sorted(unknown)}")
        margins.update({name: float(value) for name, value in scientific_nulls.items()})
    if any(not np.isfinite(value) for value in margins.values()):
        raise ValueError("Direction scientific nulls must be finite")

    def result(column: int, name: str) -> BootstrapResult:
        estimate = observed[column]
        p_greater = (1.0 + np.count_nonzero(samples[:, column] - estimate >= estimate)) / (
            n_bootstrap + 1.0
        )
        scientific_null = margins[name]
        p_scientific = (
            1.0
            + np.count_nonzero(
                samples[:, column] - estimate >= estimate - scientific_null
            )
        ) / (n_bootstrap + 1.0)
        return BootstrapResult(
            estimate=estimate,
            ci_low=float(limits[0, column]),
            ci_high=float(limits[1, column]),
            p_greater=float(p_greater),
            n_bootstrap=n_bootstrap,
            anchors_reestimated=anchor_bootstrap is not None,
            scientific_null=scientific_null,
            p_greater_scientific_null=float(p_scientific),
        )

    return DirectionBootstrapResult(
        result(0, "beta_o"), result(1, "beta_r"), result(2, "beta_diff")
    )


def balanced_assignment_permutation(
    rows: Iterable[DirectionObservation | Mapping[str, Any]],
    anchors: AnchorMap,
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 20260726,
    chunk_size: int = 32,
    **panel_expectations: Any,
) -> PermutationResult:
    """Sharp-null test that permutes displayed labels within true country.

    One profile-level permutation is shared by MO and MR and by all providers
    and items. Each permutation preserves the observed, balanced label counts.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    panel = _DirectionPanel(rows, anchors, **panel_expectations)
    observed_o, observed_r, observed_diff = panel.estimate()
    rng = np.random.default_rng(seed)
    exceed_o = exceed_r = exceed_diff = 0
    completed = 0
    while completed < n_permutations:
        size = min(chunk_size, n_permutations - completed)
        country_beta_o = np.zeros(
            (size, len(panel.providers), len(panel.items)), dtype=float
        )
        country_beta_r = np.zeros_like(country_beta_o)
        for country in panel.countries:
            count = len(panel.profile_ids[country])
            order = np.argsort(rng.random((size, count)), axis=1)
            assigned = panel.displayed[country][order]
            mo, mr, norms = panel._projection_cache[country]
            indexes = assigned[:, None, None, :, None]
            selected_mo = np.take_along_axis(
                mo[None, ...], indexes, axis=4
            )[..., 0]
            weights = panel.weights[country]
            weights = weights / weights.sum()
            numerator_o = np.sum(
                selected_mo * weights[None, None, None, :], axis=3
            )
            del selected_mo
            selected_mr = np.take_along_axis(
                mr[None, ...], indexes, axis=4
            )[..., 0]
            numerator_r = np.sum(
                selected_mr * weights[None, None, None, :], axis=3
            )
            del selected_mr
            selected_norm = np.take_along_axis(
                norms[None, ...], assigned[:, None, :, None], axis=3
            )[..., 0]
            denominator = np.sum(
                selected_norm * weights[None, None, :], axis=2
            )
            if np.any(denominator <= ZERO_TOLERANCE):
                raise ValueError(
                    "A permutation produced a zero country-direction denominator"
                )
            country_beta_o += numerator_o / denominator[:, None, :]
            country_beta_r += numerator_r / denominator[:, None, :]
        country_beta_o /= len(panel.countries)
        country_beta_r /= len(panel.countries)
        beta_o = np.mean(country_beta_o, axis=(1, 2))
        beta_r = np.mean(country_beta_r, axis=(1, 2))
        beta_diff = beta_o - beta_r
        exceed_o += int(np.count_nonzero(beta_o >= observed_o))
        exceed_r += int(np.count_nonzero(beta_r >= observed_r))
        exceed_diff += int(np.count_nonzero(beta_diff >= observed_diff))
        completed += size
    p_o = (exceed_o + 1.0) / (n_permutations + 1.0)
    p_r = (exceed_r + 1.0) / (n_permutations + 1.0)
    p_diff = (exceed_diff + 1.0) / (n_permutations + 1.0)
    h1 = max(p_o, p_diff)
    mc_se = float(np.sqrt(h1 * (1.0 - h1) / (n_permutations + 1.0)))
    return PermutationResult(
        beta_o=observed_o,
        beta_r=observed_r,
        beta_diff=observed_diff,
        p_beta_o_greater=float(p_o),
        p_beta_r_greater=float(p_r),
        p_beta_diff_greater=float(p_diff),
        h1_iut_p=float(h1),
        n_permutations=n_permutations,
        monte_carlo_se_h1_upper_bound=mc_se,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise adjusted p-values with monotonic correction."""
    if not p_values:
        raise ValueError("No p-values supplied for Holm correction")
    for name, value in p_values.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid p-value for {name}: {value}")
    ordered = sorted(p_values.items(), key=lambda pair: pair[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


class ReferenceAnchorResampler:
    """Cluster reference respondents within country and rebuild item anchors."""

    def __init__(
        self,
        rows: Iterable[ReferenceObservation | Mapping[str, Any]],
        *,
        expected_countries: Sequence[str] | None = None,
        expected_items: Sequence[str] | None = None,
    ) -> None:
        observations: list[ReferenceObservation] = []
        for raw in rows:
            if isinstance(raw, ReferenceObservation):
                row = raw
            else:
                row = ReferenceObservation(
                    country=str(raw.get("country", raw.get("country_code", ""))),
                    profile_id=str(_required(raw, "profile_id")),
                    item_id=str(_required(raw, "item_id")),
                    answer_index=int(_required(raw, "answer_index")),
                    n_options=int(_required(raw, "n_options")),
                    weight=float(raw.get("weight", raw.get("analysis_weight", np.nan))),
                )
            if (
                not row.country
                or not row.profile_id
                or not row.item_id
                or row.n_options < 2
                or not -1 <= row.answer_index < row.n_options
                or not np.isfinite(row.weight)
                or row.weight <= 0
            ):
                raise ValueError(f"Invalid reference observation: {row}")
            observations.append(row)
        if not observations:
            raise ValueError("Reference panel is empty")
        self.countries = _check_expected(
            [row.country for row in observations], expected_countries, "reference countries"
        )
        self.items = _check_expected(
            [row.item_id for row in observations], expected_items, "reference items"
        )
        records: dict[tuple[str, str, str], ReferenceObservation] = {}
        self.profile_ids: dict[str, tuple[str, ...]] = {}
        for row in observations:
            key = (row.country, row.profile_id, row.item_id)
            if key in records:
                raise ValueError(f"Duplicate reference observation: {key}")
            records[key] = row
        self.answers: dict[str, np.ndarray] = {}
        self.weights: dict[str, np.ndarray] = {}
        self.n_options: dict[str, int] = {}
        for country in self.countries:
            profiles = tuple(
                sorted({row.profile_id for row in observations if row.country == country})
            )
            if not profiles:
                raise ValueError(f"No reference profiles in {country}")
            answers = np.empty((len(self.items), len(profiles)), dtype=int)
            weights = np.empty(len(profiles), dtype=float)
            for i_index, item in enumerate(self.items):
                for r_index, profile_id in enumerate(profiles):
                    key = (country, profile_id, item)
                    if key not in records:
                        raise ValueError(f"Missing reference observation: {key}")
                    row = records[key]
                    self.n_options.setdefault(item, row.n_options)
                    if self.n_options[item] != row.n_options:
                        raise ValueError(f"Option count differs for reference item {item}")
                    answers[i_index, r_index] = row.answer_index
                    if i_index == 0:
                        weights[r_index] = row.weight
                    elif not np.isclose(weights[r_index], row.weight):
                        raise ValueError(
                            f"Reference weight differs across items for {country}/{profile_id}"
                        )
            self.profile_ids[country] = profiles
            self.answers[country] = answers
            self.weights[country] = weights

    def estimate(self, sample_indexes: Mapping[str, np.ndarray] | None = None) -> dict[str, dict[str, list[float]]]:
        anchors: dict[str, dict[str, list[float]]] = {
            item: {} for item in self.items
        }
        for country in self.countries:
            indexes = (
                np.arange(len(self.profile_ids[country]))
                if sample_indexes is None
                else np.asarray(sample_indexes[country], dtype=int)
            )
            weights = self.weights[country][indexes]
            for i_index, item in enumerate(self.items):
                valid = self.answers[country][i_index, indexes] >= 0
                if not np.any(valid):
                    raise ValueError(f"Empty resampled anchor for {item}/{country}")
                counts = np.bincount(
                    self.answers[country][i_index, indexes][valid],
                    weights=weights[valid],
                    minlength=self.n_options[item],
                )
                if counts.sum() <= 0:
                    raise ValueError(f"Empty resampled anchor for {item}/{country}")
                anchors[item][country] = (counts / counts.sum()).tolist()
        return anchors

    def __call__(self, rng: np.random.Generator) -> AnchorMap:
        sampled = {
            country: rng.integers(0, len(self.profile_ids[country]), len(self.profile_ids[country]))
            for country in self.countries
        }
        return self.estimate(sampled)


def run_strict_confirmatory_inference(
    direction_rows: Iterable[DirectionObservation | Mapping[str, Any]],
    utility_rows: Iterable[ScalarObservation | Mapping[str, Any]],
    anchors: AnchorMap,
    *,
    anchor_bootstrap: AnchorBootstrap,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    bootstrap_seed: int = 20260725,
    permutation_seed: int = 20260726,
    expected_providers: Sequence[str] | None = None,
    expected_countries: Sequence[str] | None = None,
    expected_items: Sequence[str] | None = None,
    beta_o_scientific_null: float = 0.0,
    beta_diff_scientific_null: float = 0.05,
    utility_scientific_null: float = 0.02,
) -> StrictInferenceResult:
    """Run H1/H2 inference and Holm adjustment, failing closed on omissions."""
    expectations = {
        "expected_providers": expected_providers,
        "expected_countries": expected_countries,
        "expected_items": expected_items,
    }
    direction_list = list(direction_rows)
    utility_list = list(utility_rows)
    direction_bootstrap = direction_profile_bootstrap(
        direction_list,
        anchors,
        anchor_bootstrap=anchor_bootstrap,
        require_anchor_reestimation=True,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
        scientific_nulls={
            "beta_o": beta_o_scientific_null,
            "beta_diff": beta_diff_scientific_null,
        },
        **expectations,
    )
    assignment = balanced_assignment_permutation(
        direction_list,
        anchors,
        n_permutations=n_permutations,
        seed=permutation_seed,
        **expectations,
    )
    utility = paired_profile_bootstrap(
        utility_list,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed + 1,
        scientific_null=utility_scientific_null,
        **expectations,
    )
    if utility.p_greater is None:
        raise RuntimeError("Utility bootstrap did not return an H2 p-value")
    if any(
        value is None
        for value in (
            direction_bootstrap.beta_o.p_greater,
            direction_bootstrap.beta_diff.p_greater,
            direction_bootstrap.beta_o.p_greater_scientific_null,
            direction_bootstrap.beta_diff.p_greater_scientific_null,
            utility.p_greater_scientific_null,
        )
    ):
        raise RuntimeError("Bootstrap did not return the required p-values")
    # The confirmatory H1 concerns positive average effects (a weak null), so
    # its IUT uses the paired profile/reference bootstrap.  The assignment
    # permutation remains a separately reported Fisher sharp-null diagnostic;
    # treating its unstudentized p-value as the weak-null test would overstate
    # what the randomization inference identifies.
    positive_h1 = max(
        direction_bootstrap.beta_o.p_greater,
        direction_bootstrap.beta_diff.p_greater,
    )
    sesoi_h1 = max(
        direction_bootstrap.beta_o.p_greater_scientific_null,
        direction_bootstrap.beta_diff.p_greater_scientific_null,
    )
    raw = {
        "H1": positive_h1,
        "H2": utility.p_greater,
    }
    sesoi = {
        "H1": sesoi_h1,
        "H2": utility.p_greater_scientific_null,
    }
    return StrictInferenceResult(
        direction_bootstrap=direction_bootstrap,
        assignment_permutation=assignment,
        utility_h2=utility,
        raw_hypothesis_p=raw,
        holm_adjusted_p=holm_adjust(raw),
        raw_sesoi_exceedance_p=sesoi,
    )
