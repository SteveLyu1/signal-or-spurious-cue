"""Conservative 504-profile sensitivity analysis based on the Gate C pilot.

Each Gate C delta is observed under one displayed label only. Accordingly,
profiles are resampled with replacement inside true-country x original-label
cells, from two pilot profiles to fourteen planned profiles per cell. Labels and
model responses are never reassigned. Reference respondents are independently
resampled and country anchors are rebuilt on every draw.

This estimates sensitivity under pilot-transferred sampling error and assumes
the Gate C point estimates are the future effects. It is not power for the
planned assignment-permutation test and cannot identify unobserved potential
outcomes under alternative displayed labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from api_pilot.strict_inference import (
    AnchorBootstrap,
    AnchorMap,
    DirectionObservation,
    ScalarObservation,
    _DirectionPanel,
    _ScalarPanel,
)


DEFAULT_PROFILES_PER_COUNTRY = 84
DEFAULT_ALPHA = 0.025
DEFAULT_BETA_O_NULL = 0.0
DEFAULT_BETA_DIFF_NULL = 0.05
DEFAULT_H2_NULL = 0.02
DEFAULT_REPLICATES = 10_000
DEFAULT_POWER_TARGET = 0.80
DEFAULT_BETA_O_ALTERNATIVE = 0.05
DEFAULT_BETA_DIFF_ALTERNATIVE = 0.05
DEFAULT_H2_ALTERNATIVE = 0.02


@dataclass(frozen=True)
class MarginalSensitivity:
    assumed_effect: float
    null_threshold: float
    alpha: float
    error_critical_value: float
    estimated_power: float
    monte_carlo_se: float


@dataclass(frozen=True)
class ProviderSensitivity:
    pilot_beta_o: float
    pilot_beta_r: float
    pilot_beta_diff: float
    pilot_verified_utility: float
    beta_o: MarginalSensitivity
    beta_diff: MarginalSensitivity
    h1_iut_power: float
    h1_iut_monte_carlo_se: float
    h2_verified_utility: MarginalSensitivity
    joint_co_primary_power: float
    joint_co_primary_monte_carlo_se: float
    minimum_co_primary_power: float
    reaches_80_percent: bool


@dataclass(frozen=True)
class SensitivitySimulationResult:
    providers: tuple[str, ...]
    profiles_per_country: int
    profiles_total: int
    displayed_labels_per_country: int
    pilot_profiles_per_country_label: int
    planned_profiles_per_country_label: int
    calibration_replicates: int
    sensitivity_replicates: int
    calibration_seed: int
    sensitivity_seed: int
    alpha: float
    beta_o_null: float
    beta_diff_null: float
    h2_null: float
    results: dict[str, ProviderSensitivity]
    lowest_provider_diagnostic_sensitivity: float
    all_provider_diagnostics_reach_80_percent: bool
    macro_reaches_80_percent: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamilyDesignPower:
    h1_iut_unadjusted_power: float
    h2_unadjusted_power: float
    h1_holm_power: float
    h2_holm_power: float
    joint_holm_power: float
    joint_holm_wilson_low: float
    joint_holm_wilson_high: float
    any_holm_rejection: float
    joint_holm_monte_carlo_se: float


@dataclass(frozen=True)
class ProspectiveDesignPowerResult:
    """Assumption-based 504-profile power for the equal-provider macro.

    The primary decision uses only ``macro_equal_provider``. Provider entries
    use the same SESOI alternatives and are variance diagnostics, not gates.
    """

    providers: tuple[str, ...]
    profiles_per_country: int
    profiles_total: int
    displayed_labels_per_country: int
    pilot_profiles_per_country_label: int
    planned_profiles_per_country_label: int
    calibration_replicates: int
    power_replicates: int
    calibration_seed: int
    power_seed: int
    familywise_alpha: float
    power_target: float
    null: dict[str, float]
    sesoi_alternative: dict[str, float]
    macro_error_covariance: list[list[float]]
    macro_error_correlation: list[list[float]]
    macro_standard_errors: dict[str, float]
    macro_power: FamilyDesignPower
    null_calibration: FamilyDesignPower
    provider_diagnostics: dict[str, FamilyDesignPower]
    variance_inflation_sensitivity: dict[str, ProspectiveSensitivityScenario]
    single_provider_dropout_sensitivity: dict[str, ProspectiveSensitivityScenario]
    primary_gate_passes: bool
    identifiable_assignment_power: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProspectiveSensitivityScenario:
    """Non-gating stress test using the same prospective bootstrap draws."""

    providers: tuple[str, ...]
    excluded_provider: str | None
    variance_multiplier: float
    standard_deviation_multiplier: float
    macro_power: FamilyDesignPower
    null_calibration: FamilyDesignPower
    reaches_power_target: bool


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, probability, interpolation="higher"))


def _mc_se(power: float, replicates: int) -> float:
    return float(np.sqrt(power * (1.0 - power) / replicates))


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Invalid binomial counts")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))


def _empirical_upper_tail_pvalues(
    calibration_errors: np.ndarray,
    test_statistics: np.ndarray,
) -> np.ndarray:
    """Return finite-simulation one-sided empirical p-values.

    Columns are separate elementary hypotheses and rows are joint replicates.
    Keeping the rows intact preserves covariance among beta_O, beta_diff, and
    verified utility when Holm decisions are simulated.
    """
    calibration = np.asarray(calibration_errors, dtype=float)
    statistics = np.asarray(test_statistics, dtype=float)
    if calibration.ndim != 2 or statistics.ndim != 2:
        raise ValueError("Calibration errors and test statistics must be matrices")
    if calibration.shape[1] != statistics.shape[1]:
        raise ValueError("Calibration and test statistic columns differ")
    pvalues = np.empty_like(statistics)
    for column in range(calibration.shape[1]):
        sorted_errors = np.sort(calibration[:, column])
        first_ge = np.searchsorted(
            sorted_errors, statistics[:, column], side="left"
        )
        exceedances = len(sorted_errors) - first_ge
        pvalues[:, column] = (exceedances + 1.0) / (
            len(sorted_errors) + 1.0
        )
    return pvalues


def _holm_two_family_rejections(
    h1_pvalues: np.ndarray,
    h2_pvalues: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Holm's step-down rule to two co-primary family p-values."""
    h1 = np.asarray(h1_pvalues, dtype=float)
    h2 = np.asarray(h2_pvalues, dtype=float)
    if h1.shape != h2.shape:
        raise ValueError("H1 and H2 p-value arrays are not aligned")
    h1_first = h1 <= h2
    first_p = np.where(h1_first, h1, h2)
    second_p = np.where(h1_first, h2, h1)
    first_reject = first_p <= alpha / 2.0
    second_reject = first_reject & (second_p <= alpha)
    h1_reject = np.where(h1_first, first_reject, second_reject)
    h2_reject = np.where(h1_first, second_reject, first_reject)
    return h1_reject, h2_reject


def _family_design_power(
    calibration_errors: np.ndarray,
    evaluation_errors: np.ndarray,
    alternatives: np.ndarray,
    *,
    alpha: float,
) -> FamilyDesignPower:
    """Simulate IUT H1 and H2, then apply Holm across the two families."""
    calibration = np.asarray(calibration_errors, dtype=float)
    evaluation = np.asarray(evaluation_errors, dtype=float)
    alternatives = np.asarray(alternatives, dtype=float)
    if calibration.shape[1] != 3 or evaluation.shape[1] != 3:
        raise ValueError("Power matrices must contain beta_O, beta_diff, utility")
    if alternatives.shape != (3,):
        raise ValueError("Alternatives must have exactly three values")
    elementary_p = _empirical_upper_tail_pvalues(
        calibration, evaluation + alternatives[None, :]
    )
    # H1 is an intersection-union test. Its valid p-value is the larger of the
    # two one-sided component p-values, without an internal multiplicity penalty.
    h1_p = np.maximum(elementary_p[:, 0], elementary_p[:, 1])
    h2_p = elementary_p[:, 2]
    h1_unadjusted = h1_p <= alpha
    h2_unadjusted = h2_p <= alpha
    h1_holm, h2_holm = _holm_two_family_rejections(h1_p, h2_p, alpha=alpha)
    joint_holm = h1_holm & h2_holm
    any_holm = h1_holm | h2_holm
    joint_successes = int(np.count_nonzero(joint_holm))
    joint_low, joint_high = _wilson_interval(joint_successes, len(joint_holm))
    return FamilyDesignPower(
        h1_iut_unadjusted_power=float(np.mean(h1_unadjusted)),
        h2_unadjusted_power=float(np.mean(h2_unadjusted)),
        h1_holm_power=float(np.mean(h1_holm)),
        h2_holm_power=float(np.mean(h2_holm)),
        joint_holm_power=float(np.mean(joint_holm)),
        joint_holm_wilson_low=joint_low,
        joint_holm_wilson_high=joint_high,
        any_holm_rejection=float(np.mean(any_holm)),
        joint_holm_monte_carlo_se=_mc_se(float(np.mean(joint_holm)), len(joint_holm)),
    )


def _stratified_profile_indexes(
    panel: _DirectionPanel,
    rng: np.random.Generator,
    profiles_per_country: int,
) -> dict[str, np.ndarray]:
    """Resample within original country-label cells without relabelling."""
    label_count = len(panel.labels)
    if profiles_per_country < label_count or profiles_per_country % label_count:
        raise ValueError(
            "profiles_per_country must be a positive multiple of displayed labels"
        )
    per_label = profiles_per_country // label_count
    sampled: dict[str, np.ndarray] = {}
    for country in panel.countries:
        pieces = []
        for label_index in range(label_count):
            candidates = np.flatnonzero(panel.displayed[country] == label_index)
            if not len(candidates):
                raise ValueError(
                    f"No observed profiles for {country}/{panel.labels[label_index]}"
                )
            pieces.append(rng.choice(candidates, size=per_label, replace=True))
        indexes = np.concatenate(pieces)
        sampled[country] = indexes[rng.permutation(len(indexes))]
    return sampled


def _validate_alignment(
    direction: _DirectionPanel,
    utility: _ScalarPanel,
) -> None:
    if (
        direction.providers != utility.providers
        or direction.countries != utility.countries
        or direction.items != utility.items
    ):
        raise ValueError("Direction and utility panels have different dimensions")
    for country in direction.countries:
        if direction.profile_ids[country] != utility.profile_ids[country]:
            raise ValueError(
                f"Direction and utility profiles are not aligned in {country}"
            )


def _draw_statistics(
    direction: _DirectionPanel,
    utility: _ScalarPanel,
    anchor_bootstrap: AnchorBootstrap,
    *,
    replicates: int,
    profiles_per_country: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if replicates < 100:
        raise ValueError("At least 100 simulation replicates are required")
    rng = np.random.default_rng(seed)
    # Final row is the equal-provider macro.
    direction_samples = np.empty(
        (replicates, len(direction.providers) + 1, 3), dtype=float
    )
    utility_samples = np.empty(
        (replicates, len(direction.providers) + 1), dtype=float
    )
    for replicate in range(replicates):
        indexes = _stratified_profile_indexes(
            direction, rng, profiles_per_country
        )
        projections = direction.projection_arrays(anchor_bootstrap(rng))
        provider_direction = direction.estimate_by_provider(
            projections=projections,
            sample_indexes=indexes,
        )
        provider_utility = utility.estimate_by_provider(indexes)
        direction_samples[replicate, :-1] = provider_direction
        direction_samples[replicate, -1] = np.mean(provider_direction, axis=0)
        utility_samples[replicate, :-1] = provider_utility
        utility_samples[replicate, -1] = np.mean(provider_utility)
    return direction_samples, utility_samples


def _marginal_sensitivity(
    calibration_error: np.ndarray,
    sensitivity_error: np.ndarray,
    *,
    assumed_effect: float,
    null_threshold: float,
    alpha: float,
) -> tuple[MarginalSensitivity, np.ndarray]:
    """Use empirical centred errors to test an assumed effect against a threshold."""
    critical = _higher_quantile(calibration_error, 1.0 - alpha)
    rejects = assumed_effect + sensitivity_error - null_threshold > critical
    power = float(np.mean(rejects))
    return (
        MarginalSensitivity(
            assumed_effect=assumed_effect,
            null_threshold=null_threshold,
            alpha=alpha,
            error_critical_value=critical,
            estimated_power=power,
            monte_carlo_se=_mc_se(power, len(rejects)),
        ),
        rejects,
    )


def simulate_confirmatory_sensitivity(
    direction_rows: Iterable[DirectionObservation | Mapping[str, Any]],
    utility_rows: Iterable[ScalarObservation | Mapping[str, Any]],
    anchors: AnchorMap,
    *,
    anchor_bootstrap: AnchorBootstrap,
    profiles_per_country: int = DEFAULT_PROFILES_PER_COUNTRY,
    calibration_replicates: int = DEFAULT_REPLICATES,
    sensitivity_replicates: int = DEFAULT_REPLICATES,
    alpha: float = DEFAULT_ALPHA,
    beta_o_null: float = DEFAULT_BETA_O_NULL,
    beta_diff_null: float = DEFAULT_BETA_DIFF_NULL,
    h2_null: float = DEFAULT_H2_NULL,
    calibration_seed: int = 2026072501,
    sensitivity_seed: int = 2026072502,
    expected_providers: Sequence[str] | None = None,
    expected_countries: Sequence[str] | None = None,
    expected_items: Sequence[str] | None = None,
) -> SensitivitySimulationResult:
    """Estimate threshold sensitivity at the Gate C point-estimate alternatives.

    Independent stratified bootstrap draws calibrate one-sided error cut-offs and
    evaluate sensitivity at the supplied point-effect thresholds. This diagnostic
    predates the prospective macro design-power approximation and is non-gating.
    """
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and 0.5")
    expectations = {
        "expected_providers": expected_providers,
        "expected_countries": expected_countries,
        "expected_items": expected_items,
    }
    direction = _DirectionPanel(list(direction_rows), anchors, **expectations)
    utility = _ScalarPanel(list(utility_rows), **expectations)
    _validate_alignment(direction, utility)

    calibration_direction, calibration_utility = _draw_statistics(
        direction,
        utility,
        anchor_bootstrap,
        replicates=calibration_replicates,
        profiles_per_country=profiles_per_country,
        seed=calibration_seed,
    )
    sensitivity_direction, sensitivity_utility = _draw_statistics(
        direction,
        utility,
        anchor_bootstrap,
        replicates=sensitivity_replicates,
        profiles_per_country=profiles_per_country,
        seed=sensitivity_seed,
    )
    # Only sampling errors are transferred. Gate C point estimates define the
    # assumed future effects; empirical bootstrap means are removed separately.
    calibration_direction -= np.mean(
        calibration_direction, axis=0, keepdims=True
    )
    calibration_utility -= np.mean(calibration_utility, axis=0, keepdims=True)
    sensitivity_direction -= np.mean(
        sensitivity_direction, axis=0, keepdims=True
    )
    sensitivity_utility -= np.mean(sensitivity_utility, axis=0, keepdims=True)

    observed_direction = direction.estimate_by_provider()
    observed_utility = utility.estimate_by_provider()
    observed_direction = np.vstack(
        (observed_direction, np.mean(observed_direction, axis=0))
    )
    observed_utility = np.append(observed_utility, np.mean(observed_utility))
    names = (*direction.providers, "macro_equal_provider")
    results: dict[str, ProviderSensitivity] = {}
    for index, name in enumerate(names):
        beta_o, reject_o = _marginal_sensitivity(
            calibration_direction[:, index, 0],
            sensitivity_direction[:, index, 0],
            assumed_effect=float(observed_direction[index, 0]),
            null_threshold=beta_o_null,
            alpha=alpha,
        )
        beta_diff, reject_diff = _marginal_sensitivity(
            calibration_direction[:, index, 2],
            sensitivity_direction[:, index, 2],
            assumed_effect=float(observed_direction[index, 2]),
            null_threshold=beta_diff_null,
            alpha=alpha,
        )
        h1_reject = reject_o & reject_diff
        h1_power = float(np.mean(h1_reject))
        h2, reject_h2 = _marginal_sensitivity(
            calibration_utility[:, index],
            sensitivity_utility[:, index],
            assumed_effect=float(observed_utility[index]),
            null_threshold=h2_null,
            alpha=alpha,
        )
        joint_reject = h1_reject & reject_h2
        joint_power = float(np.mean(joint_reject))
        minimum = min(h1_power, h2.estimated_power)
        results[name] = ProviderSensitivity(
            pilot_beta_o=float(observed_direction[index, 0]),
            pilot_beta_r=float(observed_direction[index, 1]),
            pilot_beta_diff=float(observed_direction[index, 2]),
            pilot_verified_utility=float(observed_utility[index]),
            beta_o=beta_o,
            beta_diff=beta_diff,
            h1_iut_power=h1_power,
            h1_iut_monte_carlo_se=_mc_se(h1_power, sensitivity_replicates),
            h2_verified_utility=h2,
            joint_co_primary_power=joint_power,
            joint_co_primary_monte_carlo_se=_mc_se(
                joint_power, sensitivity_replicates
            ),
            minimum_co_primary_power=minimum,
            reaches_80_percent=minimum >= 0.80,
        )

    provider_minimum = min(
        results[provider].minimum_co_primary_power
        for provider in direction.providers
    )
    all_providers = provider_minimum >= 0.80
    macro_pass = results["macro_equal_provider"].reaches_80_percent
    label_count = len(direction.labels)
    original_cell_counts = {
        len(np.flatnonzero(direction.displayed[country] == label_index))
        for country in direction.countries
        for label_index in range(label_count)
    }
    if len(original_cell_counts) != 1:
        raise ValueError(
            f"Pilot country-label cells have unequal counts: {original_cell_counts}"
        )
    return SensitivitySimulationResult(
        providers=direction.providers,
        profiles_per_country=profiles_per_country,
        profiles_total=profiles_per_country * len(direction.countries),
        displayed_labels_per_country=label_count,
        pilot_profiles_per_country_label=next(iter(original_cell_counts)),
        planned_profiles_per_country_label=profiles_per_country // label_count,
        calibration_replicates=calibration_replicates,
        sensitivity_replicates=sensitivity_replicates,
        calibration_seed=calibration_seed,
        sensitivity_seed=sensitivity_seed,
        alpha=alpha,
        beta_o_null=beta_o_null,
        beta_diff_null=beta_diff_null,
        h2_null=h2_null,
        results=results,
        lowest_provider_diagnostic_sensitivity=provider_minimum,
        all_provider_diagnostics_reach_80_percent=all_providers,
        macro_reaches_80_percent=macro_pass,
    )


def simulate_prospective_macro_design_power(
    direction_rows: Iterable[DirectionObservation | Mapping[str, Any]],
    utility_rows: Iterable[ScalarObservation | Mapping[str, Any]],
    anchors: AnchorMap,
    *,
    anchor_bootstrap: AnchorBootstrap,
    profiles_per_country: int = DEFAULT_PROFILES_PER_COUNTRY,
    calibration_replicates: int = DEFAULT_REPLICATES,
    power_replicates: int = DEFAULT_REPLICATES,
    familywise_alpha: float = 0.05,
    power_target: float = DEFAULT_POWER_TARGET,
    beta_o_alternative: float = DEFAULT_BETA_O_ALTERNATIVE,
    beta_diff_alternative: float = DEFAULT_BETA_DIFF_ALTERNATIVE,
    h2_alternative: float = DEFAULT_H2_ALTERNATIVE,
    calibration_seed: int = 2026072511,
    power_seed: int = 2026072512,
    expected_providers: Sequence[str] | None = None,
    expected_countries: Sequence[str] | None = None,
    expected_items: Sequence[str] | None = None,
    variance_inflation_factors: Sequence[float] = (1.5, 2.0),
) -> ProspectiveDesignPowerResult:
    """Approximate prospective power for the two equal-provider macro families.

    The null for all elementary hypotheses is zero. SESOIs define the
    prospective alternatives rather than non-zero null margins. Joint
    true-country x displayed-label resampling retains provider covariance,
    shared profiles/items, survey weights, and anchor uncertainty. It assumes
    that the pilot's within-cell centred error distribution transfers to the
    504-profile run and that the same fixed four systems remain available.

    Because each pilot profile was queried under one displayed label, this
    cannot identify exact assignment-randomization power. No unobserved label
    potential outcome is manufactured or relabelled.
    """
    if not 0 < familywise_alpha < 0.5:
        raise ValueError("familywise_alpha must be between zero and 0.5")
    if not 0 < power_target < 1:
        raise ValueError("power_target must be between zero and one")
    variance_inflation_factors = tuple(float(value) for value in variance_inflation_factors)
    if any(
        not np.isfinite(value) or value <= 1.0
        for value in variance_inflation_factors
    ):
        raise ValueError("variance inflation factors must be finite and greater than one")
    if len(set(variance_inflation_factors)) != len(variance_inflation_factors):
        raise ValueError("variance inflation factors must be unique")
    alternatives = np.asarray(
        [beta_o_alternative, beta_diff_alternative, h2_alternative], dtype=float
    )
    if np.any(alternatives <= 0) or not np.all(np.isfinite(alternatives)):
        raise ValueError("SESOI alternatives must be positive and finite")
    expectations = {
        "expected_providers": expected_providers,
        "expected_countries": expected_countries,
        "expected_items": expected_items,
    }
    direction = _DirectionPanel(list(direction_rows), anchors, **expectations)
    utility = _ScalarPanel(list(utility_rows), **expectations)
    _validate_alignment(direction, utility)

    calibration_direction, calibration_utility = _draw_statistics(
        direction,
        utility,
        anchor_bootstrap,
        replicates=calibration_replicates,
        profiles_per_country=profiles_per_country,
        seed=calibration_seed,
    )
    power_direction, power_utility = _draw_statistics(
        direction,
        utility,
        anchor_bootstrap,
        replicates=power_replicates,
        profiles_per_country=profiles_per_country,
        seed=power_seed,
    )
    calibration_direction -= np.mean(calibration_direction, axis=0, keepdims=True)
    calibration_utility -= np.mean(calibration_utility, axis=0, keepdims=True)
    power_direction -= np.mean(power_direction, axis=0, keepdims=True)
    power_utility -= np.mean(power_utility, axis=0, keepdims=True)

    # Final row is the equal-provider macro. Keep beta_O and beta_diff only.
    def combine(direction_values: np.ndarray, utility_values: np.ndarray, index: int) -> np.ndarray:
        return np.column_stack(
            (direction_values[:, index, 0], direction_values[:, index, 2], utility_values[:, index])
        )

    macro_index = len(direction.providers)
    macro_calibration = combine(calibration_direction, calibration_utility, macro_index)
    macro_evaluation = combine(power_direction, power_utility, macro_index)
    macro_power = _family_design_power(
        macro_calibration,
        macro_evaluation,
        alternatives,
        alpha=familywise_alpha,
    )
    null_calibration = _family_design_power(
        macro_calibration,
        macro_evaluation,
        np.zeros(3, dtype=float),
        alpha=familywise_alpha,
    )
    provider_diagnostics = {
        provider: _family_design_power(
            combine(calibration_direction, calibration_utility, index),
            combine(power_direction, power_utility, index),
            alternatives,
            alpha=familywise_alpha,
        )
        for index, provider in enumerate(direction.providers)
    }
    variance_sensitivity: dict[str, ProspectiveSensitivityScenario] = {}
    for multiplier in variance_inflation_factors:
        scale = float(np.sqrt(multiplier))
        scenario_calibration = macro_calibration * scale
        scenario_evaluation = macro_evaluation * scale
        scenario_power = _family_design_power(
            scenario_calibration,
            scenario_evaluation,
            alternatives,
            alpha=familywise_alpha,
        )
        scenario_null = _family_design_power(
            scenario_calibration,
            scenario_evaluation,
            np.zeros(3, dtype=float),
            alpha=familywise_alpha,
        )
        variance_sensitivity[f"variance_x{multiplier:g}"] = ProspectiveSensitivityScenario(
            providers=direction.providers,
            excluded_provider=None,
            variance_multiplier=multiplier,
            standard_deviation_multiplier=scale,
            macro_power=scenario_power,
            null_calibration=scenario_null,
            reaches_power_target=scenario_power.joint_holm_power >= power_target,
        )

    dropout_sensitivity: dict[str, ProspectiveSensitivityScenario] = {}
    if len(direction.providers) < 2:
        raise ValueError("Single-provider dropout sensitivity requires at least two providers")
    for excluded_index, excluded_provider in enumerate(direction.providers):
        retained_indices = [
            index for index in range(len(direction.providers)) if index != excluded_index
        ]
        dropout_calibration = np.column_stack(
            (
                np.mean(calibration_direction[:, retained_indices, 0], axis=1),
                np.mean(calibration_direction[:, retained_indices, 2], axis=1),
                np.mean(calibration_utility[:, retained_indices], axis=1),
            )
        )
        dropout_evaluation = np.column_stack(
            (
                np.mean(power_direction[:, retained_indices, 0], axis=1),
                np.mean(power_direction[:, retained_indices, 2], axis=1),
                np.mean(power_utility[:, retained_indices], axis=1),
            )
        )
        dropout_power = _family_design_power(
            dropout_calibration,
            dropout_evaluation,
            alternatives,
            alpha=familywise_alpha,
        )
        dropout_null = _family_design_power(
            dropout_calibration,
            dropout_evaluation,
            np.zeros(3, dtype=float),
            alpha=familywise_alpha,
        )
        retained_providers = tuple(
            provider
            for index, provider in enumerate(direction.providers)
            if index != excluded_index
        )
        dropout_sensitivity[excluded_provider] = ProspectiveSensitivityScenario(
            providers=retained_providers,
            excluded_provider=excluded_provider,
            variance_multiplier=1.0,
            standard_deviation_multiplier=1.0,
            macro_power=dropout_power,
            null_calibration=dropout_null,
            reaches_power_target=dropout_power.joint_holm_power >= power_target,
        )
    covariance = np.cov(macro_calibration, rowvar=False, ddof=1)
    correlation = np.corrcoef(macro_calibration, rowvar=False)
    label_count = len(direction.labels)
    original_cell_counts = {
        len(np.flatnonzero(direction.displayed[country] == label_index))
        for country in direction.countries
        for label_index in range(label_count)
    }
    if len(original_cell_counts) != 1:
        raise ValueError(
            f"Pilot country-label cells have unequal counts: {original_cell_counts}"
        )
    return ProspectiveDesignPowerResult(
        providers=direction.providers,
        profiles_per_country=profiles_per_country,
        profiles_total=profiles_per_country * len(direction.countries),
        displayed_labels_per_country=label_count,
        pilot_profiles_per_country_label=next(iter(original_cell_counts)),
        planned_profiles_per_country_label=profiles_per_country // label_count,
        calibration_replicates=calibration_replicates,
        power_replicates=power_replicates,
        calibration_seed=calibration_seed,
        power_seed=power_seed,
        familywise_alpha=familywise_alpha,
        power_target=power_target,
        null={"beta_o": 0.0, "beta_diff": 0.0, "utility_v": 0.0},
        sesoi_alternative={
            "beta_o": beta_o_alternative,
            "beta_diff": beta_diff_alternative,
            "utility_v": h2_alternative,
        },
        macro_error_covariance=covariance.tolist(),
        macro_error_correlation=correlation.tolist(),
        macro_standard_errors={
            "beta_o": float(np.std(macro_calibration[:, 0], ddof=1)),
            "beta_diff": float(np.std(macro_calibration[:, 1], ddof=1)),
            "utility_v": float(np.std(macro_calibration[:, 2], ddof=1)),
        },
        macro_power=macro_power,
        null_calibration=null_calibration,
        provider_diagnostics=provider_diagnostics,
        variance_inflation_sensitivity=variance_sensitivity,
        single_provider_dropout_sensitivity=dropout_sensitivity,
        # The paper treats H1 and H2 as co-primary, so successful design power
        # requires both Holm-adjusted family decisions in the same replicate.
        primary_gate_passes=macro_power.joint_holm_power >= power_target,
        identifiable_assignment_power=False,
    )


# Backward import compatibility. The old interpretation is intentionally gone.
simulate_confirmatory_power = simulate_confirmatory_sensitivity
