#!/usr/bin/env python3
"""Render the post-hoc model-by-country result figure from the final CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "api_pilot/outputs/final_tables/table_9_model_country_preference_metrics.csv"
CONTRAST_INPUT = ROOT / "api_pilot/outputs/final_tables/table_16_targeted_country_contrasts.csv"
PNG = ROOT / "api_pilot/outputs/figures/model_country_preference_patterns.png"
PDF = ROOT / "api_pilot/outputs/figures/model_country_preference_patterns.pdf"
PROVIDERS = ("deepseek", "gemini", "mistral", "openai", "qwen")
MODEL_LABELS = ("DeepSeek", "Gemini", "Mistral", "OpenAI", "Qwen")
COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")


def load_matrix(metric: str, field: str) -> np.ndarray:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        (row["provider"], row["country_code"]): float(row[field])
        for row in rows
        if row["view"] == "clean_hybrid_all20" and row["metric"] == metric
    }
    expected = {(provider, country) for provider in PROVIDERS for country in COUNTRIES}
    if set(lookup) != expected:
        raise ValueError(f"Incomplete final matrix for {metric}/{field}")
    return np.asarray(
        [[lookup[(provider, country)] for country in COUNTRIES] for provider in PROVIDERS]
    )


def load_mistral_france_summary() -> tuple[float, float, float, float, float]:
    with CONTRAST_INPUT.open(encoding="utf-8", newline="") as handle:
        contrast = next(
            row
            for row in csv.DictReader(handle)
            if row["view"] == "clean_hybrid_all20"
            and row["provider"] == "mistral"
            and row["metric"] == "mo_tv"
            and row["contrast"] == "France_minus_mean_other_countries"
        )
    with INPUT.open(encoding="utf-8", newline="") as handle:
        residual = next(
            row
            for row in csv.DictReader(handle)
            if row["view"] == "clean_hybrid_all20"
            and row["provider"] == "mistral"
            and row["metric"] == "mo_tv"
            and row["country_code"] == "FR"
        )
    return (
        float(contrast["estimate"]),
        float(contrast["ci_low"]),
        float(contrast["ci_high"]),
        float(residual["interaction_estimate"]),
        float(residual["interaction_max_t_p"]),
    )


def heatmap(ax: plt.Axes, matrix: np.ndarray, title: str, unit: str) -> None:
    bound = float(np.max(np.abs(matrix)))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(COUNTRIES)), COUNTRIES)
    ax.set_yticks(range(len(MODEL_LABELS)), MODEL_LABELS)
    ax.set_title(title, loc="left", fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            colour = "white" if abs(value) > bound * 0.58 else "#202124"
            ax.text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=8,
                fontweight="bold" if (row == 2 or column == 3) else "normal",
            )
    ax.add_patch(
        plt.Rectangle((2.5, -0.5), 1, len(PROVIDERS), fill=False, edgecolor="#202124", linewidth=1.4)
    )
    ax.add_patch(
        plt.Rectangle((-0.5, 1.5), len(COUNTRIES), 1, fill=False, edgecolor="#202124", linewidth=1.4)
    )
    bar = ax.figure.colorbar(image, ax=ax, shrink=0.78, pad=0.025)
    bar.set_label(unit)


def main() -> int:
    tv = load_matrix("mo_tv", "estimate")
    tv_interaction = load_matrix("mo_tv", "interaction_estimate")
    anchor_interaction = load_matrix("m0_anchor_l2", "interaction_estimate")
    brier_interaction = load_matrix("m0_brier", "interaction_estimate")
    raw_france, raw_low, raw_high, residual_france, residual_p = (
        load_mistral_france_summary()
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(11.2, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1))

    ax = fig.add_subplot(grid[0, 0])
    colours = plt.cm.Set2(np.linspace(0, 1, len(PROVIDERS)))
    x = np.arange(len(COUNTRIES))
    for index, label in enumerate(MODEL_LABELS):
        ax.plot(
            x,
            tv[index],
            marker="s" if label == "Mistral" else "o",
            linewidth=2.4 if label == "Mistral" else 1.5,
            markersize=5,
            label=label,
            color=colours[index],
        )
    ax.axvspan(2.8, 3.2, color="#6b7280", alpha=0.12, linewidth=0)
    ax.set_xticks(x, COUNTRIES)
    ax.set_ylim(0, max(tv.ravel()) * 1.12)
    ax.set_ylabel("Mean TV(MO, M0)")
    ax.set_title("A  Absolute response to each random label", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    p_label = "<.001" if residual_p < 0.001 else f"={residual_p:.3f}"
    ax.text(
        0.03,
        0.045,
        "Mistral, France:\n"
        f"raw FR - other labels {raw_france:+.3f} "
        f"(95% CI {raw_low:+.3f}, {raw_high:+.3f})\n"
        f"model x label residual {residual_france:+.3f} (max-T p{p_label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="#374151",
    )

    heatmap(
        fig.add_subplot(grid[0, 1]),
        tv_interaction,
        "B  TV model-by-country residual",
        "two-way centred TV",
    )
    heatmap(
        fig.add_subplot(grid[1, 0]),
        anchor_interaction,
        "C  Baseline no-label anchor-distance residual",
        "two-way centred squared L2",
    )
    heatmap(
        fig.add_subplot(grid[1, 1]),
        brier_interaction,
        "D  True-country Brier residual",
        "two-way centred Brier",
    )
    fig.suptitle(
        "Post-hoc exploratory model-country heterogeneity in the clean replacement panel",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.002,
        "Original 504-profile results for 12 correct items plus 72-profile corrected replacements for 8 items. "
        "Outlined row: Mistral; outlined column: France. Residuals remove model and country main effects.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=220, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PNG.relative_to(ROOT)} and {PDF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
