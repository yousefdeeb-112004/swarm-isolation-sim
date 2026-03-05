"""
Section 15 — Publication-Quality Visualization.

Generates journal-ready figures from sweep results and statistical analysis:

  1. plot_sweep_comparison     — grouped bar chart with error bars across conditions
  2. plot_kaplan_meier         — survival curves with confidence bands
  3. plot_dose_response        — fitness impact vs isolation parameter
  4. plot_extinction_heatmap   — heatmap of extinction rates across 2 parameters
  5. plot_effect_sizes         — forest plot of Cohen's d across conditions
  6. plot_fitness_trajectories — per-generation fitness for ctrl vs treat
  7. generate_publication_figures — one-call generation of all figures

Style: high-resolution (300 DPI), consistent typography, colorblind-safe
palette, proper axis labels, legends, and error bars for all measures.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Any, Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np


# ======================================================================
# Style Configuration — journal-quality defaults
# ======================================================================

# Colorblind-safe palette (Wong 2011 / IBM Design)
PALETTE = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "cyan":    "#56B4E9",
    "yellow":  "#F0E442",
    "grey":    "#999999",
}

CONDITION_COLORS = [
    PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
    PALETTE["red"], PALETTE["purple"], PALETTE["cyan"],
]

CTRL_COLOR = PALETTE["blue"]
TREAT_COLOR = PALETTE["red"]

DPI = 300
FONT_SIZE = 10
TITLE_SIZE = 12
LABEL_SIZE = 10

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def _save(fig: plt.Figure, filepath: Optional[str]) -> None:
    """Save figure if path provided, then close."""
    if filepath:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        fig.savefig(filepath, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)


# ======================================================================
# 1. Sweep Comparison Bar Chart
# ======================================================================

def plot_sweep_comparison(
    sweep_summary: Dict[str, Any],
    metric: str = "fitness_impact",
    title: str = "Fitness Impact Across Conditions",
    ylabel: str = "Fitness Impact (Treatment - Control)",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Grouped bar chart with error bars comparing conditions.

    Parameters
    ----------
    sweep_summary : output of SweepRunner.get_summary()
    metric : which metric to plot. Supports:
             'fitness_impact', 'food_reduction', 'extinction_rate'
    """
    names = sorted(sweep_summary.keys())
    n = len(names)

    means = []
    ci95s = []
    for name in names:
        s = sweep_summary[name]
        if metric == "fitness_impact":
            means.append(s.get("fitness_impact_mean", 0))
            ci95s.append(s.get("fitness_impact_ci95", 0))
        elif metric == "food_reduction":
            means.append(s.get("food_reduction_mean", 0))
            ci95s.append(1.96 * s.get("food_reduction_std", 0) /
                         max(math.sqrt(s.get("n_seeds", 1)), 1))
        elif metric == "extinction_rate":
            means.append(s.get("extinction_rate", 0) * 100)
            ci95s.append(0)  # proportion, no CI bar
        else:
            means.append(0)
            ci95s.append(0)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n)
    colors = [CONDITION_COLORS[i % len(CONDITION_COLORS)] for i in range(n)]

    bars = ax.bar(x, means, yerr=ci95s, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5,
                  error_kw={"linewidth": 1.2, "capthick": 1.2})

    # Zero line for impact plots
    if metric in ("fitness_impact",):
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    # Labels
    display_names = [n.replace("_", " ").title() for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")

    # Annotate values on bars
    for bar, mean, ci in zip(bars, means, ci95s):
        y_pos = bar.get_height() + ci + abs(max(means)) * 0.02
        if mean < 0:
            y_pos = bar.get_height() - ci - abs(min(means)) * 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{mean:.3f}", ha="center", va="bottom" if mean >= 0 else "top",
                fontsize=8, fontweight="bold")

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 2. Kaplan–Meier Survival Curves
# ======================================================================

def plot_kaplan_meier(
    km_curves: Dict[str, Dict[str, Any]],
    title: str = "Swarm Survival (Kaplan–Meier)",
    xlabel: str = "Simulation Step",
    ylabel: str = "Survival Probability",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Plot Kaplan–Meier survival curves with confidence bands.

    Parameters
    ----------
    km_curves : dict of {condition_name: kaplan_meier() output}
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, km) in enumerate(sorted(km_curves.items())):
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)]
        times = km.get("times", [])
        survival = km.get("survival", [])
        ci_lo = km.get("ci_lower", [])
        ci_hi = km.get("ci_upper", [])

        if not times:
            continue

        display_name = name.replace("_", " ").title()
        median = km.get("median_survival")
        label = f"{display_name}"
        if median is not None:
            label += f" (median={median})"

        # Step function
        ax.step(times, survival, where="post", color=color,
                linewidth=1.8, label=label)

        # Confidence band
        if len(ci_lo) == len(times) and len(ci_hi) == len(times):
            ax.fill_between(times, ci_lo, ci_hi,
                            step="post", alpha=0.15, color=color)

    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if any(km.get("times") for km in km_curves.values()):
        ax.legend(loc="best", framealpha=0.9)
    ax.axhline(0.5, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 3. Dose-Response Curve
# ======================================================================

def plot_dose_response(
    sweep_summary: Dict[str, Any],
    param_key: str = "isolation_fraction",
    metric: str = "fitness_impact",
    title: str = "Dose-Response: Isolation Fraction vs Fitness Impact",
    xlabel: str = "Isolation Fraction",
    ylabel: str = "Fitness Impact",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
) -> plt.Figure:
    """
    Line plot showing metric vs parameter value (dose-response).

    Extracts the parameter value from each condition's config.
    """
    points = []
    for name, s in sweep_summary.items():
        cond = s.get("condition", {})
        param_val = cond.get(param_key, None)
        if param_val is None:
            continue

        if metric == "fitness_impact":
            y = s.get("fitness_impact_mean", 0)
            yerr = s.get("fitness_impact_ci95", 0)
        elif metric == "food_reduction":
            y = s.get("food_reduction_mean", 0)
            n_seeds = s.get("n_seeds", 1)
            yerr = 1.96 * s.get("food_reduction_std", 0) / max(math.sqrt(n_seeds), 1)
        elif metric == "extinction_rate":
            y = s.get("extinction_rate", 0) * 100
            yerr = 0
        else:
            y, yerr = 0, 0

        points.append((param_val, y, yerr, name))

    points.sort(key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=figsize)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    errs = [p[2] for p in points]

    ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=PALETTE["blue"],
                linewidth=2, markersize=8, capsize=5, capthick=1.5,
                markerfacecolor="white", markeredgewidth=2)

    # Annotate points
    for x, y, err, name in points:
        ax.annotate(f"{y:.3f}",
                    (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)

    if metric in ("fitness_impact",):
        ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 4. Extinction Heatmap
# ======================================================================

def plot_extinction_heatmap(
    data: Dict[Tuple[str, str], float],
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Extinction Rate (%)",
    row_title: str = "Condition A",
    col_title: str = "Condition B",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
) -> plt.Figure:
    """
    Heatmap of extinction rates across two parameter dimensions.

    Parameters
    ----------
    data : dict mapping (row_label, col_label) → extinction rate [0-1]
    """
    nr, nc = len(row_labels), len(col_labels)
    matrix = np.zeros((nr, nc))
    for i, rl in enumerate(row_labels):
        for j, cl in enumerate(col_labels):
            matrix[i, j] = data.get((rl, cl), 0) * 100

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=100)

    ax.set_xticks(range(nc))
    ax.set_xticklabels([c.replace("_", " ") for c in col_labels], rotation=30, ha="right")
    ax.set_yticks(range(nr))
    ax.set_yticklabels([r.replace("_", " ") for r in row_labels])
    ax.set_xlabel(col_title)
    ax.set_ylabel(row_title)
    ax.set_title(title, fontweight="bold")

    # Annotate cells
    for i in range(nr):
        for j in range(nc):
            val = matrix[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Extinction Rate (%)", shrink=0.8)
    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 5. Effect Size Forest Plot
# ======================================================================

def plot_effect_sizes(
    analysis: Dict[str, Any],
    title: str = "Effect Sizes (Cohen's d): Control vs Treatment",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Forest plot of Cohen's d effect sizes from t-test results.
    """
    t_tests = analysis.get("t_tests", {})
    names = sorted(t_tests.keys())
    n = len(names)

    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        _save(fig, filepath)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    ds = [t_tests[name]["cohens_d"] for name in names]
    sigs = [t_tests[name]["significant_005"] for name in names]
    display_names = [n.replace("_", " ").title() for n in names]

    y_pos = np.arange(n)
    colors = [PALETTE["red"] if sig else PALETTE["grey"] for sig in sigs]

    ax.barh(y_pos, ds, color=colors, edgecolor="white",
            height=0.6, linewidth=0.5)

    # Effect size thresholds
    for threshold, label in [(0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
        ax.axvline(threshold, color="grey", linewidth=0.6, linestyle=":",
                   alpha=0.5)
        ax.axvline(-threshold, color="grey", linewidth=0.6, linestyle=":",
                   alpha=0.5)
    ax.axvline(0, color="black", linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Cohen's d (positive = ctrl > treat)")
    ax.set_title(title, fontweight="bold")

    # Annotate
    for i, (d, sig) in enumerate(zip(ds, sigs)):
        marker = " ***" if t_tests[names[i]]["significant_001"] else (
            " *" if sig else "")
        ax.text(d + 0.02 * (1 if d >= 0 else -1), i,
                f"{d:.2f}{marker}", va="center",
                ha="left" if d >= 0 else "right",
                fontsize=8, fontweight="bold")

    # Legend
    sig_patch = mpatches.Patch(color=PALETTE["red"], label="Significant (p<0.05)")
    ns_patch = mpatches.Patch(color=PALETTE["grey"], label="Not significant")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right", framealpha=0.9)

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 6. Fitness Trajectories (per-generation)
# ======================================================================

def plot_fitness_trajectories(
    sweep_results: Dict[str, Any],
    conditions: Optional[List[str]] = None,
    title: str = "Fitness Trajectories: Control vs Treatment",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Per-generation fitness for control and treatment, averaged across seeds.

    Shows mean ± SE bands for each condition.
    """
    all_results = sweep_results.get("all_results", [])

    # Group by condition
    by_cond: Dict[str, List[Dict]] = {}
    for r in all_results:
        name = r["condition"]["name"]
        if conditions and name not in conditions:
            continue
        by_cond.setdefault(name, []).append(r)

    if not by_cond:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        _save(fig, filepath)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, results) in enumerate(sorted(by_cond.items())):
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)]
        display = name.replace("_", " ").title()

        # Get max generation count
        max_gens = max(len(r.get("ctrl_gen_records", [])) for r in results)

        # Aggregate control and treatment fitness per generation
        for group, key, style in [("ctrl", "ctrl_gen_records", "-"),
                                   ("treat", "treat_gen_records", "--")]:
            gen_vals: Dict[int, List[float]] = {}
            for r in results:
                for rec in r.get(key, []):
                    g = rec.get("generation", 0)
                    gen_vals.setdefault(g, []).append(rec.get("avg_fitness", 0))

            gens = sorted(gen_vals.keys())
            means = [np.mean(gen_vals[g]) for g in gens]
            sems = [np.std(gen_vals[g], ddof=1) / max(math.sqrt(len(gen_vals[g])), 1)
                    for g in gens]

            label = f"{display} ({'Control' if group == 'ctrl' else 'Treatment'})"
            ax.plot(gens, means, style, color=color, linewidth=1.5, label=label)
            lower = [m - s for m, s in zip(means, sems)]
            upper = [m + s for m, s in zip(means, sems)]
            ax.fill_between(gens, lower, upper, alpha=0.12, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Fitness")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 7. Multi-Panel Summary Dashboard
# ======================================================================

def plot_summary_dashboard(
    sweep_results: Dict[str, Any],
    analysis: Dict[str, Any],
    title: str = "Experiment Summary",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    4-panel summary figure combining key visualizations.

    Panels:
      (A) Fitness impact bar chart
      (B) Food reduction bar chart
      (C) Effect sizes forest plot
      (D) Fitness trajectories
    """
    summary = sweep_results.get("sweep_summary", {})
    names = sorted(summary.keys())
    n = len(names)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel A: Fitness Impact ---
    ax_a = fig.add_subplot(gs[0, 0])
    _bar_panel(ax_a, summary, names, "fitness_impact_mean", "fitness_impact_ci95",
               "A. Fitness Impact", "Fitness Impact")

    # --- Panel B: Food Reduction ---
    ax_b = fig.add_subplot(gs[0, 1])
    food_means = [summary[n].get("food_reduction_mean", 0) for n in names]
    food_stds = [summary[n].get("food_reduction_std", 0) for n in names]
    n_seeds = [summary[n].get("n_seeds", 1) for n in names]
    food_ci = [1.96 * s / max(math.sqrt(ns), 1) for s, ns in zip(food_stds, n_seeds)]

    x = np.arange(len(names))
    colors = [CONDITION_COLORS[i % len(CONDITION_COLORS)] for i in range(len(names))]
    ax_b.bar(x, food_means, yerr=food_ci, capsize=3,
             color=colors, edgecolor="white", linewidth=0.5,
             error_kw={"linewidth": 1})
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([n.replace("_", " ") for n in names],
                         rotation=30, ha="right", fontsize=8)
    ax_b.set_ylabel("Food Reduction (%)")
    ax_b.set_title("B. Food Acquisition Reduction", fontweight="bold")

    # --- Panel C: Effect Sizes ---
    ax_c = fig.add_subplot(gs[1, 0])
    t_tests = analysis.get("t_tests", {})
    t_names = sorted(t_tests.keys())
    if t_names:
        ds = [t_tests[tn]["cohens_d"] for tn in t_names]
        sigs = [t_tests[tn]["significant_005"] for tn in t_names]
        y_pos = np.arange(len(t_names))
        bar_colors = [PALETTE["red"] if s else PALETTE["grey"] for s in sigs]
        ax_c.barh(y_pos, ds, color=bar_colors, height=0.5, edgecolor="white")
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels([n.replace("_", " ") for n in t_names], fontsize=8)
        ax_c.axvline(0, color="black", linewidth=0.8)
        for thresh in [0.2, 0.5, 0.8]:
            ax_c.axvline(thresh, color="grey", linewidth=0.5, linestyle=":", alpha=0.4)
            ax_c.axvline(-thresh, color="grey", linewidth=0.5, linestyle=":", alpha=0.4)
    ax_c.set_xlabel("Cohen's d")
    ax_c.set_title("C. Effect Sizes", fontweight="bold")

    # --- Panel D: Fitness Trajectories ---
    ax_d = fig.add_subplot(gs[1, 1])
    all_results = sweep_results.get("all_results", [])
    by_cond: Dict[str, List] = {}
    for r in all_results:
        cname = r["condition"]["name"]
        by_cond.setdefault(cname, []).append(r)

    for i, (cname, results) in enumerate(sorted(by_cond.items())):
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)]
        for group, key, style in [("ctrl", "ctrl_gen_records", "-"),
                                   ("treat", "treat_gen_records", "--")]:
            gen_vals: Dict[int, List[float]] = {}
            for r in results:
                for rec in r.get(key, []):
                    g = rec.get("generation", 0)
                    gen_vals.setdefault(g, []).append(rec.get("avg_fitness", 0))
            gens = sorted(gen_vals.keys())
            means = [np.mean(gen_vals[g]) for g in gens]
            lbl = f"{cname.replace('_',' ')} {'C' if group=='ctrl' else 'T'}"
            ax_d.plot(gens, means, style, color=color, linewidth=1.2, label=lbl)

    ax_d.set_xlabel("Generation")
    ax_d.set_ylabel("Avg Fitness")
    ax_d.set_title("D. Fitness Trajectories", fontweight="bold")
    if len(by_cond) <= 4:
        ax_d.legend(loc="best", fontsize=6, ncol=2, framealpha=0.8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    _save(fig, filepath)
    return fig


def _bar_panel(ax, summary, names, mean_key, ci_key, title, ylabel):
    """Helper for a single bar panel in the dashboard."""
    means = [summary[n].get(mean_key, 0) for n in names]
    cis = [summary[n].get(ci_key, 0) for n in names]
    x = np.arange(len(names))
    colors = [CONDITION_COLORS[i % len(CONDITION_COLORS)] for i in range(len(names))]
    ax.bar(x, means, yerr=cis, capsize=3,
           color=colors, edgecolor="white", linewidth=0.5,
           error_kw={"linewidth": 1})
    if any(m < 0 for m in means):
        ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")


# ======================================================================
# 8. Main Entry Point
# ======================================================================

def generate_publication_figures(
    sweep_results: Dict[str, Any],
    analysis: Dict[str, Any],
    output_dir: str = "data/figures",
    prefix: str = "pub",
) -> Dict[str, str]:
    """
    Generate all publication-quality figures.

    Returns dict of {figure_name: filepath}.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = sweep_results.get("sweep_summary", {})
    km_curves = analysis.get("kaplan_meier", {})

    generated = {}

    # 1. Fitness impact comparison
    path = os.path.join(output_dir, f"{prefix}_fitness_impact.png")
    plot_sweep_comparison(summary, metric="fitness_impact",
                          title="Fitness Impact Across Conditions",
                          filepath=path)
    generated["fitness_impact"] = path

    # 2. Food reduction comparison
    path = os.path.join(output_dir, f"{prefix}_food_reduction.png")
    plot_sweep_comparison(summary, metric="food_reduction",
                          title="Food Acquisition Reduction",
                          ylabel="Food Reduction (%)",
                          filepath=path)
    generated["food_reduction"] = path

    # 3. Extinction rate comparison
    path = os.path.join(output_dir, f"{prefix}_extinction_rate.png")
    plot_sweep_comparison(summary, metric="extinction_rate",
                          title="Extinction Rate Across Conditions",
                          ylabel="Extinction Rate (%)",
                          filepath=path)
    generated["extinction_rate"] = path

    # 4. Kaplan-Meier survival
    if km_curves:
        path = os.path.join(output_dir, f"{prefix}_kaplan_meier.png")
        plot_kaplan_meier(km_curves, filepath=path)
        generated["kaplan_meier"] = path

    # 5. Effect sizes
    path = os.path.join(output_dir, f"{prefix}_effect_sizes.png")
    plot_effect_sizes(analysis, filepath=path)
    generated["effect_sizes"] = path

    # 6. Fitness trajectories
    path = os.path.join(output_dir, f"{prefix}_trajectories.png")
    plot_fitness_trajectories(sweep_results, filepath=path)
    generated["trajectories"] = path

    # 7. Summary dashboard
    path = os.path.join(output_dir, f"{prefix}_dashboard.png")
    plot_summary_dashboard(sweep_results, analysis, filepath=path)
    generated["dashboard"] = path

    # 8. Dose-response if applicable (try common parameter keys)
    for param, xlabel in [("isolation_fraction", "Isolation Fraction"),
                           ("isolation_duration", "Isolation Duration (steps)"),
                           ("steps_per_generation", "Steps per Generation")]:
        # Check if conditions vary this parameter
        vals = set()
        for s in summary.values():
            cond = s.get("condition", {})
            v = cond.get(param)
            if v is not None:
                vals.add(v)
        if len(vals) >= 3:  # Only plot if 3+ distinct values
            path = os.path.join(output_dir, f"{prefix}_dose_{param}.png")
            plot_dose_response(summary, param_key=param,
                               title=f"Dose-Response: {xlabel}",
                               xlabel=xlabel, filepath=path)
            generated[f"dose_{param}"] = path

    return generated
