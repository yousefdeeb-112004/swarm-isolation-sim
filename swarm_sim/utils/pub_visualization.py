"""
Section 15 — Publication-Quality Visualization.

Generates journal-ready figures from sweep results and statistical analyses:

  1. Condition comparison bar charts with error bars and significance
  2. Kaplan–Meier survival curves with confidence bands
  3. Forest plot of effect sizes (Cohen's d) across conditions
  4. Sweep summary heatmap (conditions × metrics)
  5. Per-generation fitness trajectories (control vs treatment)
  6. Multi-panel summary figure combining key results
  7. Extinction time distribution (violin/box plots)

All figures:
  - 300 DPI for print quality
  - Consistent color palette and typography
  - Proper axis labels, legends, and titles
  - Ready for direct inclusion in LaTeX / Word manuscripts

Depends on: SweepRunner results, BatchLogger CSVs, stats_analysis outputs.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ======================================================================
# Style Configuration
# ======================================================================

# Consistent publication palette
COLORS = {
    "control": "#2171B5",      # Blue
    "treatment": "#CB181D",    # Red
    "neutral": "#636363",      # Gray
    "highlight": "#238B45",    # Green
    "accent1": "#6A51A3",      # Purple
    "accent2": "#D94801",      # Orange
    "ci_band": "#BDBDBD",      # Light gray for CIs
}

# Condition palette for sweep comparisons
CONDITION_COLORS = [
    "#2171B5", "#CB181D", "#238B45", "#D94801",
    "#6A51A3", "#E6550D", "#31A354", "#756BB1",
]

PUB_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style():
    """Apply publication matplotlib style."""
    if HAS_MPL:
        plt.rcParams.update(PUB_STYLE)


def _save(fig: plt.Figure, path: Optional[str], dpi: int = 300) -> None:
    """Save figure at publication DPI."""
    if path and HAS_MPL:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ======================================================================
# 1. Condition Comparison Bar Chart with Error Bars
# ======================================================================

def plot_condition_comparison(
    summary: Dict[str, Any],
    metric: str = "fitness_impact",
    title: str = "Impact of Isolation Across Conditions",
    ylabel: str = "Fitness Impact",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> Optional[plt.Figure]:
    """
    Bar chart comparing a metric across sweep conditions.

    Each bar shows the mean with 95% CI error bars.
    Significance stars are added based on CI crossing zero.
    """
    if not HAS_MPL:
        return None
    _apply_style()

    names = sorted(summary.keys())
    means = []
    ci95s = []

    metric_mean_key = f"{metric}_mean" if f"{metric}_mean" in summary.get(names[0], {}) else "fitness_impact_mean"
    metric_ci_key = f"{metric}_ci95" if f"{metric}_ci95" in summary.get(names[0], {}) else "fitness_impact_ci95"

    for name in names:
        s = summary[name]
        means.append(s.get(metric_mean_key, 0))
        ci95s.append(s.get(metric_ci_key, 0))

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(names))
    colors = [CONDITION_COLORS[i % len(CONDITION_COLORS)] for i in range(len(names))]
    bars = ax.bar(x, means, yerr=ci95s, capsize=4,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
                  error_kw={"linewidth": 1.0, "capthick": 1.0})

    # Add significance indicators
    for i, (m, ci) in enumerate(zip(means, ci95s)):
        if abs(m) > ci and ci > 0:  # CI doesn't cross zero
            ax.text(i, m + ci + abs(m) * 0.05 + 0.01, "***",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

    ax.set_xticks(x)
    labels = [n.replace("_", "\n") for n in names]
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 2. Kaplan–Meier Survival Curves
# ======================================================================

def plot_kaplan_meier(
    km_results: Dict[str, Dict[str, Any]],
    title: str = "Swarm Survival Under Isolation",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> Optional[plt.Figure]:
    """
    Plot KM survival curves for multiple conditions with CI bands.

    Parameters
    ----------
    km_results : {condition_name: kaplan_meier() output}
    """
    if not HAS_MPL:
        return None
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, km) in enumerate(sorted(km_results.items())):
        times = km["times"]
        surv = km["survival"]
        ci_lo = km["ci_lower"]
        ci_hi = km["ci_upper"]
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)]
        label = name.replace("_", " ")

        if len(times) < 2:
            # No events — flat line at 1.0
            ax.axhline(y=1.0, color=color, linestyle="--", alpha=0.6, label=f"{label} (no events)")
            continue

        # Step function
        ax.step(times, surv, where="post", color=color, linewidth=1.5, label=label)

        # CI band
        ax.fill_between(times, ci_lo, ci_hi, step="post",
                        color=color, alpha=0.15)

        # Median marker
        if km.get("median_survival") is not None:
            med = km["median_survival"]
            ax.plot(med, 0.5, "o", color=color, markersize=5)
            ax.annotate(f"  median={med}", (med, 0.5), fontsize=7, color=color)

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 3. Forest Plot of Effect Sizes
# ======================================================================

def plot_forest(
    t_tests: Dict[str, Dict[str, Any]],
    title: str = "Effect Sizes: Control vs Treatment Fitness",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> Optional[plt.Figure]:
    """
    Forest plot showing Cohen's d with CIs for each condition.

    Horizontal bars centered on the effect size with confidence intervals.
    """
    if not HAS_MPL:
        return None
    _apply_style()

    names = sorted(t_tests.keys())
    ds = [t_tests[n]["cohens_d"] for n in names]
    # Approximate CI for d (using SE ≈ sqrt(2/n) simplified)
    ns = [t_tests[n]["ctrl"]["n"] + t_tests[n]["treat"]["n"] for n in names]
    ci_widths = [1.96 * math.sqrt(2 / max(n, 2) + d**2 / (2 * max(n, 2)))
                 for d, n in zip(ds, ns)]

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(names))
    for i, (d, ci, name) in enumerate(zip(ds, ci_widths, names)):
        color = COLORS["treatment"] if d < 0 else COLORS["control"]
        ax.errorbar(d, i, xerr=ci, fmt="o", color=color,
                    capsize=4, markersize=6, linewidth=1.2)

        # Significance label
        t = t_tests[name]
        if t["significant_001"]:
            ax.text(d + ci + 0.05, i, "***", va="center", fontsize=8)
        elif t["significant_005"]:
            ax.text(d + ci + 0.05, i, "*", va="center", fontsize=8)

    # Zero line
    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")

    # Effect size reference zones
    for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline(x=-thresh, color=COLORS["ci_band"], linewidth=0.3, linestyle=":")
        ax.axvline(x=thresh, color=COLORS["ci_band"], linewidth=0.3, linestyle=":")

    ax.set_yticks(y)
    labels = [n.replace("_", " ") for n in names]
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_title(title)
    ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 4. Sweep Summary Heatmap
# ======================================================================

def plot_sweep_heatmap(
    summary: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    title: str = "Sweep Results Heatmap",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4),
) -> Optional[plt.Figure]:
    """
    Heatmap of key metrics across conditions.

    Rows = conditions, columns = metrics, cells colored by normalized value.
    """
    if not HAS_MPL:
        return None
    _apply_style()

    if metrics is None:
        metrics = [
            "fitness_impact_mean", "extinction_rate",
            "food_reduction_mean", "ctrl_fitness_mean", "treat_fitness_mean",
        ]

    names = sorted(summary.keys())
    display_metrics = [m.replace("_mean", "").replace("_", " ") for m in metrics]

    # Build matrix
    data = np.zeros((len(names), len(metrics)))
    for i, name in enumerate(names):
        for j, m in enumerate(metrics):
            data[i, j] = summary[name].get(m, 0) or 0

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize each column independently for color mapping
    normed = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            normed[:, j] = (col - cmin) / (cmax - cmin)
        else:
            normed[:, j] = 0.5

    im = ax.imshow(normed, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)

    # Annotate cells with actual values
    for i in range(len(names)):
        for j in range(len(metrics)):
            val = data[i, j]
            # Format: percentage for rates/reductions, decimal for others
            if "rate" in metrics[j] or "reduction" in metrics[j]:
                txt = f"{val:.0f}%"
            else:
                txt = f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if normed[i, j] > 0.6 else "black")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(display_metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    labels = [n.replace("_", " ") for n in names]
    ax.set_yticklabels(labels)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Relative magnitude")

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 5. Per-Generation Fitness Trajectories
# ======================================================================

def plot_fitness_trajectories(
    sweep_results: Dict[str, Any],
    condition_name: Optional[str] = None,
    title: str = "Fitness Across Generations",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> Optional[plt.Figure]:
    """
    Line plot of mean fitness per generation for control and treatment,
    averaged across seeds with shaded ±1 SD bands.

    If condition_name is None, uses all results.
    """
    if not HAS_MPL:
        return None
    _apply_style()

    all_results = sweep_results.get("all_results", [])
    if condition_name:
        results = [r for r in all_results if r["condition"]["name"] == condition_name]
    else:
        results = all_results

    if not results:
        return None

    # Collect per-generation fitnesses
    max_gens = max(len(r.get("ctrl_gen_records", [])) for r in results)
    ctrl_by_gen = [[] for _ in range(max_gens)]
    treat_by_gen = [[] for _ in range(max_gens)]

    for r in results:
        for g, rec in enumerate(r.get("ctrl_gen_records", [])):
            ctrl_by_gen[g].append(rec.get("avg_fitness", 0))
        for g, rec in enumerate(r.get("treat_gen_records", [])):
            treat_by_gen[g].append(rec.get("avg_fitness", 0))

    gens = list(range(max_gens))
    ctrl_mean = [np.mean(v) if v else 0 for v in ctrl_by_gen]
    ctrl_std = [np.std(v) if len(v) > 1 else 0 for v in ctrl_by_gen]
    treat_mean = [np.mean(v) if v else 0 for v in treat_by_gen]
    treat_std = [np.std(v) if len(v) > 1 else 0 for v in treat_by_gen]

    fig, ax = plt.subplots(figsize=figsize)

    # Control
    ax.plot(gens, ctrl_mean, color=COLORS["control"], linewidth=1.5, label="Control")
    ax.fill_between(gens,
                     [m - s for m, s in zip(ctrl_mean, ctrl_std)],
                     [m + s for m, s in zip(ctrl_mean, ctrl_std)],
                     color=COLORS["control"], alpha=0.2)

    # Treatment
    ax.plot(gens, treat_mean, color=COLORS["treatment"], linewidth=1.5, label="Treatment")
    ax.fill_between(gens,
                     [m - s for m, s in zip(treat_mean, treat_std)],
                     [m + s for m, s in zip(treat_mean, treat_std)],
                     color=COLORS["treatment"], alpha=0.2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness")
    label = condition_name.replace("_", " ") if condition_name else "All Conditions"
    ax.set_title(f"{title} ({label})")
    ax.legend(loc="best")

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 6. Extinction Time Distribution
# ======================================================================

def plot_extinction_distribution(
    sweep_results: Dict[str, Any],
    title: str = "Treatment Extinction Time Distribution",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> Optional[plt.Figure]:
    """
    Box plot of extinction times across conditions.
    """
    if not HAS_MPL:
        return None
    _apply_style()

    all_results = sweep_results.get("all_results", [])
    by_cond: Dict[str, List[float]] = {}
    for r in all_results:
        name = r["condition"]["name"]
        ext = r.get("treat_extinction_step")
        if ext is not None:
            by_cond.setdefault(name, []).append(ext)

    if not by_cond:
        return None

    names = sorted(by_cond.keys())
    data = [by_cond[n] for n in names]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(data, labels=[n.replace("_", "\n") for n in names],
                    patch_artist=True, notch=True, widths=0.6)

    for i, patch in enumerate(bp["boxes"]):
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)]
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Extinction Step")
    ax.set_title(title)

    fig.tight_layout()
    _save(fig, filepath)
    return fig


# ======================================================================
# 7. Multi-Panel Summary Figure
# ======================================================================

def plot_publication_summary(
    sweep_results: Dict[str, Any],
    analysis: Dict[str, Any],
    title: str = "Isolation Experiment: Summary of Results",
    filepath: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
) -> Optional[plt.Figure]:
    """
    4-panel summary figure combining key results.

    Panel A: Fitness impact bar chart
    Panel B: Kaplan-Meier survival curves
    Panel C: Effect size forest plot
    Panel D: Extinction time box plot
    """
    if not HAS_MPL:
        return None
    _apply_style()

    summary = sweep_results.get("sweep_summary", {})
    km_results = analysis.get("kaplan_meier", {})
    t_tests = analysis.get("t_tests", {})

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # Panel A: Fitness impact
    ax = axes[0, 0]
    names = sorted(summary.keys())
    means = [summary[n].get("fitness_impact_mean", 0) for n in names]
    ci95s = [summary[n].get("fitness_impact_ci95", 0) for n in names]
    x = np.arange(len(names))
    colors = [CONDITION_COLORS[i % len(CONDITION_COLORS)] for i in range(len(names))]
    ax.bar(x, means, yerr=ci95s, capsize=3, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.4, error_kw={"linewidth": 0.8})
    ax.axhline(y=0, color="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("Fitness Impact")
    ax.set_title("(A) Fitness Impact by Condition", fontsize=10)

    # Panel B: KM Survival
    ax = axes[0, 1]
    for i, (name, km) in enumerate(sorted(km_results.items())):
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)]
        times = km["times"]
        surv = km["survival"]
        label = name.replace("_", " ")
        if len(times) >= 2:
            ax.step(times, surv, where="post", color=color, linewidth=1.2, label=label)
            ax.fill_between(times, km["ci_lower"], km["ci_upper"],
                          step="post", color=color, alpha=0.1)
        else:
            ax.axhline(y=1.0, color=color, linestyle="--", alpha=0.5, label=f"{label}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(B) Kaplan-Meier Survival", fontsize=10)
    ax.legend(loc="best", fontsize=6, ncol=1)

    # Panel C: Forest plot
    ax = axes[1, 0]
    t_names = sorted(t_tests.keys())
    ds = [t_tests[n]["cohens_d"] for n in t_names]
    ns = [t_tests[n]["ctrl"]["n"] + t_tests[n]["treat"]["n"] for n in t_names]
    cis = [1.96 * math.sqrt(2 / max(n, 2) + d**2 / (2 * max(n, 2)))
           for d, n in zip(ds, ns)]
    y_pos = np.arange(len(t_names))
    for i, (d, ci) in enumerate(zip(ds, cis)):
        color = COLORS["treatment"] if d < 0 else COLORS["control"]
        ax.errorbar(d, i, xerr=ci, fmt="o", color=color, capsize=3,
                   markersize=5, linewidth=1.0)
    ax.axvline(x=0, color="black", linewidth=0.4, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_", " ") for n in t_names], fontsize=7)
    ax.set_xlabel("Cohen's d")
    ax.set_title("(C) Effect Sizes", fontsize=10)
    ax.invert_yaxis()

    # Panel D: Extinction rates
    ax = axes[1, 1]
    ext_rates = [summary[n].get("extinction_rate", 0) * 100 for n in names]
    food_reds = [summary[n].get("food_reduction_mean", 0) for n in names]
    width = 0.35
    ax.bar(x - width/2, ext_rates, width, color=COLORS["treatment"],
           alpha=0.7, label="Extinction %", edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, food_reds, width, color=COLORS["accent1"],
           alpha=0.7, label="Food Reduction %", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(D) Extinction & Food Reduction", fontsize=10)
    ax.legend(loc="best", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filepath)
    return fig


# ======================================================================
# 8. Main: Generate All Publication Figures
# ======================================================================

def generate_publication_figures(
    sweep_results: Dict[str, Any],
    analysis: Dict[str, Any],
    output_dir: str = "data/figures",
    prefix: str = "fig",
) -> Dict[str, str]:
    """
    Generate all publication-quality figures from sweep + analysis data.

    Returns dict mapping figure name to filepath.
    """
    if not HAS_MPL:
        return {"error": "matplotlib not available"}

    os.makedirs(output_dir, exist_ok=True)
    summary = sweep_results.get("sweep_summary", {})
    km_results = analysis.get("kaplan_meier", {})
    t_tests = analysis.get("t_tests", {})

    figures = {}

    # Fig 1: Condition comparison
    path = os.path.join(output_dir, f"{prefix}_condition_comparison.png")
    plot_condition_comparison(summary, filepath=path)
    figures["condition_comparison"] = path

    # Fig 2: KM survival
    path = os.path.join(output_dir, f"{prefix}_kaplan_meier.png")
    plot_kaplan_meier(km_results, filepath=path)
    figures["kaplan_meier"] = path

    # Fig 3: Forest plot
    path = os.path.join(output_dir, f"{prefix}_forest_plot.png")
    plot_forest(t_tests, filepath=path)
    figures["forest_plot"] = path

    # Fig 4: Heatmap
    path = os.path.join(output_dir, f"{prefix}_heatmap.png")
    plot_sweep_heatmap(summary, filepath=path)
    figures["heatmap"] = path

    # Fig 5: Fitness trajectories (first condition only, or all)
    path = os.path.join(output_dir, f"{prefix}_fitness_trajectories.png")
    plot_fitness_trajectories(sweep_results, filepath=path)
    figures["fitness_trajectories"] = path

    # Fig 6: Extinction distribution
    path = os.path.join(output_dir, f"{prefix}_extinction_distribution.png")
    fig = plot_extinction_distribution(sweep_results, filepath=path)
    if fig:
        figures["extinction_distribution"] = path

    # Fig 7: Multi-panel summary
    path = os.path.join(output_dir, f"{prefix}_summary.png")
    plot_publication_summary(sweep_results, analysis, filepath=path)
    figures["summary"] = path

    # Also save as PDF for LaTeX
    path_pdf = os.path.join(output_dir, f"{prefix}_summary.pdf")
    plot_publication_summary(sweep_results, analysis, filepath=path_pdf)
    figures["summary_pdf"] = path_pdf

    return figures
