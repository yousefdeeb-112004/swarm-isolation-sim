"""
Visualization & Analysis — plotting and charting for simulation data.

Provides:
  - plot_fitness_evolution: best/avg fitness + diversity over generations
  - plot_gene_evolution: gene value trends across generations
  - plot_population_dynamics: alive agents, energy, food over steps
  - plot_isolation_comparison: control vs treatment bar chart
  - plot_isolation_timeline: side-by-side population curves
  - plot_genome_heatmap: gene values across agents
  - plot_environment_snapshot: 2D grid view with agents, food, obstacles
  - plot_inner_state_distribution: distribution of inner states across agents
  - generate_report: full multi-panel dashboard PDF/PNG

All functions return a matplotlib Figure and optionally save to disk.
Uses Agg backend for headless environments (WSL, servers).
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np


# Consistent style
COLORS = {
    "control": "#2196F3",       # blue
    "treatment": "#F44336",     # red
    "best_fitness": "#4CAF50",  # green
    "avg_fitness": "#FF9800",   # orange
    "diversity": "#9C27B0",     # purple
    "food": "#8BC34A",          # light green
    "energy": "#FFC107",        # amber
    "alive": "#00BCD4",         # cyan
    "isolation": "#E91E63",     # pink
}

GENE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
]


def _setup_figure(
    figsize: Tuple[float, float] = (10, 6),
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save_fig(fig: plt.Figure, filepath: Optional[str], dpi: int = 150) -> None:
    """Save figure if filepath is provided."""
    if filepath:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")


# ======================================================================
# 1. Fitness Evolution
# ======================================================================

def plot_fitness_evolution(
    generations: List[int],
    best_fitness: List[float],
    avg_fitness: List[float],
    diversity: Optional[List[float]] = None,
    title: str = "Fitness Evolution Across Generations",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Plot best/avg fitness and genome diversity over generations.

    Parameters
    ----------
    generations : list of generation numbers
    best_fitness : best fitness per generation
    avg_fitness : average fitness per generation
    diversity : optional genome diversity per generation
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(generations, best_fitness,
             color=COLORS["best_fitness"], linewidth=2.5,
             marker="o", markersize=4, label="Best Fitness")
    ax1.plot(generations, avg_fitness,
             color=COLORS["avg_fitness"], linewidth=2,
             marker="s", markersize=3, label="Avg Fitness")
    ax1.fill_between(generations, avg_fitness, best_fitness,
                     alpha=0.15, color=COLORS["best_fitness"])

    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Fitness", fontsize=12, color="black")
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)

    if diversity is not None:
        ax2 = ax1.twinx()
        ax2.plot(generations, diversity,
                 color=COLORS["diversity"], linewidth=1.5,
                 linestyle="--", marker="^", markersize=3,
                 label="Genome Diversity", alpha=0.8)
        ax2.set_ylabel("Diversity", fontsize=12, color=COLORS["diversity"])
        ax2.tick_params(axis="y", labelcolor=COLORS["diversity"])
        ax2.spines["top"].set_visible(False)
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="lower right", fontsize=10)
    else:
        ax1.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


# ======================================================================
# 2. Gene Evolution
# ======================================================================

def plot_gene_evolution(
    generations: List[int],
    gene_trends: Dict[str, List[float]],
    title: str = "Gene Value Evolution",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how average gene values change across generations.

    Parameters
    ----------
    generations : list of generation numbers
    gene_trends : dict of {gene_name: [avg_values_per_gen]}
    """
    fig, ax = _setup_figure(figsize=(12, 6), title=title)

    for i, (gene_name, values) in enumerate(gene_trends.items()):
        color = GENE_COLORS[i % len(GENE_COLORS)]
        ax.plot(generations, values,
                color=color, linewidth=2,
                marker="o", markersize=3,
                label=gene_name)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Average Gene Value", fontsize=12)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


# ======================================================================
# 3. Population Dynamics (step-level)
# ======================================================================

def plot_population_dynamics(
    steps: List[int],
    alive: List[int],
    avg_energy: Optional[List[float]] = None,
    food_count: Optional[List[int]] = None,
    title: str = "Population Dynamics",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Plot population size, average energy, and food over simulation steps.
    """
    num_axes = 1 + (avg_energy is not None) + (food_count is not None)
    fig, axes = plt.subplots(num_axes, 1, figsize=(12, 3.5 * num_axes),
                              sharex=True)
    if num_axes == 1:
        axes = [axes]

    idx = 0

    # Population
    axes[idx].fill_between(steps, alive, alpha=0.3, color=COLORS["alive"])
    axes[idx].plot(steps, alive, color=COLORS["alive"], linewidth=1.5)
    axes[idx].set_ylabel("Agents Alive", fontsize=11)
    axes[idx].grid(True, alpha=0.3, linestyle="--")
    axes[idx].spines["top"].set_visible(False)
    axes[idx].spines["right"].set_visible(False)
    idx += 1

    # Energy
    if avg_energy is not None:
        axes[idx].plot(steps, avg_energy,
                       color=COLORS["energy"], linewidth=1.5)
        axes[idx].set_ylabel("Avg Energy", fontsize=11)
        axes[idx].grid(True, alpha=0.3, linestyle="--")
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
        idx += 1

    # Food
    if food_count is not None:
        axes[idx].plot(steps, food_count,
                       color=COLORS["food"], linewidth=1.5)
        axes[idx].set_ylabel("Food Available", fontsize=11)
        axes[idx].grid(True, alpha=0.3, linestyle="--")
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)

    axes[-1].set_xlabel("Step", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


# ======================================================================
# 4. Isolation Comparison (bar chart)
# ======================================================================

def plot_isolation_comparison(
    comparison: Dict[str, Dict[str, Any]],
    title: str = "Isolation Impact: Control vs Treatment",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing control vs treatment across fitness, survival,
    diversity, and food metrics.

    Parameters
    ----------
    comparison : dict with keys 'fitness', 'survival', 'diversity', 'food'
                 each containing 'control_avg', 'treatment_avg'
    """
    metrics = []
    ctrl_vals = []
    treat_vals = []

    for key in ["fitness", "survival", "diversity", "food"]:
        if key in comparison:
            metrics.append(key.capitalize())
            c = comparison[key].get("control_avg", 0)
            t = comparison[key].get("treatment_avg", 0)
            # Normalize food to same scale as others for display
            if key == "food":
                max_val = max(abs(c), abs(t), 1)
                c = c / max_val
                t = t / max_val
                metrics[-1] = "Food (normalized)"
            ctrl_vals.append(c)
            treat_vals.append(t)

    if not metrics:
        fig, ax = _setup_figure(title="No comparison data")
        return fig

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ctrl_vals, width,
                   color=COLORS["control"], alpha=0.85, label="Control")
    bars2 = ax.bar(x + width/2, treat_vals, width,
                   color=COLORS["treatment"], alpha=0.85, label="Treatment")

    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


# ======================================================================
# 5. Isolation Timeline (population curves)
# ======================================================================

def plot_isolation_timeline(
    control_alive: List[int],
    treatment_alive: List[int],
    control_energy: Optional[List[float]] = None,
    treatment_energy: Optional[List[float]] = None,
    title: str = "Isolation Timeline: Population Over Time",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of population survival for control vs treatment.
    """
    has_energy = (control_energy is not None and treatment_energy is not None)
    num_rows = 2 if has_energy else 1

    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 4 * num_rows),
                              sharex=True)
    if num_rows == 1:
        axes = [axes]

    steps = list(range(1, len(control_alive) + 1))

    # Population
    axes[0].plot(steps, control_alive,
                 color=COLORS["control"], linewidth=2, label="Control")
    axes[0].plot(steps[:len(treatment_alive)], treatment_alive,
                 color=COLORS["treatment"], linewidth=2, label="Treatment")
    axes[0].fill_between(steps, control_alive, alpha=0.15, color=COLORS["control"])
    axes[0].fill_between(steps[:len(treatment_alive)], treatment_alive,
                         alpha=0.15, color=COLORS["treatment"])
    axes[0].set_ylabel("Agents Alive", fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Energy
    if has_energy:
        axes[1].plot(steps, control_energy,
                     color=COLORS["control"], linewidth=1.5, label="Control Avg Energy")
        axes[1].plot(steps[:len(treatment_energy)], treatment_energy,
                     color=COLORS["treatment"], linewidth=1.5, label="Treatment Avg Energy")
        axes[1].set_ylabel("Avg Energy", fontsize=11)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

    axes[-1].set_xlabel("Step", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


# ======================================================================
# 6. Genome Heatmap
# ======================================================================

def plot_genome_heatmap(
    agents_data: List[Dict[str, float]],
    gene_names: Optional[List[str]] = None,
    title: str = "Genome Heatmap Across Agents",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of gene values across agents.

    Parameters
    ----------
    agents_data : list of dicts with gene_name: value pairs
    gene_names : optional list of gene names to include
    """
    if not agents_data:
        fig, ax = _setup_figure(title="No agent data")
        return fig

    if gene_names is None:
        gene_names = [k for k in agents_data[0].keys()
                      if k.startswith("gene_")]
        gene_names = [g.replace("gene_", "") for g in gene_names]

    # Build matrix
    matrix = []
    for agent in agents_data:
        row = []
        for gene in gene_names:
            key = f"gene_{gene}" if f"gene_{gene}" in agent else gene
            val = agent.get(key, agent.get(f"gene_{gene}", 0))
            row.append(float(val))
        matrix.append(row)

    matrix = np.array(matrix)

    # Normalize columns to [0, 1] for display
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    matrix_norm = (matrix - col_min) / col_range

    fig, ax = plt.subplots(figsize=(max(8, len(gene_names) * 0.8),
                                     max(4, len(agents_data) * 0.15 + 2)))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")

    ax.set_xticks(range(len(gene_names)))
    ax.set_xticklabels([g[:12] for g in gene_names],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Agent Index", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Gene Value", fontsize=10)

    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


# ======================================================================
# 7. Environment Snapshot
# ======================================================================

def plot_environment_snapshot(
    grid: np.ndarray,
    agents: Optional[List[Dict[str, Any]]] = None,
    title: str = "Environment Snapshot",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    2D visualization of the environment grid.

    Parameters
    ----------
    grid : 2D numpy array (0=empty, 1=food, 2=obstacle, 3=predator)
    agents : optional list of dicts with 'x', 'y', optionally 'alive',
             'is_isolated', 'energy'
    """
    h, w = grid.shape
    fig, ax = plt.subplots(figsize=(min(12, w * 0.15 + 2),
                                     min(12, h * 0.15 + 2)))

    # Color map: empty=white, food=green, obstacle=grey, predator=red
    display = np.ones((h, w, 3))  # white background

    food_mask = (grid == 1)
    obstacle_mask = (grid == 2)
    predator_mask = (grid == 3)

    display[food_mask] = [0.56, 0.93, 0.56]    # light green
    display[obstacle_mask] = [0.5, 0.5, 0.5]    # grey
    display[predator_mask] = [1.0, 0.3, 0.3]    # red

    ax.imshow(display, origin="lower", interpolation="nearest")

    # Draw agents
    if agents:
        for agent_data in agents:
            x = agent_data.get("x", 0)
            y = agent_data.get("y", 0)
            alive = agent_data.get("alive", True)
            isolated = agent_data.get("is_isolated", False)

            if not alive:
                continue

            if isolated:
                color = COLORS["isolation"]
                marker = "x"
                size = 60
            else:
                color = "#2196F3"
                marker = "o"
                size = 40

            ax.scatter(x, y, c=color, marker=marker, s=size,
                      edgecolors="black", linewidths=0.5, zorder=5)

    # Legend
    patches = [
        mpatches.Patch(color=[0.56, 0.93, 0.56], label="Food"),
        mpatches.Patch(color=[0.5, 0.5, 0.5], label="Obstacle"),
        mpatches.Patch(color=[1.0, 0.3, 0.3], label="Predator"),
    ]
    if agents:
        patches.append(mpatches.Patch(color="#2196F3", label="Agent"))
        patches.append(mpatches.Patch(color=COLORS["isolation"], label="Isolated"))
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


# ======================================================================
# 8. Inner State Distribution
# ======================================================================

def plot_inner_state_distribution(
    agents_data: List[Dict[str, float]],
    state_names: Optional[List[str]] = None,
    title: str = "Inner State Distribution",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Box plot of inner state values across all agents.

    Parameters
    ----------
    agents_data : list of dicts with inner_hunger, inner_fear, etc.
    """
    if state_names is None:
        state_names = ["hunger", "fear", "curiosity", "loneliness", "aggression"]

    data = []
    labels = []
    for state in state_names:
        key = f"inner_{state}" if f"inner_{state}" in (agents_data[0] if agents_data else {}) else state
        vals = [float(a.get(key, a.get(f"inner_{state}", 0)))
                for a in agents_data]
        if vals:
            data.append(vals)
            labels.append(state.capitalize())

    if not data:
        fig, ax = _setup_figure(title="No inner state data")
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)

    colors = ["#FF6B6B", "#FFA07A", "#98D8C8", "#87CEEB", "#DDA0DD"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Value [0, 1]", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


# ======================================================================
# 9. Full Dashboard / Report
# ======================================================================

def generate_report(
    evolution_data: Optional[Dict[str, Any]] = None,
    experiment_data: Optional[Dict[str, Any]] = None,
    step_data: Optional[List[Dict[str, Any]]] = None,
    agent_snapshots: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "data/plots",
    prefix: str = "report",
) -> Dict[str, str]:
    """
    Generate a full set of analysis plots.

    Parameters
    ----------
    evolution_data : dict with 'generations', 'best_fitness', 'avg_fitness',
                     'diversity', 'gene_trends'
    experiment_data : dict from IsolationExperiment.get_results()
    step_data : list of per-step metric dicts
    agent_snapshots : list of per-agent snapshot dicts

    Returns dict of {plot_name: filepath}
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    # 1. Fitness Evolution
    if evolution_data and "generations" in evolution_data:
        fp = os.path.join(output_dir, f"{prefix}_fitness_evolution.png")
        plot_fitness_evolution(
            generations=evolution_data["generations"],
            best_fitness=evolution_data["best_fitness"],
            avg_fitness=evolution_data["avg_fitness"],
            diversity=evolution_data.get("diversity"),
            filepath=fp,
        )
        plt.close()
        generated["fitness_evolution"] = fp

    # 2. Gene Evolution
    if evolution_data and "gene_trends" in evolution_data:
        fp = os.path.join(output_dir, f"{prefix}_gene_evolution.png")
        plot_gene_evolution(
            generations=evolution_data["generations"],
            gene_trends=evolution_data["gene_trends"],
            filepath=fp,
        )
        plt.close()
        generated["gene_evolution"] = fp

    # 3. Population Dynamics
    if step_data:
        steps = [d.get("step", i+1) for i, d in enumerate(step_data)]
        alive = [d.get("agents_alive", 0) for d in step_data]
        energy = [d.get("avg_energy", 0) for d in step_data]
        food = [d.get("total_food", d.get("food_count", 0)) for d in step_data]

        fp = os.path.join(output_dir, f"{prefix}_population_dynamics.png")
        plot_population_dynamics(
            steps=steps, alive=alive,
            avg_energy=energy, food_count=food,
            filepath=fp,
        )
        plt.close()
        generated["population_dynamics"] = fp

    # 4. Isolation Comparison
    if experiment_data and "comparison" in experiment_data:
        fp = os.path.join(output_dir, f"{prefix}_isolation_comparison.png")
        plot_isolation_comparison(
            comparison=experiment_data["comparison"],
            filepath=fp,
        )
        plt.close()
        generated["isolation_comparison"] = fp

    # 5. Isolation Timeline
    if experiment_data:
        ctrl_hist = experiment_data.get("control_history", [])
        treat_hist = experiment_data.get("treatment_history", [])
        if ctrl_hist and treat_hist:
            ctrl_alive = [d.get("agents_alive", 0) for d in ctrl_hist]
            treat_alive = [d.get("agents_alive", 0) for d in treat_hist]
            ctrl_energy = [d.get("avg_energy", 0) for d in ctrl_hist]
            treat_energy = [d.get("avg_energy", 0) for d in treat_hist]

            fp = os.path.join(output_dir, f"{prefix}_isolation_timeline.png")
            plot_isolation_timeline(
                control_alive=ctrl_alive,
                treatment_alive=treat_alive,
                control_energy=ctrl_energy,
                treatment_energy=treat_energy,
                filepath=fp,
            )
            plt.close()
            generated["isolation_timeline"] = fp

    # 6. Genome Heatmap
    if agent_snapshots:
        fp = os.path.join(output_dir, f"{prefix}_genome_heatmap.png")
        plot_genome_heatmap(
            agents_data=agent_snapshots[:100],  # cap at 100 for readability
            filepath=fp,
        )
        plt.close()
        generated["genome_heatmap"] = fp

    # 7. Inner State Distribution
    if agent_snapshots:
        fp = os.path.join(output_dir, f"{prefix}_inner_states.png")
        plot_inner_state_distribution(
            agents_data=agent_snapshots,
            filepath=fp,
        )
        plt.close()
        generated["inner_states"] = fp

    return generated


# ======================================================================
# High-level convenience: generate from World / Experiment objects
# ======================================================================

def visualize_evolution(
    world,
    output_dir: str = "data/plots",
    prefix: str = "evolution",
) -> Dict[str, str]:
    """
    Generate evolution plots from a World that has run evolution.
    """
    em = world.evolution_manager
    trend = em.get_fitness_trend()

    # Build gene trends
    gene_names = ["adventurousness", "affiliation_need", "xenophobia",
                  "plasticity", "exploration_rate", "reproduction_threshold"]
    gene_trends = {}
    for gene in gene_names:
        values = em.get_gene_trend(gene)
        if values:
            gene_trends[gene] = values

    evolution_data = {
        "generations": trend["generations"],
        "best_fitness": trend["best_fitness"],
        "avg_fitness": trend["avg_fitness"],
        "diversity": trend["diversity"],
        "gene_trends": gene_trends,
    }

    return generate_report(
        evolution_data=evolution_data,
        step_data=world.step_history,
        output_dir=output_dir,
        prefix=prefix,
    )


def visualize_experiment(
    experiment,
    output_dir: str = "data/plots",
    prefix: str = "experiment",
) -> Dict[str, str]:
    """
    Generate all plots from a completed IsolationExperiment.
    """
    results = experiment.get_results()

    # Build evolution data from control generations
    ctrl_gens = results.get("control_generations", [])
    evolution_data = None
    if ctrl_gens:
        evolution_data = {
            "generations": list(range(len(ctrl_gens))),
            "best_fitness": [g.get("best_fitness", 0) for g in ctrl_gens],
            "avg_fitness": [g.get("avg_fitness", 0) for g in ctrl_gens],
            "diversity": [g.get("genome_diversity", 0) for g in ctrl_gens],
        }

    experiment_data = {
        "comparison": results.get("comparison", {}),
        "control_history": experiment.control_history,
        "treatment_history": experiment.treatment_history,
    }

    return generate_report(
        evolution_data=evolution_data,
        experiment_data=experiment_data,
        output_dir=output_dir,
        prefix=prefix,
    )


# ======================================================================
# SECTION 11 — Research Analysis Visualizations
# ======================================================================

def plot_subjective_experience(
    analysis: dict,
    title: str = "Subjective Inner State Changes During Isolation",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart: pre / during / post isolation inner states.

    Parameters
    ----------
    analysis : dict from ResearchExperiment subjective_experience result
    """
    states = ["hunger", "fear", "curiosity", "loneliness", "aggression"]
    present = [s for s in states if s in analysis and isinstance(analysis[s], dict)]
    if not present:
        fig, ax = _setup_figure(title="No subjective data")
        return fig

    pre = [analysis[s].get("pre_isolation_mean", 0) for s in present]
    dur = [analysis[s].get("during_isolation_mean", 0) for s in present]
    post = [analysis[s].get("post_return_mean", 0) for s in present]

    x = np.arange(len(present))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, pre, w, color="#4CAF50", alpha=0.85, label="Pre-Isolation")
    ax.bar(x,     dur, w, color="#FF9800", alpha=0.85, label="During Isolation")
    ax.bar(x + w, post, w, color="#F44336", alpha=0.85, label="Post-Return")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in present], fontsize=11)
    ax.set_ylabel("Mean Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


def plot_loneliness_trajectory(
    trajectory: List[float],
    confidence_trajectory: Optional[List[float]] = None,
    title: str = "Loneliness & Confidence During Isolation",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of loneliness (and optionally confidence) during isolation.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    steps = list(range(1, len(trajectory) + 1))

    ax1.plot(steps, trajectory, color="#E91E63", linewidth=2.5,
             marker="o", markersize=4, label="Loneliness")
    ax1.fill_between(steps, trajectory, alpha=0.15, color="#E91E63")
    ax1.set_xlabel("Sample Point During Isolation", fontsize=12)
    ax1.set_ylabel("Loneliness", fontsize=12, color="#E91E63")
    ax1.tick_params(axis="y", labelcolor="#E91E63")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)

    if confidence_trajectory and len(confidence_trajectory) == len(trajectory):
        ax2 = ax1.twinx()
        ax2.plot(steps, confidence_trajectory, color="#9C27B0", linewidth=2,
                 linestyle="--", marker="^", markersize=3,
                 label="Belief Confidence", alpha=0.8)
        ax2.set_ylabel("Avg Confidence", fontsize=12, color="#9C27B0")
        ax2.tick_params(axis="y", labelcolor="#9C27B0")
        ax2.spines["top"].set_visible(False)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)
    else:
        ax1.legend(loc="upper left", fontsize=10)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


def plot_belief_divergence(
    analysis: dict,
    title: str = "Belief Divergence: Isolated Agents vs Group",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of belief divergence metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = {
        "Confidence\nat Isolation": analysis.get("confidence_at_isolation", 0),
        "Confidence\nat Return": analysis.get("confidence_at_return", 0),
        "Food Belief\nGap (Mean)": analysis.get("food_belief_gap_mean", 0),
        "Belief\nDivergence": analysis.get("mean_belief_divergence", 0),
    }

    bars = ax.bar(
        list(metrics.keys()), list(metrics.values()),
        color=["#4CAF50", "#F44336", "#FF9800", "#9C27B0"],
        alpha=0.85, width=0.5,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f"{h:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


def plot_group_inner_comparison(
    inner_state_comparison: dict,
    title: str = "Group Inner States: Control vs Treatment",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Paired bar chart comparing group-level inner states.
    """
    states = [s for s in ["hunger", "fear", "curiosity", "loneliness", "aggression"]
              if s in inner_state_comparison]
    if not states:
        fig, ax = _setup_figure(title="No data")
        return fig

    ctrl = [inner_state_comparison[s]["control_mean"] for s in states]
    treat = [inner_state_comparison[s]["treatment_mean"] for s in states]

    x = np.arange(len(states))
    w = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, ctrl, w, color=COLORS["control"], alpha=0.85, label="Control")
    ax.bar(x + w/2, treat, w, color=COLORS["treatment"], alpha=0.85, label="Treatment")

    # Add difference annotations
    for i, s in enumerate(states):
        diff = inner_state_comparison[s]["difference"]
        arrow = "↑" if diff > 0 else "↓"
        ax.text(i, max(ctrl[i], treat[i]) + 0.02,
                f"{arrow}{abs(diff):.3f}", ha="center", fontsize=9,
                color="#F44336" if diff > 0 else "#4CAF50")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in states], fontsize=11)
    ax.set_ylabel("Mean Inner State Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


def plot_behavioral_change(
    analysis: dict,
    title: str = "Behavioral Changes After Isolation",
    filepath: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of behavioral metric changes.
    """
    metrics = {}
    label_map = {
        "mean_energy_change": "Energy Change",
        "mean_food_rate_change": "Food Rate Δ",
        "mean_reward_change": "RL Reward Δ",
        "post_return_survival_rate": "Survival Rate (30 steps)",
    }
    for key, label in label_map.items():
        if key in analysis:
            metrics[label] = analysis[key]

    if not metrics:
        fig, ax = _setup_figure(title="No behavioral data")
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors_list = ["#F44336" if v < 0 else "#4CAF50" for v in values]

    bars = ax.barh(labels, values, color=colors_list, alpha=0.85, height=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(v + (0.005 if v >= 0 else -0.005), bar.get_y() + bar.get_height()/2.,
                f"{v:+.4f}", ha="left" if v >= 0 else "right",
                va="center", fontsize=10)

    ax.set_xlabel("Change", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return fig


def generate_research_plots(
    results: dict,
    output_dir: str = "data/plots",
    prefix: str = "research",
) -> Dict[str, str]:
    """
    Generate all research analysis plots from experiment results.

    Parameters
    ----------
    results : dict from ResearchExperiment.analyze()
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    se = results.get("subjective_experience", {})
    bd = results.get("belief_divergence", {})
    bc = results.get("behavioral_change", {})
    gd = results.get("group_dynamics", {})

    # 1. Subjective experience bars
    if se and "hunger" in se:
        fp = os.path.join(output_dir, f"{prefix}_subjective_experience.png")
        plot_subjective_experience(se, filepath=fp)
        plt.close()
        generated["subjective_experience"] = fp

    # 2. Loneliness trajectory
    lone_traj = se.get("loneliness_trajectory_avg", [])
    conf_traj = se.get("confidence_trajectory_avg", [])
    if lone_traj and len(lone_traj) >= 2:
        fp = os.path.join(output_dir, f"{prefix}_loneliness_trajectory.png")
        plot_loneliness_trajectory(lone_traj, conf_traj or None, filepath=fp)
        plt.close()
        generated["loneliness_trajectory"] = fp

    # 3. Belief divergence
    if bd and "mean_belief_divergence" in bd:
        fp = os.path.join(output_dir, f"{prefix}_belief_divergence.png")
        plot_belief_divergence(bd, filepath=fp)
        plt.close()
        generated["belief_divergence"] = fp

    # 4. Group inner state comparison
    isc = gd.get("inner_state_comparison", {})
    if isc:
        fp = os.path.join(output_dir, f"{prefix}_group_inner_states.png")
        plot_group_inner_comparison(isc, filepath=fp)
        plt.close()
        generated["group_inner_states"] = fp

    # 5. Behavioral change
    if bc and "mean_energy_change" in bc:
        fp = os.path.join(output_dir, f"{prefix}_behavioral_change.png")
        plot_behavioral_change(bc, filepath=fp)
        plt.close()
        generated["behavioral_change"] = fp

    return generated
