#!/usr/bin/env python3
"""
Phase 1, Step D — analyse the three-arm spatial-redistribution test.

Loads the checkpointed control / isolation / sham run and answers:

  - Effect sizes (fitness impact, extinction rate, energy) for isolation and
    sham vs control. Does SHAM reproduce the paradox?
  - Is control dispersion LOWER (more clumped) than the treatment arms?
  - Is LOCAL food availability HIGHER in the treatment arms than control,
    despite identical GLOBAL standing food?
  - Per-agent successful-eat rate per arm.

All arms are paired on the same seed, so the unit of analysis is the seed:
per-seed treatment-minus-control differences are tested against zero (paired t;
plus Cohen's d treat-vs-control on the run scalars, matching the factorial's
convention). Trajectory contrasts are restricted to timepoints where both the
arm and control are still alive.

Outputs:
  <out>/spatial_analysis.json
  <out>/figures/spatial_trajectories.{png,pdf}   dispersion/local-food/pop/energy
  <out>/figures/spatial_effect_sizes.{png,pdf}    fitness d, energy d, extinction
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from swarm_sim.utils.pub_visualization import COLORS, PUB_STYLE
from swarm_sim.analysis.stats_analysis import cohens_d

ARMS = ("control", "isolation", "sham")
ARM_COLOR = {
    "control": COLORS["control"],       # blue
    "isolation": COLORS["treatment"],   # red
    "sham": COLORS["highlight"],        # green
}
ARM_LABEL = {
    "control": "Control",
    "isolation": "Isolation 30%",
    "sham": "Sham relocation 30%",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/results/phase1/spatial_test")
    p.add_argument("--steps-per-gen", type=int, default=1000)
    p.add_argument("--generations", type=int, default=5)
    return p.parse_args()


def _load(ckpt):
    recs = []
    with open(ckpt) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["seed"])
    return recs


def _align(recs, arm, metric, grid_index):
    n, G = len(recs), len(grid_index)
    M = np.full((n, G), np.nan)
    for i, r in enumerate(recs):
        st = r["step_trajectories"]
        for s, v in zip(st["steps"], st[arm][metric]):
            j = grid_index.get(s)
            if j is not None and v is not None:
                M[i, j] = float(v)
    return M


def _mean_ci(M):
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0, ddof=1)
    nvalid = np.sum(~np.isnan(M), axis=0)
    sem = np.where(nvalid > 1, std / np.sqrt(np.maximum(nvalid, 1)), 0.0)
    return mean, 1.96 * sem


def _contiguous_true(mask, grid):
    spans, start = [], None
    for k, on in enumerate(mask):
        if on and start is None:
            start = grid[k]
        elif not on and start is not None:
            spans.append((start, grid[k]))
            start = None
    if start is not None:
        spans.append((start, grid[-1]))
    return spans


def _paired(per_seed_diffs):
    """Summarise per-seed paired differences (treat - control)."""
    v = per_seed_diffs[~np.isnan(per_seed_diffs)]
    if len(v) >= 2:
        t, p = sp_stats.ttest_1samp(v, 0.0)
    else:
        t, p = float("nan"), float("nan")
    return {
        "n": int(len(v)),
        "mean_diff": round(float(np.mean(v)), 4) if len(v) else None,
        "paired_t": round(float(t), 4) if not np.isnan(t) else None,
        "paired_p": round(float(p), 6) if not np.isnan(p) else None,
        "n_treat_gt_ctrl": int(np.sum(v > 0)),
        "n_total": int(len(v)),
    }


def _per_seed_metric_mean(recs, arm, metric, grid_index, alive_mask_arm,
                          alive_mask_ctrl):
    """Per-seed mean of a trajectory metric over both-alive timepoints."""
    M = _align(recs, arm, metric, grid_index)
    out = np.full(len(recs), np.nan)
    for i in range(len(recs)):
        sel = alive_mask_arm[i] & alive_mask_ctrl[i] & ~np.isnan(M[i])
        if sel.any():
            out[i] = np.nanmean(M[i][sel])
    return out


def main():
    args = parse_args()
    out = args.out
    recs = _load(os.path.join(out, "checkpoint.jsonl"))
    n_seeds = len(recs)
    print(f"[*] Loaded {n_seeds} seeds")

    total_steps = args.generations * args.steps_per_gen
    grid = np.arange(5, total_steps + 1, 5)
    grid_index = {int(s): j for j, s in enumerate(grid)}

    # Alive masks per arm.
    alive = {a: _align(recs, a, "agents_alive", grid_index) > 0 for a in ARMS}

    # Isolation-on mask (from isolation arm's relocated count).
    reloc = _align(recs, "isolation", "currently_relocated", grid_index)
    with np.errstate(invalid="ignore"):
        iso_on = np.nan_to_num(np.nanmean(reloc, axis=0)) > 0.0
    iso_spans = _contiguous_true(iso_on, grid)

    # ---- Arm-level scalars (fitness / extinction / energy) ----
    fit = {a: np.array([r["arms"][a]["avg_fitness"] for r in recs]) for a in ARMS}
    ext = {a: sum(1 for r in recs if r["arms"][a]["extinct"]) for a in ARMS}

    effect = {}
    for a in ("isolation", "sham"):
        d_fit = cohens_d(fit[a].tolist(), fit["control"].tolist())  # treat vs ctrl
        paired_fit = _paired(fit[a] - fit["control"])
        energy_seed = _per_seed_metric_mean(recs, a, "avg_energy", grid_index,
                                            alive[a], alive["control"])
        energy_ctrl = _per_seed_metric_mean(recs, "control", "avg_energy",
                                            grid_index, alive[a], alive["control"])
        d_energy = cohens_d(energy_seed[~np.isnan(energy_seed)].tolist(),
                            energy_ctrl[~np.isnan(energy_ctrl)].tolist())
        effect[a] = {
            "fitness_impact_mean": round(float(np.mean(fit[a] - fit["control"])), 4),
            "cohens_d_fitness": round(float(d_fit), 4),
            "paired_fitness": paired_fit,
            "extinction_rate": round(ext[a] / n_seeds, 4),
            "energy_mean_diff": _paired(energy_seed - energy_ctrl)["mean_diff"],
            "cohens_d_energy": round(float(d_energy), 4),
            "paired_energy": _paired(energy_seed - energy_ctrl),
        }
    control_summary = {
        "mean_fitness": round(float(np.mean(fit["control"])), 4),
        "extinction_rate": round(ext["control"] / n_seeds, 4),
    }

    # ---- Decisive questions: dispersion + local food (in-window, both alive) ----
    def contrast(metric):
        res = {}
        for a in ("isolation", "sham"):
            arm_seed = _per_seed_metric_mean(recs, a, metric, grid_index,
                                             alive[a] & iso_on[None, :], alive["control"])
            ctrl_seed = _per_seed_metric_mean(recs, "control", metric, grid_index,
                                              alive[a] & iso_on[None, :], alive["control"])
            res[a] = {
                "treat_mean": round(float(np.nanmean(arm_seed)), 4),
                "control_mean": round(float(np.nanmean(ctrl_seed)), 4),
                "paired": _paired(arm_seed - ctrl_seed),
            }
        return res

    dispersion = contrast("avg_distance_between_agents")
    local_food = contrast("local_food_mean")

    # ---- Per-agent eat rate per arm (sum eaten / sum agent-steps, in-window) ----
    eat_rate = {}
    for a in ARMS:
        eaten = _align(recs, a, "food_eaten_this_step", grid_index)
        pop = _align(recs, a, "agents_alive", grid_index)
        mask = (pop > 0) & iso_on[None, :]
        num = np.nansum(np.where(mask, eaten, np.nan))
        den = np.nansum(np.where(mask, pop, np.nan))
        eat_rate[a] = round(float(num / den), 5) if den > 0 else None

    analysis = {
        "n_seeds": n_seeds,
        "total_steps": int(total_steps),
        "isolation_spans_global_steps": [[int(a), int(b)] for a, b in iso_spans],
        "control": control_summary,
        "effect_sizes": effect,
        "dispersion_in_window": dispersion,
        "local_food_in_window": local_food,
        "global_food_mean": {a: round(float(np.nanmean(_align(recs, a, "total_food", grid_index))), 2)
                             for a in ARMS},
        "eat_rate_per_agent_in_window": eat_rate,
    }

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "spatial_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    _fig_trajectories(grid, iso_spans, args, recs, grid_index, out)
    _fig_effects(effect, control_summary, out)

    # ---- Console summary ----
    print(f"[*] Extinction — " + ", ".join(
        f"{a} {ext[a]}/{n_seeds} ({ext[a]/n_seeds:.1%})" for a in ARMS))
    for a in ("isolation", "sham"):
        e = effect[a]
        print(f"[*] {a:9s}: fit_impact={e['fitness_impact_mean']:+.4f} "
              f"d_fit={e['cohens_d_fitness']:+.3f} paired_p={e['paired_fitness']['paired_p']} "
              f"| d_energy={e['cohens_d_energy']:+.3f}")
    print(f"[*] Global food mean/arm: {analysis['global_food_mean']}")
    for label, res in [("dispersion", dispersion), ("local_food", local_food)]:
        for a in ("isolation", "sham"):
            r = res[a]
            print(f"[*] {label} {a}: treat={r['treat_mean']} ctrl={r['control_mean']} "
                  f"diff={r['paired']['mean_diff']} p={r['paired']['paired_p']} "
                  f"({r['paired']['n_treat_gt_ctrl']}/{r['paired']['n_total']})")
    print(f"[*] Eat rate/agent in-window: {eat_rate}")
    print(f"[*] Wrote {out}/spatial_analysis.json + figures.")


def _fig_trajectories(grid, iso_spans, args, recs, grid_index, out):
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 10.5), sharex=True)
    panels = [
        ("avg_distance_between_agents", "Mean pairwise distance", "Spatial dispersion (higher = less clumped)"),
        ("local_food_mean", "Food in sensor range", "Local food availability"),
        ("agents_alive", "Agents alive", "Population"),
        ("avg_energy", "Mean agent energy", "Energy (pools relocated agents)"),
    ]
    for ax, (metric, ylab, title) in zip(axes, panels):
        for a, b in iso_spans:
            ax.axvspan(a, b, color=COLORS["ci_band"], alpha=0.20, lw=0)
        for g in range(1, args.generations):
            ax.axvline(g * args.steps_per_gen, color=COLORS["neutral"],
                       lw=0.5, ls=":", alpha=0.6)
        for arm in ARMS:
            m, ci = _mean_ci(_align(recs, arm, metric, grid_index))
            ax.plot(grid, m, color=ARM_COLOR[arm], lw=1.2, label=ARM_LABEL[arm])
            ax.fill_between(grid, m - ci, m + ci, color=ARM_COLOR[arm], alpha=0.15)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, args.generations * args.steps_per_gen)

    axes[-1].set_xlabel("Global step (5 generations × 1000 steps)")
    handles = [plt.Line2D([0], [0], color=ARM_COLOR[a], lw=1.2, label=ARM_LABEL[a])
               for a in ARMS]
    handles.append(Patch(facecolor=COLORS["ci_band"], alpha=0.20,
                         label="Isolation/relocation window"))
    axes[0].legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Spatial-redistribution test — control vs isolation vs sham relocation\n"
                 "harshest cell (md 0.0, predator protection OFF, 30%), mean ± 95% CI, 30 seeds",
                 fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out, "spatial_trajectories")


def _fig_effects(effect, control_summary, out):
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    arms = ["isolation", "sham"]
    colors = [ARM_COLOR[a] for a in arms]

    # Cohen's d fitness
    axes[0].bar(arms, [effect[a]["cohens_d_fitness"] for a in arms], color=colors)
    axes[0].axhline(0, color=COLORS["neutral"], lw=0.6)
    axes[0].set_title("Cohen's d — fitness\n(treat vs control)", fontsize=9)
    axes[0].set_ylabel("d")

    # Cohen's d energy
    axes[1].bar(arms, [effect[a]["cohens_d_energy"] for a in arms], color=colors)
    axes[1].axhline(0, color=COLORS["neutral"], lw=0.6)
    axes[1].set_title("Cohen's d — energy", fontsize=9)

    # Extinction rate
    rates = [control_summary["extinction_rate"]] + [effect[a]["extinction_rate"] for a in arms]
    axes[2].bar(["control"] + arms, rates,
                color=[ARM_COLOR["control"]] + colors)
    axes[2].set_title("Extinction rate", fontsize=9)
    axes[2].set_ylabel("fraction")

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Effect sizes by arm", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "spatial_effect_sizes")


def _save(fig, out, stem):
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, stem)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[*] Figure: {path}.png / .pdf")


if __name__ == "__main__":
    main()
