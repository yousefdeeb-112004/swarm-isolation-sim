#!/usr/bin/env python3
"""
Phase 1, Step C — analyse the density-relief harshest-cell run.

Loads the checkpointed harshest-cell run (md 0.0 / predator_protection OFF /
30% isolation, 30 seeds, 5 x 1000), aligns the per-step trajectories persisted by
ExtendedExperiment (agents_alive, standing total_food, mean avg_energy, and the
treatment isolation count) across seeds on a common global-step grid, and tests
the density-relief hypothesis directly:

  Q1 (food)   Do available (standing) food levels rise DURING isolation windows in
              treatment relative to control, at matched timepoints where both
              populations are still alive?
  Q2 (energy) Does mean agent energy improve in treatment during those windows?
              (Caveat: avg_energy pools isolated + non-isolated agents.)
  Q3 (pop)    Does the population trajectory show recovery timed to the isolation
              windows (treatment stabilising while control collapses)?

Each treatment run is paired to its own no-isolation control on the same seed, so
the unit of analysis is the seed: per seed we average the treatment-minus-control
difference across the in-window, both-alive timepoints, then test those 30
per-seed means against zero (one-sample t + sign count). This avoids
pseudoreplication across the correlated within-run timepoints.

Outputs:
  <out>/density_relief_analysis.json   all computed numbers
  <out>/figures/density_relief_trajectories.{png,pdf}   3-panel overlay

Usage:
  python scripts/analyze_density_relief.py --out data/results/phase1/density_relief
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/results/phase1/density_relief")
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


def _align(recs, world, metric, grid_index):
    """Return an [n_seeds x n_grid] matrix (NaN where a run has no sample)."""
    n, G = len(recs), len(grid_index)
    M = np.full((n, G), np.nan)
    for i, r in enumerate(recs):
        st = r["step_trajectories"]
        steps = st["steps"]
        vals = st[world][metric]
        for s, v in zip(steps, vals):
            j = grid_index.get(s)
            if j is not None and v is not None:
                M[i, j] = float(v)
    return M


def _mean_ci(M):
    """Column-wise nan-mean and 95% CI half-width; also n valid per column."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0, ddof=1)
    nvalid = np.sum(~np.isnan(M), axis=0)
    sem = np.where(nvalid > 1, std / np.sqrt(np.maximum(nvalid, 1)), 0.0)
    ci = 1.96 * sem
    return mean, ci, nvalid


def _contiguous_true(mask, grid):
    """Yield (start_step, end_step) spans where mask is True."""
    spans = []
    start = None
    for k, on in enumerate(mask):
        if on and start is None:
            start = grid[k]
        elif not on and start is not None:
            spans.append((start, grid[k]))
            start = None
    if start is not None:
        spans.append((start, grid[-1]))
    return spans


def main():
    args = parse_args()
    out = args.out
    ckpt = os.path.join(out, "checkpoint.jsonl")
    recs = _load(ckpt)
    n_seeds = len(recs)
    print(f"[*] Loaded {n_seeds} runs from {ckpt}")

    total_steps = args.generations * args.steps_per_gen
    grid = np.arange(5, total_steps + 1, 5)          # 5,10,...,5000
    grid_index = {int(s): j for j, s in enumerate(grid)}

    # Aligned matrices.
    ctrl_food = _align(recs, "ctrl", "total_food", grid_index)
    treat_food = _align(recs, "treat", "total_food", grid_index)
    ctrl_pop = _align(recs, "ctrl", "agents_alive", grid_index)
    treat_pop = _align(recs, "treat", "agents_alive", grid_index)
    ctrl_en = _align(recs, "ctrl", "avg_energy", grid_index)
    treat_en = _align(recs, "treat", "avg_energy", grid_index)
    treat_iso = _align(recs, "treat", "currently_isolated", grid_index)

    # Isolation-on mask (any seed isolating at that step → treat mean iso > 0).
    with np.errstate(invalid="ignore"):
        iso_mean = np.nanmean(treat_iso, axis=0)
    iso_on = np.nan_to_num(iso_mean) > 0.0
    iso_spans = _contiguous_true(iso_on, grid)

    # Extinction rates (from run-level fields).
    treat_ext = sum(1 for r in recs if r["treat_extinct"])
    ctrl_ext = sum(1 for r in recs
                   if r.get("ctrl_extinction_step") is not None)

    # -------- Paired per-seed in-window (both-alive) contrasts --------
    both_alive = (ctrl_pop > 0) & (treat_pop > 0)
    in_window = both_alive & iso_on[None, :]
    off_window = both_alive & (~iso_on[None, :])

    def _per_seed_mean(diff, mask):
        """Mean of diff over masked columns, per seed; NaN if no valid cols."""
        out_vals = np.full(diff.shape[0], np.nan)
        for i in range(diff.shape[0]):
            sel = mask[i] & ~np.isnan(diff[i])
            if sel.any():
                out_vals[i] = np.nanmean(diff[i][sel])
        return out_vals

    d_food = treat_food - ctrl_food
    d_en = treat_en - ctrl_en
    d_pop = treat_pop - ctrl_pop

    def _summarise(diff, mask, label):
        per_seed = _per_seed_mean(diff, mask)
        valid = per_seed[~np.isnan(per_seed)]
        if len(valid) >= 2:
            t, p = sp_stats.ttest_1samp(valid, 0.0)
        else:
            t, p = float("nan"), float("nan")
        n_pos = int(np.sum(valid > 0))
        return {
            "label": label,
            "n_seeds_with_data": int(len(valid)),
            "mean_per_seed_diff": round(float(np.mean(valid)), 4) if len(valid) else None,
            "sd_per_seed_diff": round(float(np.std(valid, ddof=1)), 4) if len(valid) > 1 else None,
            "t": round(float(t), 4) if not np.isnan(t) else None,
            "p_value": round(float(p), 6) if not np.isnan(p) else None,
            "n_seeds_treat_gt_ctrl": n_pos,
            "n_seeds_total": int(len(valid)),
        }

    q1_food_inwin = _summarise(d_food, in_window, "food: treat-ctrl, in-window, both alive")
    q1_food_offwin = _summarise(d_food, off_window, "food: treat-ctrl, off-window, both alive")
    q2_en_inwin = _summarise(d_en, in_window, "energy: treat-ctrl, in-window, both alive")
    q3_pop_inwin = _summarise(d_pop, in_window, "population: treat-ctrl, in-window, both alive")

    # Per-generation end-of-gen population means (from gen records).
    def _gen_pop(field):
        arr = np.array([[g["alive_at_end"] for g in r[field]] for r in recs],
                       dtype=float)
        return arr.mean(axis=0).round(2).tolist()
    gen_pop_ctrl = _gen_pop("ctrl_gen_records")
    gen_pop_treat = _gen_pop("treat_gen_records")

    # Timepoint count where control is already dead but treatment alive
    # (illustrates why late timepoints must be excluded from the food contrast).
    ctrl_dead_treat_alive = int(np.sum((ctrl_pop == 0) & (treat_pop > 0)))

    analysis = {
        "n_seeds": n_seeds,
        "condition": recs[0]["condition"]["name"],
        "total_steps": int(total_steps),
        "downsample_every": recs[0]["step_trajectories"]["downsample_every"],
        "isolation_spans_global_steps": [[int(a), int(b)] for a, b in iso_spans],
        "extinction": {
            "control": {"n": ctrl_ext, "rate": round(ctrl_ext / n_seeds, 4)},
            "treatment": {"n": treat_ext, "rate": round(treat_ext / n_seeds, 4)},
        },
        "per_generation_mean_pop": {
            "control": gen_pop_ctrl, "treatment": gen_pop_treat,
        },
        "n_gridcells_ctrl_dead_treat_alive": ctrl_dead_treat_alive,
        "Q1_food_in_window": q1_food_inwin,
        "Q1_food_off_window": q1_food_offwin,
        "Q2_energy_in_window": q2_en_inwin,
        "Q3_population_in_window": q3_pop_inwin,
    }

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "density_relief_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # ---------------- Figure: 3-panel overlay ----------------
    _figure(grid, iso_spans, args,
            (_mean_ci(ctrl_food), _mean_ci(treat_food)),
            (_mean_ci(ctrl_pop), _mean_ci(treat_pop)),
            (_mean_ci(ctrl_en), _mean_ci(treat_en)),
            out)

    # ---------------- Console summary ----------------
    print(f"[*] Isolation-on spans (global steps): {analysis['isolation_spans_global_steps'][:3]} ...")
    print(f"[*] Extinction — control {ctrl_ext}/{n_seeds} "
          f"({ctrl_ext/n_seeds:.1%}), treatment {treat_ext}/{n_seeds} "
          f"({treat_ext/n_seeds:.1%})")
    print(f"[*] Per-gen mean pop  control:   {gen_pop_ctrl}")
    print(f"[*] Per-gen mean pop  treatment: {gen_pop_treat}")
    for tag, q in [("Q1 food in-window", q1_food_inwin),
                   ("Q1 food off-window", q1_food_offwin),
                   ("Q2 energy in-window", q2_en_inwin),
                   ("Q3 pop in-window", q3_pop_inwin)]:
        print(f"[*] {tag}: mean(treat-ctrl)={q['mean_per_seed_diff']} "
              f"p={q['p_value']} treat>ctrl in {q['n_seeds_treat_gt_ctrl']}/{q['n_seeds_total']} seeds")
    print(f"[*] Wrote {out}/density_relief_analysis.json and figure.")


def _figure(grid, iso_spans, args, food, pop, energy, out):
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.4), sharex=True)

    panels = [
        (axes[0], food, "Standing food (units on grid)", "Q1: available food"),
        (axes[1], pop, "Agents alive", "Q3: population"),
        (axes[2], energy, "Mean agent energy", "Q2: energy (pools isolated)"),
    ]
    for ax, ((cm, cci, _), (tm, tci, _)), ylab, title in panels:
        # Isolation windows (shade first, behind lines).
        for a, b in iso_spans:
            ax.axvspan(a, b, color=COLORS["ci_band"], alpha=0.25, lw=0)
        # Generation boundaries.
        for g in range(1, args.generations):
            ax.axvline(g * args.steps_per_gen, color=COLORS["neutral"],
                       lw=0.5, ls=":", alpha=0.6)
        ax.plot(grid, cm, color=COLORS["control"], lw=1.3, label="Control")
        ax.fill_between(grid, cm - cci, cm + cci, color=COLORS["control"], alpha=0.18)
        ax.plot(grid, tm, color=COLORS["treatment"], lw=1.3, label="Treatment (30% isolation)")
        ax.fill_between(grid, tm - tci, tm + tci, color=COLORS["treatment"], alpha=0.18)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, args.generations * args.steps_per_gen)

    axes[-1].set_xlabel("Global step (5 generations × 1000 steps)")
    handles = [
        plt.Line2D([0], [0], color=COLORS["control"], lw=1.3, label="Control"),
        plt.Line2D([0], [0], color=COLORS["treatment"], lw=1.3,
                   label="Treatment (30% isolation)"),
        Patch(facecolor=COLORS["ci_band"], alpha=0.25, label="Isolation window active"),
    ]
    axes[0].legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle("Density-relief test — harshest cell (md 0.0, predator protection OFF, 30% isolation)\n"
                 "control vs treatment, mean ± 95% CI across 30 seeds",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    stem = os.path.join(fig_dir, "density_relief_trajectories")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[*] Figure: {stem}.png / .pdf")


if __name__ == "__main__":
    main()
