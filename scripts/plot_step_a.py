#!/usr/bin/env python3
"""
Step A figures: Kaplan-Meier survival curve and population trajectory,
comparing the DEFAULT (pre-tuning) and CALIBRATED control baselines.

Reads the JSON + traj.npz files produced by run_control_baseline.py and
writes 300-DPI PNG + PDF figures to <out>/figures/.

Usage:
  python scripts/plot_step_a.py \
      --default  data/results/phase1/control_baseline_default_baseline \
      --calibrated data/results/phase1/control_baseline_calibrated \
      --out data/results/phase1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swarm_sim.utils.pub_visualization import PUB_STYLE, COLORS


def _load(stem: str):
    with open(stem + ".json") as f:
        summ = json.load(f)
    traj = np.load(stem + "_traj.npz")
    return summ, traj


def _save(fig, path_stem: str):
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def plot_km(default, calibrated, out_stem: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for summ, traj, color, label in [
        (default[0], default[1], COLORS["treatment"], "Default (pre-tuning)"),
        (calibrated[0], calibrated[1], COLORS["control"], "Calibrated"),
    ]:
        if summ is None:
            continue
        t = traj["km_times"]
        s = traj["km_survival"]
        lo = traj["km_ci_lower"]
        hi = traj["km_ci_upper"]
        # step function
        ax.step(t, s, where="post", color=color, linewidth=2,
                label=f"{label} (ext={summ['extinction_rate']:.0%}, "
                      f"median={summ['km_median_survival']})")
        ax.fill_between(t, lo, hi, step="post", color=color, alpha=0.15)

    max_step = calibrated[0]["num_generations"] * calibrated[0]["steps_per_generation"]
    ax.axhline(0.5, color=COLORS["neutral"], linestyle=":", linewidth=1,
               label="50% survival")
    ax.set_xlabel("Simulation step (5 generations concatenated)")
    ax.set_ylabel("Population survival probability")
    ax.set_title("Kaplan–Meier: control-population survival\n"
                 "default vs. calibrated environment")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, max_step)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, out_stem)
    print(f"[*] wrote {out_stem}.png / .pdf")


def plot_trajectory(default, calibrated, out_stem: str):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for summ, traj, color, label in [
        (default[0], default[1], COLORS["treatment"], "Default (pre-tuning)"),
        (calibrated[0], calibrated[1], COLORS["control"], "Calibrated"),
    ]:
        if summ is None:
            continue
        y = traj["mean_traj"]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color=color, linewidth=1.3, label=label)

    # Generation boundaries
    spg = int(calibrated[1]["steps_per_generation"])
    ngen = int(calibrated[1]["num_generations"])
    for g in range(1, ngen):
        ax.axvline(g * spg, color=COLORS["neutral"], linestyle="--",
                   linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Simulation step (generations separated by dashed lines)")
    ax.set_ylabel("Mean living agents (avg. over seeds)")
    ax.set_title("Control population trajectory: default vs. calibrated")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _save(fig, out_stem)
    print(f"[*] wrote {out_stem}.png / .pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--default", required=True,
                   help="stem of default baseline (no extension)")
    p.add_argument("--calibrated", required=True,
                   help="stem of calibrated baseline (no extension)")
    p.add_argument("--out", default="data/results/phase1")
    args = p.parse_args()

    plt.rcParams.update(PUB_STYLE)

    default = _load(args.default)
    calibrated = _load(args.calibrated)

    fig_dir = os.path.join(args.out, "figures")
    plot_km(default, calibrated, os.path.join(fig_dir, "step_a_kaplan_meier"))
    plot_trajectory(default, calibrated,
                    os.path.join(fig_dir, "step_a_population_trajectory"))


if __name__ == "__main__":
    main()
