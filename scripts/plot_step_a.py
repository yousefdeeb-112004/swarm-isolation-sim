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


def plot_energy_response(points, chosen_ev, out_stem: str):
    """Extinction rate vs energy_value (four 30-seed points), band shaded."""
    evs = [p[0] for p in points]
    rates = [p[1] * 100 for p in points]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axhspan(20, 50, color=COLORS["highlight"], alpha=0.12,
               label="target band (20–50%)")
    ax.plot(evs, rates, "-o", color=COLORS["neutral"], linewidth=1.6,
            markersize=6, zorder=3)
    for ev, rate in zip(evs, rates):
        is_chosen = (ev == chosen_ev)
        ax.plot([ev], [rate], "o", markersize=11 if is_chosen else 0,
                markerfacecolor="none",
                markeredgecolor=COLORS["treatment"], markeredgewidth=2,
                zorder=4)
        ax.annotate(f"{rate:.1f}%", (ev, rate),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=9,
                    fontweight="bold" if is_chosen else "normal")
    ax.set_xlabel("Food energy_value")
    ax.set_ylabel("Control extinction rate (%, 30 seeds)")
    ax.set_title("Energy-response calibration curve\n"
                 f"(chosen: energy_value={chosen_ev}, red ring)")
    ax.set_ylim(0, 100)
    ax.set_xticks(evs)
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
    p.add_argument("--chosen-ev", type=int, default=50,
                   help="chosen energy_value (highlighted on the curve)")
    args = p.parse_args()

    plt.rcParams.update(PUB_STYLE)

    default = _load(args.default)
    calibrated = _load(args.calibrated)

    fig_dir = os.path.join(args.out, "figures")
    plot_km(default, calibrated, os.path.join(fig_dir, "step_a_kaplan_meier"))
    plot_trajectory(default, calibrated,
                    os.path.join(fig_dir, "step_a_population_trajectory"))

    # Energy-response curve from the four 30-seed points (skip any missing).
    ev_stems = {
        20: "control_baseline_default_baseline",
        40: "control_baseline_calibrated_ev40",
        50: "control_baseline_calibrated_ev50",
        55: "control_baseline_calibrated_ev55",
    }
    points = []
    for ev, stem in sorted(ev_stems.items()):
        path = os.path.join(args.out, stem + ".json")
        if os.path.exists(path):
            with open(path) as f:
                points.append((ev, json.load(f)["extinction_rate"]))
    if len(points) >= 2:
        plot_energy_response(points, args.chosen_ev,
                             os.path.join(fig_dir, "step_a_energy_response"))


if __name__ == "__main__":
    main()
