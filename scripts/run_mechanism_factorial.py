#!/usr/bin/env python3
"""
Phase 1, Step B — driver for the 2x4 mechanism factorial (checkpointed).

Runs the `mechanism_sweep` conditions (8 mechanism cells x 4 isolation ratios =
32 treatment conditions; the no-isolation control is captured per run) with the
existing multiprocessing infrastructure, CHECKPOINTING every completed run to
disk (JSONL) so an interruption resumes without redoing finished work.

After the sweep it produces:
  - raw CSVs (per-run, per-generation) + full JSON via BatchLogger
  - generic stats via analyze_sweep (t-tests, one-way ANOVA, KM)
  - mechanism-specific analysis via analyze_mechanism_sweep
      * per (discount, protection, ratio) cell: impact, extinction, Cohen's d,
        KM median, t-test
      * two-way ANOVA (discount x protection) at the 30% ratio and pooled
      * CSV (mechanism_cells.csv) + LaTeX (anova_two_way_*.tex)
  - figures: phase diagram + 2x4 dose-response (300-DPI PNG + PDF)
  - control extinction rate observed inside the factorial (over unique seeds)

Resume: re-running with the same --out reuses <out>/checkpoint.jsonl and only
runs the (condition, seed) pairs not already present.

Usage (band-faithful 1000 steps/gen):
  python scripts/run_mechanism_factorial.py \
      --config configs/calibrated.yaml --seeds 30 --generations 5 \
      --steps 1000 --workers 0 --out data/results/phase1/mechanism
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from swarm_sim.core.config import SimulationConfig
from swarm_sim.experiments.extended import (
    build_experiment_suite, SweepRunner, _run_single_experiment,
)
from swarm_sim.utils.batch_logger import BatchLogger
from swarm_sim.analysis.stats_analysis import analyze_sweep, analyze_mechanism_sweep
from swarm_sim.utils.pub_visualization import generate_mechanism_figures


def parse_args():
    p = argparse.ArgumentParser(description="Run the 2x4 mechanism factorial")
    p.add_argument("--config", default="configs/calibrated.yaml")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--steps", type=int, default=None,
                   help="Steps per generation (default: config world.max_steps)")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0=all cores)")
    p.add_argument("--out", default="data/results/phase1/mechanism")
    p.add_argument("--focus-ratio", type=float, default=0.30)
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip running; just (re)build analysis from checkpoint")
    return p.parse_args()


def _job_key(cond_name, seed):
    return f"{cond_name}::{seed}"


def _load_checkpoint(path):
    """Return {job_key: result} for completed runs in the checkpoint."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard kill
            key = _job_key(r["condition"]["name"], r["seed"])
            done[key] = r  # later duplicates overwrite (idempotent)
    return done


def main():
    args = parse_args()
    config = SimulationConfig.from_yaml(args.config)
    steps = args.steps or config.world.max_steps

    conditions = build_experiment_suite(
        num_generations=args.generations, steps_per_generation=steps,
    )["mechanism_sweep"]

    # Same deterministic seed schedule as SweepRunner (subset if seeds<30).
    seeds = [1000 + 137 * i for i in range(args.seeds)]

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoint.jsonl")

    print(f"[*] Mechanism factorial (checkpointed)")
    print(f"    config      : {args.config} "
          f"(energy_value={config.environment.food.energy_value})")
    print(f"    conditions  : {len(conditions)} (8 cells x 4 ratios)")
    print(f"    seeds       : {args.seeds}  gen x steps: {args.generations} x {steps}")
    print(f"    total runs  : {len(conditions) * len(seeds)}")
    print(f"    checkpoint  : {ckpt_path}")

    done = _load_checkpoint(ckpt_path)
    print(f"    already done: {len(done)} runs (resuming)")

    if not args.analyze_only:
        jobs = []
        for cond in conditions:
            for seed in seeds:
                if _job_key(cond.name, seed) not in done:
                    jobs.append((cond, config, seed))
        total = len(conditions) * len(seeds)
        print(f"    remaining   : {len(jobs)} runs")

        if jobs:
            workers = args.workers
            if workers == 0:
                workers = mp.cpu_count()
            workers = max(1, min(workers, mp.cpu_count()))
            print(f"    workers     : {workers}")

            t0 = time.time()
            completed = len(done)
            # Append-mode checkpoint; only the main process writes.
            with open(ckpt_path, "a") as ckpt, \
                    mp.Pool(processes=workers) as pool:
                for r in pool.imap_unordered(_run_single_experiment, jobs):
                    ckpt.write(json.dumps(r, default=str) + "\n")
                    ckpt.flush()
                    os.fsync(ckpt.fileno())
                    completed += 1
                    elapsed = time.time() - t0
                    rate = (completed - len(done)) / max(elapsed, 1e-9)
                    remaining = total - completed
                    eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
                    cname = r["condition"]["name"]
                    fi = r["fitness_impact"]
                    te = "EXT" if r["treat_extinct"] else "ok"
                    print(f"    [{completed}/{total}] {cname} seed={r['seed']} "
                          f"impact={fi:+.4f} {te} "
                          f"| {elapsed/60:.1f}min elapsed, ETA {eta_min:.0f}min",
                          flush=True)
            print(f"[*] Sweep done in {(time.time()-t0)/60:.1f} min")
        else:
            print("[*] Nothing to run; all runs already in checkpoint.")

    # ---- Assemble results from the checkpoint (authoritative) ----
    done = _load_checkpoint(ckpt_path)
    all_results = list(done.values())
    all_results.sort(key=lambda r: (r["condition"]["name"], r["seed"]))
    print(f"[*] Assembling analysis from {len(all_results)} checkpointed runs")

    # Reuse SweepRunner's aggregation without re-running anything.
    runner = SweepRunner(conditions, config, num_seeds=args.seeds)
    runner.all_results = all_results
    sweep_results = {
        "sweep_summary": runner.get_summary(),
        "all_results": all_results,
        "elapsed_seconds": None,
        "total_runs": len(all_results),
    }

    # Control extinction observed inside the factorial, over UNIQUE seeds
    # (control is identical across cells for a given seed).
    ctrl_by_seed = {}
    for r in all_results:
        s = r["seed"]
        ext = r.get("ctrl_extinction_step") is not None
        # all cells share the same control for a seed; keep any (consistent)
        ctrl_by_seed[s] = ctrl_by_seed.get(s, False) or ext
    n_seeds_seen = len(ctrl_by_seed)
    ctrl_ext_rate = (sum(1 for v in ctrl_by_seed.values() if v)
                     / max(n_seeds_seen, 1))
    print(f"[*] Control extinction inside factorial: "
          f"{ctrl_ext_rate:.1%} ({sum(ctrl_by_seed.values())}/{n_seeds_seen} seeds)")

    # Raw CSVs + JSON
    print("[*] BatchLogger raw export...")
    logger = BatchLogger(args.out)
    logger.export_sweep(sweep_results, prefix="mechanism_sweep")

    # Generic + mechanism-specific stats
    stats_dir = os.path.join(args.out, "stats")
    analyze_sweep(sweep_results, output_dir=stats_dir)
    mech = analyze_mechanism_sweep(sweep_results, output_dir=stats_dir,
                                   focus_ratio=args.focus_ratio)
    mech["control_extinction_in_factorial"] = round(ctrl_ext_rate, 4)
    mech["control_extinction_n_seeds"] = n_seeds_seen
    with open(os.path.join(stats_dir, "mechanism_analysis.json"), "w") as f:
        json.dump(mech, f, indent=2, default=str)

    for label, key in [("@%d%%" % int(args.focus_ratio * 100), "anova_focus_ratio"),
                       ("pooled", "anova_pooled")]:
        e = mech[key]["effects"]
        print(f"[*] Two-way ANOVA {label}:")
        for src in ["metabolism_discount", "predator_protection",
                    "metabolism_discount:predator_protection"]:
            print(f"      {src}: F={e[src]['F']}, p={e[src]['p_value']}, "
                  f"eta2={e[src]['partial_eta2']}")

    # Figures
    fig_dir = os.path.join(args.out, "figures")
    figs = generate_mechanism_figures(mech, fig_dir, prefix="mechanism")
    print(f"[*] Figures: {list(figs.values())}")
    print(f"[*] Done. Output in {args.out}/")


if __name__ == "__main__":
    main()
