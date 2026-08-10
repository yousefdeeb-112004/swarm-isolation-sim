#!/usr/bin/env python3
"""
Control-baseline measurement for Phase 1, Step A.

Runs the CONTROL condition only (no isolation) for N seeds x G generations,
mirroring exactly the control world used inside
swarm_sim.experiments.extended.ExtendedExperiment:

  - World(config), stepped for `steps_per_generation` steps per generation
  - population evaluated with World.get_population_stats() at generation end
    (avg_fitness over LIVING agents), matching ctrl_avg_fitness in the sweep
  - World.evolve() between generations
  - a run is "extinct" if agents_alive hits 0 at any global step

Reports, aggregated across seeds:
  - extinction rate  (fraction of runs that hit 0 alive at some point)
  - Kaplan-Meier median survival (global step), via analysis.stats_analysis
  - final mean fitness (mean of per-generation avg_fitness; also last-gen)
  - population trajectory (mean alive per global step across seeds)

Deterministic: uses the same seed schedule as SweepRunner
(seeds = [1000 + 137*i]) unless --seeds-list is given.

Usage:
  python scripts/run_control_baseline.py --config configs/default.yaml \
      --seeds 30 --generations 5 --workers 0 \
      --out data/results/phase1 --label default
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.world import World
from swarm_sim.agents.agent import Agent
from swarm_sim.analysis.stats_analysis import kaplan_meier


def _run_one_control(args: tuple) -> Dict[str, Any]:
    """Run a single control world through G generations for one seed.

    Top-level (picklable) function for multiprocessing.
    """
    config, seed, num_generations, steps_per_generation = args

    config = config  # already a deep-copyable SimulationConfig
    config.world.seed = seed
    Agent.reset_id_counter()
    world = World(config)

    # alive[gen, step] = number alive after executing step (0 if not reached)
    alive_grid = np.zeros((num_generations, steps_per_generation), dtype=np.int32)
    gen_fitness: List[float] = []
    gen_alive_end: List[int] = []

    extinction_step: Optional[int] = None
    global_step = 0

    for gen in range(num_generations):
        for s in range(1, steps_per_generation + 1):
            m = world.step()
            global_step += 1
            alive = m["agents_alive"]
            alive_grid[gen, s - 1] = alive
            if extinction_step is None and alive == 0:
                extinction_step = global_step
            if alive == 0:
                break

        pop = world.get_population_stats()
        gen_fitness.append(float(pop.get("avg_fitness", 0.0)))
        gen_alive_end.append(int(pop.get("alive", 0)))

        if gen < num_generations - 1:
            world.evolve()

    return {
        "seed": seed,
        "extinction_step": extinction_step,
        "extinct": extinction_step is not None,
        "gen_fitness": gen_fitness,
        "gen_alive_end": gen_alive_end,
        "mean_fitness": float(np.mean(gen_fitness)) if gen_fitness else 0.0,
        "final_gen_fitness": gen_fitness[-1] if gen_fitness else 0.0,
        "alive_grid": alive_grid.tolist(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure control-condition viability")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config path (default: built-in defaults)")
    p.add_argument("--seeds", type=int, default=30, help="Number of seeds")
    p.add_argument("--seeds-list", type=str, default=None,
                   help="Comma-separated explicit seed list (overrides --seeds)")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--steps", type=int, default=None,
                   help="Steps per generation (default: config world.max_steps)")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0=all cores, 1=serial)")
    p.add_argument("--out", type=str, default="data/results/phase1")
    p.add_argument("--label", type=str, default="control",
                   help="Label/prefix for output files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        config = SimulationConfig.from_yaml(args.config)
        cfg_src = args.config
    else:
        config = SimulationConfig.default()
        cfg_src = "built-in defaults"

    steps_per_gen = args.steps or config.world.max_steps
    num_gen = args.generations

    if args.seeds_list:
        seeds = [int(s) for s in args.seeds_list.split(",") if s.strip()]
    else:
        seeds = [1000 + 137 * i for i in range(args.seeds)]

    print(f"[*] Control baseline")
    print(f"    config           : {cfg_src}")
    print(f"    seeds            : {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"    generations      : {num_gen}")
    print(f"    steps/generation : {steps_per_gen}")
    print(f"    max_step (censor): {num_gen * steps_per_gen}")

    jobs = [
        (SimulationConfig.from_yaml(args.config) if args.config
         else SimulationConfig.default(),
         seed, num_gen, steps_per_gen)
        for seed in seeds
    ]

    t0 = time.time()
    workers = args.workers
    if workers is not None and workers != 1:
        import multiprocessing as mp
        n = mp.cpu_count() if workers == 0 else min(workers, mp.cpu_count())
        print(f"    workers          : {n} (parallel)")
        with mp.Pool(processes=n) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_run_one_control, jobs), 1):
                te = "EXTINCT@%s" % r["extinction_step"] if r["extinct"] else "survived"
                print(f"    [{i}/{len(jobs)}] seed={r['seed']} "
                      f"meanfit={r['mean_fitness']:.4f} {te}")
                results.append(r)
    else:
        print(f"    workers          : 1 (serial)")
        results = []
        for i, job in enumerate(jobs, 1):
            r = _run_one_control(job)
            te = "EXTINCT@%s" % r["extinction_step"] if r["extinct"] else "survived"
            print(f"    [{i}/{len(jobs)}] seed={r['seed']} "
                  f"meanfit={r['mean_fitness']:.4f} {te}")
            results.append(r)

    results.sort(key=lambda r: r["seed"])
    elapsed = time.time() - t0

    summarize_and_save(results, seeds, num_gen, steps_per_gen,
                       args.label, args.out, cfg_src, elapsed)


def summarize_and_save(results, seeds, num_gen, steps_per_gen,
                       label, out, cfg_src, elapsed):
    """Aggregate per-seed control results, write JSON + traj.npz, print summary.

    Reusable by both the CLI entry point and the calibration bracket driver so
    every candidate produces byte-identical output structure.
    Returns the summary dict.
    """
    results = sorted(results, key=lambda r: r["seed"])
    n = len(results)
    n_extinct = sum(1 for r in results if r["extinct"])
    ext_rate = n_extinct / max(n, 1)
    ext_steps = [r["extinction_step"] for r in results]

    max_step = num_gen * steps_per_gen
    km = kaplan_meier(ext_steps, max_step, label=label)

    mean_fits = [r["mean_fitness"] for r in results]
    final_fits = [r["final_gen_fitness"] for r in results]

    # Mean population trajectory across seeds (mean alive per global step)
    grids = np.array([r["alive_grid"] for r in results], dtype=float)  # (n, gen, steps)
    mean_traj = grids.mean(axis=0).reshape(-1)  # flattened global-step trajectory

    summary = {
        "label": label,
        "config_source": cfg_src,
        "n_seeds": n,
        "num_generations": num_gen,
        "steps_per_generation": steps_per_gen,
        "seeds": seeds,
        "extinction_rate": round(ext_rate, 4),
        "n_extinct": n_extinct,
        "extinction_steps": ext_steps,
        "km_median_survival": km["median_survival"],
        "km_n_events": km["n_events"],
        "km_n_censored": km["n_censored"],
        "mean_fitness_mean": round(float(np.mean(mean_fits)), 6),
        "mean_fitness_std": round(float(np.std(mean_fits, ddof=1)) if n > 1 else 0.0, 6),
        "final_gen_fitness_mean": round(float(np.mean(final_fits)), 6),
        "final_gen_fitness_std": round(float(np.std(final_fits, ddof=1)) if n > 1 else 0.0, 6),
        "per_seed": [
            {"seed": r["seed"], "extinct": r["extinct"],
             "extinction_step": r["extinction_step"],
             "mean_fitness": round(r["mean_fitness"], 6),
             "final_gen_fitness": round(r["final_gen_fitness"], 6),
             "gen_alive_end": r["gen_alive_end"]}
            for r in results
        ],
        "elapsed_seconds": round(elapsed, 1),
    }

    os.makedirs(out, exist_ok=True)
    summary_path = os.path.join(out, f"control_baseline_{label}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save mean trajectory + KM curve as npz for the report figure
    traj_path = os.path.join(out, f"control_baseline_{label}_traj.npz")
    np.savez(
        traj_path,
        mean_traj=mean_traj,
        km_times=np.array(km["times"], dtype=float),
        km_survival=np.array(km["survival"], dtype=float),
        km_ci_lower=np.array(km["ci_lower"], dtype=float),
        km_ci_upper=np.array(km["ci_upper"], dtype=float),
        steps_per_generation=steps_per_gen,
        num_generations=num_gen,
    )

    print(f"\n{'='*60}")
    print(f"  CONTROL BASELINE SUMMARY ({label})")
    print(f"{'='*60}")
    print(f"  Extinction rate       : {ext_rate:.1%}  ({n_extinct}/{n})")
    print(f"  KM median survival    : {km['median_survival']} "
          f"(events={km['n_events']}, censored={km['n_censored']}, max={max_step})")
    print(f"  Mean fitness (per-run): {summary['mean_fitness_mean']:.4f} "
          f"± {summary['mean_fitness_std']:.4f}")
    print(f"  Final-gen fitness     : {summary['final_gen_fitness_mean']:.4f} "
          f"± {summary['final_gen_fitness_std']:.4f}")
    print(f"  Elapsed               : {elapsed:.1f}s")
    print(f"\n[*] Saved: {summary_path}")
    print(f"[*] Saved: {traj_path}")
    return summary


if __name__ == "__main__":
    main()
