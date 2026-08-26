#!/usr/bin/env python3
"""
Phase 1, Step C — density-relief mechanism test (checkpointed driver).

Re-runs ONLY the harshest factorial cell — metabolism_discount = 0.0,
predator_protection = False, isolation_ratio = 0.30 — at 30 seeds, N = 1000
(5 generations x 1000 steps). Each run pairs a 30%-isolation treatment world
against its own no-isolation control on the same seed (the control is captured
for free inside every ExtendedExperiment run).

This cell is where the Isolation Paradox is strongest AND where both engineered
protections for isolated agents are switched off, so any surviving benefit must
come from the isolation event itself. The candidate mechanism is density relief:
temporarily scattering a fraction of the swarm lowers local competition for the
agents left behind. Testing it needs step-resolved trajectories aligned to the
isolation schedule, which the Step-B factorial discarded. Those trajectories are
now persisted by ExtendedExperiment (downsampled every 5 steps) — see
`swarm_sim.experiments.extended.STEP_TRAJECTORY_DOWNSAMPLE`.

Conventions match the Step-B factorial driver: detached-friendly (flushed
progress), CHECKPOINTED to <out>/checkpoint.jsonl (JSONL, one line per completed
run), resumable by re-running the same command (finished (condition, seed) pairs
are skipped).

Usage:
  python scripts/run_density_relief.py \
      --config configs/calibrated.yaml --seeds 30 --generations 5 \
      --steps 1000 --workers 0 --out data/results/phase1/density_relief
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
from swarm_sim.experiments.extended import ExperimentCondition, _run_single_experiment


# The single harshest cell. Name matches the Step-B factorial condition exactly
# (mech_md00_ppoff_r30) so downstream tooling treats it identically.
def build_harshest_condition(num_generations: int,
                             steps_per_generation: int) -> ExperimentCondition:
    return ExperimentCondition(
        name="mech_md00_ppoff_r30",
        experiment_type="mechanism_sweep",
        description=("metab_discount=0.0, predator_protection=False, "
                     "isolation_ratio=30% (harshest cell)"),
        isolation_fraction=0.30,
        isolation_metabolism_discount=0.0,
        isolation_predator_protection=False,
        num_generations=num_generations,
        steps_per_generation=steps_per_generation,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Density-relief harshest-cell run")
    p.add_argument("--config", default="configs/calibrated.yaml")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000,
                   help="Steps per generation")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0=all cores)")
    p.add_argument("--out", default="data/results/phase1/density_relief")
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
            done[key] = r
    return done


def main():
    args = parse_args()
    config = SimulationConfig.from_yaml(args.config)
    steps = args.steps or config.world.max_steps

    cond = build_harshest_condition(args.generations, steps)
    # Same deterministic seed schedule as the factorial (subset if seeds<30).
    seeds = [1000 + 137 * i for i in range(args.seeds)]

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoint.jsonl")

    print(f"[*] Density-relief harshest-cell run (checkpointed)")
    print(f"    config      : {args.config} "
          f"(energy_value={config.environment.food.energy_value})")
    print(f"    condition   : {cond.name}  ({cond.description})")
    print(f"    seeds       : {args.seeds}  gen x steps: {args.generations} x {steps}")
    print(f"    total runs  : {len(seeds)}")
    print(f"    checkpoint  : {ckpt_path}")

    done = _load_checkpoint(ckpt_path)
    print(f"    already done: {len(done)} runs (resuming)")

    jobs = [(cond, config, seed) for seed in seeds
            if _job_key(cond.name, seed) not in done]
    total = len(seeds)
    print(f"    remaining   : {len(jobs)} runs")

    if jobs:
        workers = args.workers or mp.cpu_count()
        workers = max(1, min(workers, mp.cpu_count()))
        print(f"    workers     : {workers}")

        t0 = time.time()
        completed = len(done)
        with open(ckpt_path, "a") as ckpt, mp.Pool(processes=workers) as pool:
            for r in pool.imap_unordered(_run_single_experiment, jobs):
                ckpt.write(json.dumps(r, default=str) + "\n")
                ckpt.flush()
                os.fsync(ckpt.fileno())
                completed += 1
                elapsed = time.time() - t0
                rate = (completed - len(done)) / max(elapsed, 1e-9)
                remaining = total - completed
                eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
                te = "EXT" if r["treat_extinct"] else "ok"
                ce = "EXT" if r.get("ctrl_extinction_step") is not None else "ok"
                print(f"    [{completed}/{total}] seed={r['seed']} "
                      f"impact={r['fitness_impact']:+.4f} treat={te} ctrl={ce} "
                      f"| {elapsed/60:.1f}min elapsed, ETA {eta_min:.0f}min",
                      flush=True)
        print(f"[*] Run done in {(time.time()-t0)/60:.1f} min")
    else:
        print("[*] Nothing to run; all runs already in checkpoint.")

    done = _load_checkpoint(ckpt_path)
    print(f"[*] Checkpoint holds {len(done)} completed runs. "
          f"Analyze with scripts/analyze_density_relief.py")


if __name__ == "__main__":
    main()
