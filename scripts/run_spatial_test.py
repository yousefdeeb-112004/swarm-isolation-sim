#!/usr/bin/env python3
"""
Phase 1, Step D — driver for the three-arm spatial-redistribution test.

Runs control / isolation-30% / sham-relocation-30% (harshest-cell settings:
metabolism_discount = 0.0, predator_protection = False) across N seeds, each arm
mirrored on the same seed. Checkpointed (one JSONL line per seed) and resumable,
same conventions as the Step-B/Step-C drivers.

Usage:
  python scripts/run_spatial_test.py \
      --config configs/calibrated.yaml --seeds 30 --generations 5 \
      --steps 1000 --workers 0 --out data/results/phase1/spatial_test
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
from swarm_sim.experiments.spatial_test import run_single_spatial


def parse_args():
    p = argparse.ArgumentParser(description="Three-arm spatial-redistribution test")
    p.add_argument("--config", default="configs/calibrated.yaml")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--fraction", type=float, default=0.30)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default="data/results/phase1/spatial_test")
    return p.parse_args()


def _load_checkpoint(path):
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
                continue
            done[r["seed"]] = r
    return done


def main():
    args = parse_args()
    config = SimulationConfig.from_yaml(args.config)
    steps = args.steps or config.world.max_steps
    seeds = [1000 + 137 * i for i in range(args.seeds)]

    kwargs = dict(
        isolation_fraction=args.fraction,
        num_generations=args.generations,
        steps_per_generation=steps,
    )

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoint.jsonl")

    print("[*] Three-arm spatial-redistribution test (checkpointed)")
    print(f"    config     : {args.config} "
          f"(energy_value={config.environment.food.energy_value})")
    print(f"    arms       : control / isolation-{int(args.fraction*100)}% / "
          f"sham-relocation-{int(args.fraction*100)}%")
    print(f"    settings   : md=0.0, predator_protection=False (harshest cell)")
    print(f"    seeds      : {args.seeds}  gen x steps: {args.generations} x {steps}")
    print(f"    checkpoint : {ckpt_path}")

    done = _load_checkpoint(ckpt_path)
    print(f"    already done: {len(done)} seeds (resuming)")

    jobs = [(config, s, kwargs) for s in seeds if s not in done]
    total = len(seeds)
    print(f"    remaining  : {len(jobs)} seeds")

    if jobs:
        workers = args.workers or mp.cpu_count()
        workers = max(1, min(workers, mp.cpu_count()))
        print(f"    workers    : {workers}")

        t0 = time.time()
        completed = len(done)
        with open(ckpt_path, "a") as ckpt, mp.Pool(processes=workers) as pool:
            for r in pool.imap_unordered(run_single_spatial, jobs):
                ckpt.write(json.dumps(r, default=str) + "\n")
                ckpt.flush()
                os.fsync(ckpt.fileno())
                completed += 1
                elapsed = time.time() - t0
                rate = (completed - len(done)) / max(elapsed, 1e-9)
                eta_min = ((total - completed) / rate / 60) if rate > 0 else float("inf")
                fi = r["fitness_impact"]
                ext = {a: r["arms"][a]["extinct"] for a in ("control", "isolation", "sham")}
                print(f"    [{completed}/{total}] seed={r['seed']} "
                      f"impact iso={fi['isolation']:+.4f} sham={fi['sham']:+.4f} "
                      f"| ext c={int(ext['control'])} i={int(ext['isolation'])} "
                      f"s={int(ext['sham'])} | {elapsed/60:.1f}min, ETA {eta_min:.0f}min",
                      flush=True)
        print(f"[*] Run done in {(time.time()-t0)/60:.1f} min")
    else:
        print("[*] Nothing to run; all seeds already in checkpoint.")

    done = _load_checkpoint(ckpt_path)
    print(f"[*] Checkpoint holds {len(done)} seeds. "
          f"Analyze with scripts/analyze_spatial_test.py")


if __name__ == "__main__":
    main()
