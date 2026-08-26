"""
Phase 1, Step D — spatial-redistribution test (three-arm experiment).

Food in this model is globally saturated (pinned at ``max_food`` over a 100x100
grid), so scarcity is LOCAL and search-limited, not global (Step C). Isolated
agents keep foraging at full cost, so the only thing isolation actually does in
the harshest cell (metabolism_discount = 0.0, predator_protection = False) is
periodically RELOCATE a fraction of the swarm to random positions. This module
tests whether that spatial redistribution — not the "isolation" semantics — is
the active ingredient, by running three mirrored worlds on the same seed:

  - control    : no manipulation.
  - isolation  : every ``isolation_frequency`` steps, the most-adventurous
                 ``isolation_fraction`` of the (living, non-held) swarm is
                 teleported to random positions and flagged ``is_isolated``
                 (so it carries the loneliness signal and, in general cells, the
                 metabolism/predator subsidies — both OFF here), then teleported
                 back near centre after ``isolation_duration`` steps. This
                 reproduces the harshest factorial cell.
  - sham       : IDENTICAL selection, cadence and random scatter, but the agents
                 are NOT isolated — ``is_isolated`` stays False, no subsidies, no
                 loneliness boost, and on release they are NOT teleported back
                 (they simply keep acting from wherever they wandered). Sham
                 therefore strips isolation down to pure periodic relocation.

If sham reproduces the paradox, the mechanism is spatial redistribution and
isolation is merely its vehicle.

Per-step trajectories persisted (downsampled, reusing ``STEP_TRAJECTORY_DOWNSAMPLE``):
``agents_alive``, ``total_food`` (global standing stock), ``avg_energy``,
``avg_distance_between_agents`` (dispersion), ``local_food_mean`` (mean food in
each forager's sensor range), ``food_eaten_this_step``, and
``currently_relocated`` (held count in the isolation/sham arms). All are computed
in ``World.step`` or cheaply derived from state already present — no new
per-step simulation computation is added.
"""

from __future__ import annotations

import copy
from typing import Dict, Any, List, Optional

import numpy as np

from swarm_sim.agents.agent import Agent
from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.world import World
from swarm_sim.experiments.extended import (
    STEP_TRAJECTORY_DOWNSAMPLE, _safe_mean,
)

ARMS = ("control", "isolation", "sham")

# Per-step metric keys persisted for every arm.
_TRAJ_KEYS = (
    "agents_alive", "total_food", "avg_energy",
    "avg_distance_between_agents", "local_food_mean",
    "food_eaten_this_step", "currently_relocated",
)


class ThreeArmSpatialExperiment:
    """Run control / isolation / sham on a single shared seed."""

    def __init__(
        self,
        base_config: SimulationConfig,
        isolation_fraction: float = 0.30,
        isolation_frequency: int = 50,
        isolation_duration: int = 50,
        num_generations: int = 5,
        steps_per_generation: int = 1000,
        selection_criteria: str = "adventurousness",
    ):
        self.fraction = isolation_fraction
        self.frequency = isolation_frequency
        self.duration = isolation_duration
        self.num_generations = num_generations
        self.steps_per_generation = steps_per_generation
        self.selection_criteria = selection_criteria

        # Harshest-cell config: full metabolism, no predator immunity.
        cfg = copy.deepcopy(base_config)
        cfg.experiment.isolation_metabolism_discount = 0.0
        cfg.experiment.isolation_predator_protection = False
        cfg.experiment.isolation_duration = isolation_duration
        cfg.experiment.isolation_frequency = isolation_frequency
        cfg.experiment.selection_criteria = selection_criteria
        cfg.world.max_steps = steps_per_generation
        self.config = cfg

    # ------------------------------------------------------------------
    # Selection + relocation (shared by isolation and sham)
    # ------------------------------------------------------------------

    def _select(self, world: World, held: Dict[int, int]) -> List[Agent]:
        """Most-adventurous fraction of the living, not-currently-held agents."""
        eligible = [a for a in world.agents if a.alive and a.id not in held]
        n = int(len(eligible) * self.fraction)
        if n <= 0 or not eligible:
            return []
        eligible.sort(key=lambda a: a.genome["adventurousness"], reverse=True)
        return eligible[:n]

    def _relocate(self, world: World, step: int, held: Dict[int, int],
                  rng: np.random.Generator, isolate: bool) -> int:
        """Teleport the selected agents to random positions.

        isolate=True  -> flag is_isolated (isolation arm).
        isolate=False -> leave the agent fully normal (sham arm).
        """
        selected = self._select(world, held)
        w, h = self.config.world.width, self.config.world.height
        for a in selected:
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            if isolate:
                a.isolate(x, y)          # sets is_isolated=True
            else:
                a.x, a.y = x, y          # pure relocation, no isolation
            held[a.id] = step + self.duration
        return len(selected)

    def _return_isolated(self, world: World, step: int, held: Dict[int, int],
                         rng: np.random.Generator) -> None:
        """Isolation arm: teleport expired agents back near centre, unflag."""
        w, h = self.config.world.width, self.config.world.height
        for aid in [i for i, rs in held.items() if step >= rs]:
            a = next((x for x in world.agents if x.id == aid), None)
            if a is not None and a.alive and a.is_isolated:
                rx, ry = int(rng.integers(5, w - 5)), int(rng.integers(5, h - 5))
                a.return_from_isolation(rx, ry)
            del held[aid]

    @staticmethod
    def _release_sham(step: int, held: Dict[int, int]) -> None:
        """Sham arm: expiry only clears the hold — no teleport back."""
        for aid in [i for i, rs in held.items() if step >= rs]:
            del held[aid]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, seed: int) -> Dict[str, Any]:
        self.config.world.seed = seed
        Agent.reset_id_counter()
        worlds = {arm: World(self.config) for arm in ARMS}  # identical at start

        step_metrics: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
        gen_records: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
        ext_step: Dict[str, Optional[int]] = {a: None for a in ARMS}

        iso_hold: Dict[int, int] = {}
        sham_hold: Dict[int, int] = {}
        # Relocation RNGs: independent of the worlds' own streams, so choosing
        # destinations does not perturb simulation dynamics; deterministic in seed.
        iso_rng = np.random.default_rng(seed * 2 + 1)
        sham_rng = np.random.default_rng(seed * 2 + 2)

        global_step = 0
        for gen in range(self.num_generations):
            iso_hold.clear()
            sham_hold.clear()
            for step in range(1, self.steps_per_generation + 1):
                global_step += 1
                ms = {arm: worlds[arm].step() for arm in ARMS}

                # Isolation arm: apply then return.
                if step % self.frequency == 0:
                    self._relocate(worlds["isolation"], step, iso_hold,
                                   iso_rng, isolate=True)
                self._return_isolated(worlds["isolation"], step, iso_hold, iso_rng)

                # Sham arm: relocate (same cadence) then release.
                if step % self.frequency == 0:
                    self._relocate(worlds["sham"], step, sham_hold,
                                   sham_rng, isolate=False)
                self._release_sham(step, sham_hold)

                ms["control"]["currently_relocated"] = 0
                ms["isolation"]["currently_relocated"] = len(iso_hold)
                ms["sham"]["currently_relocated"] = len(sham_hold)

                for arm in ARMS:
                    if ext_step[arm] is None and ms[arm]["agents_alive"] == 0:
                        ext_step[arm] = global_step
                    step_metrics[arm].append(ms[arm])

                if all(ms[arm]["agents_alive"] == 0 for arm in ARMS):
                    break

            for arm in ARMS:
                self._record_gen(worlds[arm], gen_records[arm], gen, arm)

            if gen < self.num_generations - 1:
                for arm in ARMS:
                    worlds[arm].evolve()

        return self._compile(seed, step_metrics, gen_records, ext_step)

    @staticmethod
    def _record_gen(world: World, records: List[Dict[str, Any]],
                    gen: int, label: str) -> None:
        pop = world.get_population_stats()
        records.append({
            "generation": gen, "label": label,
            "alive_at_end": pop.get("alive", 0),
            "avg_fitness": pop.get("avg_fitness", 0),
            "best_fitness": pop.get("best_fitness", 0),
            "genome_diversity": pop.get("genome_diversity", 0),
            "total_food_eaten": pop.get("total_food_eaten", 0),
            "avg_energy": pop.get("avg_energy", 0),
        })

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def _downsample(self, step_metrics: Dict[str, List[Dict[str, Any]]]
                    ) -> Dict[str, Any]:
        stride = STEP_TRAJECTORY_DOWNSAMPLE
        n = min(len(step_metrics[a]) for a in ARMS)
        out: Dict[str, Any] = {"downsample_every": stride, "steps": []}
        if n == 0:
            for arm in ARMS:
                out[arm] = {k: [] for k in _TRAJ_KEYS}
            return out
        idx = list(range(stride - 1, n, stride))
        if not idx or idx[-1] != n - 1:
            idx.append(n - 1)
        out["steps"] = [i + 1 for i in idx]
        for arm in ARMS:
            m = step_metrics[arm]
            out[arm] = {k: [m[i].get(k, 0) for i in idx] for k in _TRAJ_KEYS}
        return out

    def _compile(self, seed, step_metrics, gen_records, ext_step) -> Dict[str, Any]:
        arms_out: Dict[str, Any] = {}
        for arm in ARMS:
            fits = [r["avg_fitness"] for r in gen_records[arm]]
            food = sum(r["total_food_eaten"] for r in gen_records[arm])
            alive_final = gen_records[arm][-1]["alive_at_end"] if gen_records[arm] else 0
            arms_out[arm] = {
                "avg_fitness": _safe_mean(fits),
                "alive_final": alive_final,
                "extinct": ext_step[arm] is not None,
                "extinction_step": ext_step[arm],
                "total_food": food,
                "gen_records": gen_records[arm],
            }
        ctrl_fit = arms_out["control"]["avg_fitness"]
        return {
            "seed": seed,
            "config": {
                "name": "spatial_md00_ppoff_r30",
                "isolation_fraction": self.fraction,
                "isolation_frequency": self.frequency,
                "isolation_duration": self.duration,
                "num_generations": self.num_generations,
                "steps_per_generation": self.steps_per_generation,
                "selection_criteria": self.selection_criteria,
                "isolation_metabolism_discount": 0.0,
                "isolation_predator_protection": False,
            },
            "arms": arms_out,
            "fitness_impact": {
                "isolation": arms_out["isolation"]["avg_fitness"] - ctrl_fit,
                "sham": arms_out["sham"]["avg_fitness"] - ctrl_fit,
            },
            "step_trajectories": self._downsample(step_metrics),
        }


def run_single_spatial(args: tuple) -> Dict[str, Any]:
    """Top-level worker for multiprocessing (must be picklable)."""
    base_config, seed, kwargs = args
    Agent.reset_id_counter()
    exp = ThreeArmSpatialExperiment(base_config, **kwargs)
    return exp.run(seed=seed)
