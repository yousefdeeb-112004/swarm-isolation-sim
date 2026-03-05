"""
Section 12 — Extended Experiment Suite for Publication Readiness.

Defines six robustness experiment types required for peer review:

  1. Random Isolation         — compare random vs adventurousness selection
  2. Isolation Ratio Sweep    — 5%, 10%, 20%, 30% of population
  3. Isolation Duration Sweep — 50, 100, 200, 500 steps
  4. Resource Variation       — food-rich (300) vs food-scarce (50)
  5. No-Return Control        — isolate without ever returning agents
  6. Generation Length Sweep  — 250, 500, 1000, 2000 steps per generation

Components:
  ExperimentCondition  — parameterized description of one experimental condition
  EXPERIMENT_SUITE     — registry of all conditions grouped by experiment type
  ExtendedExperiment   — enhanced experiment supporting no-return and all params
  SweepRunner          — runs a full parameter sweep across conditions and seeds
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np

from swarm_sim.agents.agent import Agent
from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.world import World


# ======================================================================
# 1. ExperimentCondition — describes one parameterized condition
# ======================================================================

@dataclass
class ExperimentCondition:
    """
    Fully parameterized description of a single experimental condition.

    Each condition maps to one row in the publication's comparison tables.
    """
    name: str                         # Human-readable label (e.g. "ratio_20pct")
    experiment_type: str              # Group (e.g. "ratio_sweep", "duration_sweep")
    description: str = ""

    # Isolation parameters
    selection_criteria: str = "adventurousness"
    isolation_fraction: float = 0.2
    isolation_duration: int = 50
    isolation_frequency: int = 50
    no_return: bool = False           # If True, isolated agents never return

    # Simulation parameters
    num_generations: int = 5
    steps_per_generation: int = 1000

    # Environment overrides (None = use defaults)
    food_initial: Optional[int] = None
    food_max: Optional[int] = None

    def apply_to_config(self, config: SimulationConfig) -> SimulationConfig:
        """
        Apply this condition's overrides to a config, returning a new copy.
        """
        cfg = copy.deepcopy(config)
        cfg.experiment.isolation_duration = self.isolation_duration
        cfg.experiment.isolation_frequency = self.isolation_frequency
        cfg.experiment.selection_criteria = self.selection_criteria
        cfg.world.max_steps = self.steps_per_generation

        if self.food_initial is not None:
            cfg.environment.food.initial_count = self.food_initial
        if self.food_max is not None:
            cfg.environment.food.max_food = self.food_max

        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize condition to a dict for logging."""
        return {
            "name": self.name,
            "experiment_type": self.experiment_type,
            "selection_criteria": self.selection_criteria,
            "isolation_fraction": self.isolation_fraction,
            "isolation_duration": self.isolation_duration,
            "isolation_frequency": self.isolation_frequency,
            "no_return": self.no_return,
            "num_generations": self.num_generations,
            "steps_per_generation": self.steps_per_generation,
            "food_initial": self.food_initial,
            "food_max": self.food_max,
        }

    def __repr__(self) -> str:
        tag = f"no_return" if self.no_return else f"frac={self.isolation_fraction}"
        return f"Condition({self.name}, {tag})"


# ======================================================================
# 2. EXPERIMENT_SUITE — registry of all conditions
# ======================================================================

def build_experiment_suite(
    num_generations: int = 5,
    steps_per_generation: int = 1000,
) -> Dict[str, List[ExperimentCondition]]:
    """
    Build the full suite of robustness experiments.

    Returns dict mapping experiment_type → list of conditions.
    """
    suite: Dict[str, List[ExperimentCondition]] = {}

    # --- Experiment 1: Random vs Adventurousness Selection ---
    suite["selection_criteria"] = [
        ExperimentCondition(
            name="select_adventurousness",
            experiment_type="selection_criteria",
            description="Isolate most adventurous agents (baseline)",
            selection_criteria="adventurousness",
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
        ExperimentCondition(
            name="select_random",
            experiment_type="selection_criteria",
            description="Isolate randomly selected agents",
            selection_criteria="random",
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
        ExperimentCondition(
            name="select_affiliation",
            experiment_type="selection_criteria",
            description="Isolate most affiliative agents",
            selection_criteria="affiliation_need",
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
        ExperimentCondition(
            name="select_best_fitness",
            experiment_type="selection_criteria",
            description="Isolate highest-fitness agents",
            selection_criteria="best_fitness",
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
    ]

    # --- Experiment 2: Isolation Ratio Sweep ---
    suite["ratio_sweep"] = [
        ExperimentCondition(
            name=f"ratio_{int(frac*100)}pct",
            experiment_type="ratio_sweep",
            description=f"Isolate {int(frac*100)}% of population",
            isolation_fraction=frac,
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        )
        for frac in [0.05, 0.10, 0.20, 0.30]
    ]

    # --- Experiment 3: Isolation Duration Sweep ---
    suite["duration_sweep"] = [
        ExperimentCondition(
            name=f"duration_{dur}",
            experiment_type="duration_sweep",
            description=f"Isolate for {dur} steps",
            isolation_duration=dur,
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        )
        for dur in [50, 100, 200, 500]
    ]

    # --- Experiment 4: Resource Variation ---
    suite["resource_variation"] = [
        ExperimentCondition(
            name="food_scarce",
            experiment_type="resource_variation",
            description="Food-scarce environment (50 initial, 80 max)",
            food_initial=50,
            food_max=80,
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
        ExperimentCondition(
            name="food_default",
            experiment_type="resource_variation",
            description="Default food level (150 initial, 200 max)",
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
        ExperimentCondition(
            name="food_rich",
            experiment_type="resource_variation",
            description="Food-rich environment (300 initial, 400 max)",
            food_initial=300,
            food_max=400,
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
    ]

    # --- Experiment 5: No-Return Control ---
    suite["no_return"] = [
        ExperimentCondition(
            name="with_return",
            experiment_type="no_return",
            description="Standard isolation with return (baseline)",
            no_return=False,
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
        ExperimentCondition(
            name="no_return",
            experiment_type="no_return",
            description="Isolate agents permanently (never return)",
            no_return=True,
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
        ),
    ]

    # --- Experiment 6: Generation Length Sweep ---
    suite["generation_length"] = [
        ExperimentCondition(
            name=f"gen_{length}_steps",
            experiment_type="generation_length",
            description=f"{length} steps per generation",
            steps_per_generation=length,
            num_generations=num_generations,
        )
        for length in [250, 500, 1000, 2000]
    ]

    return suite


# ======================================================================
# 3. ExtendedExperiment — enhanced with no-return and parameterization
# ======================================================================

class ExtendedExperiment:
    """
    Runs a single experimental condition (control vs treatment).

    Extends the core isolation experiment with:
    - No-return mode (agents isolated permanently)
    - Parameterized config via ExperimentCondition
    - Structured result output for aggregation
    """

    def __init__(self, condition: ExperimentCondition, base_config: SimulationConfig):
        self.condition = condition
        self.base_config = base_config
        self.config = condition.apply_to_config(base_config)

        self.control_world: Optional[World] = None
        self.treatment_world: Optional[World] = None
        self._isolated_agents: Dict[int, int] = {}  # id -> return_step

        # Tracking
        self.ctrl_step_metrics: List[Dict[str, Any]] = []
        self.treat_step_metrics: List[Dict[str, Any]] = []
        self.ctrl_gen_records: List[Dict[str, Any]] = []
        self.treat_gen_records: List[Dict[str, Any]] = []
        self.isolation_events: List[Dict[str, Any]] = []

        # Extinction tracking
        self.ctrl_extinction_step: Optional[int] = None
        self.treat_extinction_step: Optional[int] = None

    def setup(self) -> None:
        """Create mirrored control and treatment worlds."""
        self.control_world = World(self.config)
        self.treatment_world = World(self.config)
        self._isolated_agents.clear()
        self.isolation_events.clear()
        self.ctrl_step_metrics.clear()
        self.treat_step_metrics.clear()
        self.ctrl_extinction_step = None
        self.treat_extinction_step = None

    # ------------------------------------------------------------------
    # Agent selection
    # ------------------------------------------------------------------

    def _select_agents(self, world: World, count: int) -> List[Agent]:
        """Select agents for isolation based on condition criteria."""
        eligible = [a for a in world.agents if a.alive and not a.is_isolated]
        if not eligible or count <= 0:
            return []
        count = min(count, len(eligible))

        criteria = self.condition.selection_criteria

        if criteria == "random":
            rng = np.random.default_rng(42 + len(self.isolation_events))
            indices = rng.choice(len(eligible), size=count, replace=False)
            return [eligible[i] for i in indices]

        sort_map = {
            "adventurousness": lambda a: a.genome["adventurousness"],
            "affiliation_need": lambda a: a.genome["affiliation_need"],
            "xenophobia": lambda a: a.genome["xenophobia"],
            "best_fitness": lambda a: a.compute_fitness(),
            "worst_fitness": lambda a: -a.compute_fitness(),
        }
        key_fn = sort_map.get(criteria, lambda a: a.genome.get("adventurousness", 0))
        eligible.sort(key=key_fn, reverse=True)
        return eligible[:count]

    # ------------------------------------------------------------------
    # Isolation mechanics
    # ------------------------------------------------------------------

    def _apply_isolation(self, world: World, step: int) -> int:
        """Isolate selected agents in the treatment world."""
        living = [a for a in world.agents if a.alive and not a.is_isolated]
        n = int(len(living) * self.condition.isolation_fraction)
        if n <= 0:
            return 0
        selected = self._select_agents(world, n)

        count = 0
        rng = np.random.default_rng(step + len(self.isolation_events))
        for agent in selected:
            # Scatter to random positions (not all at one cell)
            iso_x = int(rng.integers(0, self.config.world.width))
            iso_y = int(rng.integers(0, self.config.world.height))
            agent.isolate(iso_x, iso_y)
            if self.condition.no_return:
                # Sentinel: never return (step = infinity)
                self._isolated_agents[agent.id] = float("inf")
            else:
                self._isolated_agents[agent.id] = step + self.condition.isolation_duration

            self.isolation_events.append({
                "step": step,
                "agent_id": agent.id,
                "action": "isolated",
                "criteria": self.condition.selection_criteria,
                "no_return": self.condition.no_return,
                "agent_energy": round(agent.energy, 2),
            })
            count += 1
        return count

    def _return_agents(self, world: World, step: int) -> int:
        """Return agents whose isolation has expired (skipped in no-return mode)."""
        if self.condition.no_return:
            return 0

        to_return = [
            aid for aid, ret_step in self._isolated_agents.items()
            if step >= ret_step
        ]
        count = 0
        for agent_id in to_return:
            agent = self._find_agent(world, agent_id)
            if agent and agent.alive and agent.is_isolated:
                rng = np.random.default_rng(agent_id + step)
                rx = int(rng.integers(5, self.config.world.width - 5))
                ry = int(rng.integers(5, self.config.world.height - 5))
                agent.return_from_isolation(rx, ry)

                self.isolation_events.append({
                    "step": step,
                    "agent_id": agent_id,
                    "action": "returned",
                    "agent_energy": round(agent.energy, 2),
                })
                count += 1
            del self._isolated_agents[agent_id]
        return count

    def _find_agent(self, world: World, agent_id: int) -> Optional[Agent]:
        for a in world.agents:
            if a.id == agent_id:
                return a
        return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, seed: int, verbose: bool = False) -> Dict[str, Any]:
        """
        Run a single-seed experiment for this condition.

        Parameters
        ----------
        seed : random seed for this run
        verbose : print per-generation summaries

        Returns structured results dict.
        """
        self.config.world.seed = seed
        Agent.reset_id_counter()
        self.setup()

        cond = self.condition
        global_step = 0

        for gen in range(cond.num_generations):
            self._isolated_agents.clear()

            for step in range(1, cond.steps_per_generation + 1):
                global_step += 1

                ctrl_m = self.control_world.step()
                treat_m = self.treatment_world.step()

                # Treatment: isolation schedule
                iso_count = 0
                ret_count = 0
                if step % cond.isolation_frequency == 0:
                    iso_count = self._apply_isolation(self.treatment_world, step)
                ret_count = self._return_agents(self.treatment_world, step)

                # Track extinction steps
                if (self.ctrl_extinction_step is None and
                        ctrl_m["agents_alive"] == 0):
                    self.ctrl_extinction_step = global_step
                if (self.treat_extinction_step is None and
                        treat_m["agents_alive"] == 0):
                    self.treat_extinction_step = global_step

                treat_m["isolated_count"] = iso_count
                treat_m["returned_count"] = ret_count
                treat_m["currently_isolated"] = len(self._isolated_agents)
                self.ctrl_step_metrics.append(ctrl_m)
                self.treat_step_metrics.append(treat_m)

                if ctrl_m["agents_alive"] == 0 and treat_m["agents_alive"] == 0:
                    break

            # Record generation summary
            self._record_generation(gen)

            if verbose:
                c = self.ctrl_gen_records[-1]
                t = self.treat_gen_records[-1]
                print(f"  Gen {gen} | Ctrl: fit={c['avg_fitness']:.4f} "
                      f"alive={c['alive_at_end']} food={c['total_food_eaten']} | "
                      f"Treat: fit={t['avg_fitness']:.4f} "
                      f"alive={t['alive_at_end']} food={t['total_food_eaten']}")

            # Evolve if not last generation
            if gen < cond.num_generations - 1:
                self.control_world.evolve()
                self.treatment_world.evolve()

        return self._compile_results(seed)

    def _record_generation(self, gen: int) -> None:
        """Record end-of-generation summary for both worlds."""
        for world, records, label in [
            (self.control_world, self.ctrl_gen_records, "control"),
            (self.treatment_world, self.treat_gen_records, "treatment"),
        ]:
            pop = world.get_population_stats()
            records.append({
                "generation": gen,
                "label": label,
                "alive_at_end": pop.get("alive", 0),
                "avg_fitness": pop.get("avg_fitness", 0),
                "best_fitness": pop.get("best_fitness", 0),
                "genome_diversity": pop.get("genome_diversity", 0),
                "total_food_eaten": pop.get("total_food_eaten", 0),
                "total_offspring": pop.get("total_offspring", 0),
                "avg_energy": pop.get("avg_energy", 0),
            })

    def _compile_results(self, seed: int) -> Dict[str, Any]:
        """Compile structured results for this run."""
        ctrl_fits = [r["avg_fitness"] for r in self.ctrl_gen_records]
        treat_fits = [r["avg_fitness"] for r in self.treat_gen_records]
        ctrl_food = sum(r["total_food_eaten"] for r in self.ctrl_gen_records)
        treat_food = sum(r["total_food_eaten"] for r in self.treat_gen_records)
        ctrl_alive_final = self.ctrl_gen_records[-1]["alive_at_end"] if self.ctrl_gen_records else 0
        treat_alive_final = self.treat_gen_records[-1]["alive_at_end"] if self.treat_gen_records else 0
        total_steps = self.condition.num_generations * self.condition.steps_per_generation

        # Treatment extinction: did it go extinct at any point?
        treat_extinct = self.treat_extinction_step is not None

        return {
            "condition": self.condition.to_dict(),
            "seed": seed,

            # Fitness
            "ctrl_avg_fitness": _safe_mean(ctrl_fits),
            "treat_avg_fitness": _safe_mean(treat_fits),
            "fitness_impact": _safe_mean(treat_fits) - _safe_mean(ctrl_fits),

            # Survival
            "ctrl_alive_final": ctrl_alive_final,
            "treat_alive_final": treat_alive_final,
            "treat_extinct": treat_extinct,
            "ctrl_extinction_step": self.ctrl_extinction_step,
            "treat_extinction_step": self.treat_extinction_step,

            # Food
            "ctrl_total_food": ctrl_food,
            "treat_total_food": treat_food,
            "food_reduction_pct": (
                (1 - treat_food / max(ctrl_food, 1)) * 100
                if ctrl_food > 0 else 0.0
            ),

            # Diversity
            "ctrl_diversity_final": (
                self.ctrl_gen_records[-1]["genome_diversity"]
                if self.ctrl_gen_records else 0
            ),
            "treat_diversity_final": (
                self.treat_gen_records[-1]["genome_diversity"]
                if self.treat_gen_records else 0
            ),

            # Isolation stats
            "total_isolations": len([
                e for e in self.isolation_events if e["action"] == "isolated"
            ]),
            "total_returns": len([
                e for e in self.isolation_events if e["action"] == "returned"
            ]),

            # Per-generation records (for downstream analysis)
            "ctrl_gen_records": self.ctrl_gen_records,
            "treat_gen_records": self.treat_gen_records,
        }


# ======================================================================
# 4. SweepRunner — executes full parameter sweeps (serial + parallel)
# ======================================================================

def _run_single_experiment(args: tuple) -> Dict[str, Any]:
    """
    Top-level function for multiprocessing.

    Must be defined at module level (not a method) so it can be pickled.
    Each worker process gets its own memory space, so Agent ID counters
    are independent.
    """
    condition, base_config, seed = args
    Agent.reset_id_counter()
    exp = ExtendedExperiment(condition, base_config)
    return exp.run(seed=seed, verbose=False)


class SweepRunner:
    """
    Runs a full parameter sweep across conditions and seeds.

    Supports both serial and parallel (multiprocessing) execution.
    Each (condition, seed) pair is fully independent, so parallelism
    scales linearly with CPU cores.

    Usage:
        runner = SweepRunner(conditions, config, num_seeds=30)
        # Serial (1 core):
        results = runner.run(verbose=True)
        # Parallel (all cores):
        results = runner.run(verbose=True, workers=0)
        # Parallel (4 cores):
        results = runner.run(verbose=True, workers=4)
    """

    def __init__(
        self,
        conditions: List[ExperimentCondition],
        base_config: SimulationConfig,
        seeds: Optional[List[int]] = None,
        num_seeds: int = 10,
    ):
        self.conditions = conditions
        self.base_config = base_config

        if seeds is not None:
            self.seeds = seeds
        else:
            # Well-spaced deterministic seeds
            self.seeds = [1000 + i * 137 for i in range(num_seeds)]

        self.all_results: List[Dict[str, Any]] = []

    def run(
        self,
        verbose: bool = False,
        workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute all condition × seed combinations.

        Parameters
        ----------
        verbose : print progress and summaries
        workers : number of parallel processes.
                  None or 1 = serial (old behavior).
                  0 = use all available CPU cores.
                  N > 1 = use N cores.

        Returns structured sweep results.
        """
        use_parallel = workers is not None and workers != 1
        if use_parallel:
            return self._run_parallel(verbose=verbose, workers=workers)
        else:
            return self._run_serial(verbose=verbose)

    # ------------------------------------------------------------------
    # Serial execution (original behavior)
    # ------------------------------------------------------------------

    def _run_serial(self, verbose: bool = False) -> Dict[str, Any]:
        """Execute sequentially — one run at a time."""
        total = len(self.conditions) * len(self.seeds)
        completed = 0
        start_time = time.time()

        for cond in self.conditions:
            cond_results = []

            if verbose:
                print(f"\n{'='*60}")
                print(f"  Condition: {cond.name}")
                print(f"  {cond.description}")
                print(f"{'='*60}")

            for seed in self.seeds:
                completed += 1
                if verbose:
                    print(f"  [{completed}/{total}] seed={seed}", end="", flush=True)

                exp = ExtendedExperiment(cond, self.base_config)
                result = exp.run(seed=seed, verbose=False)
                cond_results.append(result)
                self.all_results.append(result)

                if verbose:
                    fi = result["fitness_impact"]
                    te = "EXTINCT" if result["treat_extinct"] else "survived"
                    print(f" → impact={fi:+.4f}, treatment={te}")

            if verbose:
                self._print_condition_summary(cond.name, cond_results)

        elapsed = time.time() - start_time

        return {
            "sweep_summary": self.get_summary(),
            "all_results": self.all_results,
            "elapsed_seconds": round(elapsed, 1),
            "total_runs": total,
        }

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        verbose: bool = False,
        workers: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute using multiprocessing.

        Each (condition, seed) pair runs in a separate process.
        """
        import multiprocessing as mp

        if workers == 0:
            workers = mp.cpu_count()
        workers = max(1, min(workers, mp.cpu_count()))

        # Build all jobs: list of (condition, base_config, seed) tuples
        jobs = []
        for cond in self.conditions:
            for seed in self.seeds:
                jobs.append((cond, self.base_config, seed))

        total = len(jobs)

        if verbose:
            print(f"\n[*] Parallel execution: {workers} workers, {total} jobs")
            print(f"[*] Conditions: {len(self.conditions)}, Seeds: {len(self.seeds)}")

        start_time = time.time()

        # Use Pool with imap_unordered for progress tracking
        completed = 0
        results_list = []

        with mp.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(_run_single_experiment, jobs):
                results_list.append(result)
                completed += 1

                if verbose:
                    cname = result["condition"]["name"]
                    seed = result["seed"]
                    fi = result["fitness_impact"]
                    te = "EXTINCT" if result["treat_extinct"] else "survived"
                    print(f"  [{completed}/{total}] {cname} seed={seed} "
                          f"→ impact={fi:+.4f}, {te}")

        # Sort results by condition name then seed for deterministic ordering
        results_list.sort(
            key=lambda r: (r["condition"]["name"], r["seed"])
        )
        self.all_results = results_list

        elapsed = time.time() - start_time

        if verbose:
            # Print per-condition summaries
            by_cond: Dict[str, list] = {}
            for r in self.all_results:
                name = r["condition"]["name"]
                by_cond.setdefault(name, []).append(r)
            for name in sorted(by_cond.keys()):
                self._print_condition_summary(name, by_cond[name])

            speedup_est = total * 5  # rough: ~5s per run serial
            print(f"\n[*] Parallel finished in {elapsed:.1f}s using {workers} workers")

        return {
            "sweep_summary": self.get_summary(),
            "all_results": self.all_results,
            "elapsed_seconds": round(elapsed, 1),
            "total_runs": total,
            "workers": workers,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Aggregate results by condition.

        Returns dict mapping condition_name → aggregated statistics.
        """
        by_condition: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.all_results:
            name = r["condition"]["name"]
            by_condition.setdefault(name, []).append(r)

        summary = {}
        for name, results in by_condition.items():
            n = len(results)
            fitness_impacts = [r["fitness_impact"] for r in results]
            ctrl_fits = [r["ctrl_avg_fitness"] for r in results]
            treat_fits = [r["treat_avg_fitness"] for r in results]
            food_reductions = [r["food_reduction_pct"] for r in results]
            extinction_rate = sum(1 for r in results if r["treat_extinct"]) / max(n, 1)
            extinction_steps = [
                r["treat_extinction_step"] for r in results
                if r["treat_extinction_step"] is not None
            ]

            summary[name] = {
                "condition": results[0]["condition"],
                "n_seeds": n,
                "ctrl_fitness_mean": _safe_mean(ctrl_fits),
                "ctrl_fitness_std": _safe_std(ctrl_fits),
                "treat_fitness_mean": _safe_mean(treat_fits),
                "treat_fitness_std": _safe_std(treat_fits),
                "fitness_impact_mean": _safe_mean(fitness_impacts),
                "fitness_impact_std": _safe_std(fitness_impacts),
                "fitness_impact_ci95": _ci95(fitness_impacts),
                "food_reduction_mean": _safe_mean(food_reductions),
                "food_reduction_std": _safe_std(food_reductions),
                "extinction_rate": round(extinction_rate, 4),
                "extinction_step_mean": _safe_mean(extinction_steps) if extinction_steps else None,
                "extinction_step_std": _safe_std(extinction_steps) if extinction_steps else None,
            }

        return summary

    def get_results_for_condition(self, name: str) -> List[Dict[str, Any]]:
        """Get all per-seed results for a condition name."""
        return [r for r in self.all_results
                if r["condition"]["name"] == name]

    def _print_condition_summary(
        self, name: str, results: List[Dict[str, Any]]
    ) -> None:
        """Print summary for one condition."""
        n = len(results)
        impacts = [r["fitness_impact"] for r in results]
        ext_rate = sum(1 for r in results if r["treat_extinct"]) / max(n, 1)
        food_red = [r["food_reduction_pct"] for r in results]
        print(f"  ────────────────────────────────")
        print(f"  Summary ({n} seeds):")
        print(f"    Fitness impact: {_safe_mean(impacts):+.4f} "
              f"± {_ci95(impacts):.4f}")
        print(f"    Extinction rate: {ext_rate:.0%}")
        print(f"    Food reduction: {_safe_mean(food_red):.1f}% "
              f"± {_ci95(food_red):.1f}%")


# ======================================================================
# 5. Convenience: run a single experiment type from the suite
# ======================================================================

def run_experiment_type(
    experiment_type: str,
    base_config: SimulationConfig,
    num_seeds: int = 10,
    num_generations: int = 5,
    steps_per_generation: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function: run all conditions for one experiment type.

    Parameters
    ----------
    experiment_type : one of 'selection_criteria', 'ratio_sweep',
                      'duration_sweep', 'resource_variation', 'no_return',
                      'generation_length'
    """
    suite = build_experiment_suite(num_generations, steps_per_generation)
    if experiment_type not in suite:
        raise ValueError(
            f"Unknown experiment type: {experiment_type}. "
            f"Valid: {list(suite.keys())}"
        )

    conditions = suite[experiment_type]
    runner = SweepRunner(conditions, base_config, num_seeds=num_seeds)
    return runner.run(verbose=verbose)


def list_experiment_types() -> List[str]:
    """Return all available experiment type names."""
    return list(build_experiment_suite().keys())


def list_all_conditions() -> List[ExperimentCondition]:
    """Return a flat list of all conditions across all experiment types."""
    suite = build_experiment_suite()
    return [c for conditions in suite.values() for c in conditions]


# ======================================================================
# Helpers
# ======================================================================

def _safe_mean(values: list) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _ci95(values: list) -> float:
    import math
    n = len(values)
    if n < 2:
        return 0.0
    return 1.96 * _safe_std(values) / math.sqrt(n)