"""
Isolation Experiment Workflow — the core scientific experiment.

Compares a CONTROL group (no isolation) with a TREATMENT group
(systematic isolation of selected agents) across multiple generations.

Experiment design:
  - Both groups start from identical initial conditions (same seed)
  - Treatment group periodically isolates agents based on selection criteria
  - Isolation removes agent from the swarm for a fixed duration
  - After isolation, agent returns to a random position
  - Metrics are tracked per-step and per-generation for both conditions
  - Comparative analysis identifies the impact of isolation on:
      * Survival rates
      * Fitness evolution
      * Genome diversity
      * Social structure
      * Belief accuracy
      * RL policy quality

Selection criteria for isolation:
  - "adventurousness": most adventurous agents
  - "affiliation_need": most social agents
  - "random": random selection
  - "best_fitness": highest fitness agents
  - "worst_fitness": lowest fitness agents
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.world import World
from swarm_sim.agents.agent import Agent


class IsolationExperiment:
    """
    Runs a controlled isolation experiment.

    Creates two parallel worlds (control + treatment) from the same seed,
    applies isolation to the treatment group, and collects comparative data.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        isolation_fraction: float = 0.2,
        isolation_duration: int = 100,
        isolation_frequency: int = 5,
        selection_criteria: str = "adventurousness",
        isolation_zone: Tuple[int, int] = (0, 0),
    ):
        self.config = config or SimulationConfig.default()
        self.isolation_fraction = isolation_fraction
        self.isolation_duration = max(1, isolation_duration)
        self.isolation_frequency = max(1, isolation_frequency)
        self.selection_criteria = selection_criteria
        self.isolation_zone = isolation_zone

        # Both worlds use the same seed for fair comparison
        self.control_world: Optional[World] = None
        self.treatment_world: Optional[World] = None

        # Tracking
        self.control_history: List[Dict[str, Any]] = []
        self.treatment_history: List[Dict[str, Any]] = []
        self.isolation_events: List[Dict[str, Any]] = []

        # Per-generation summaries
        self.control_gen_records: List[Dict[str, Any]] = []
        self.treatment_gen_records: List[Dict[str, Any]] = []

        # Currently isolated agents (treatment only): agent_id -> return_step
        self._isolated_agents: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize both worlds from identical configs."""
        self.control_world = World(self.config)
        self.treatment_world = World(self.config)

        self.control_history.clear()
        self.treatment_history.clear()
        self.isolation_events.clear()
        self.control_gen_records.clear()
        self.treatment_gen_records.clear()
        self._isolated_agents.clear()

    # ------------------------------------------------------------------
    # Agent selection for isolation
    # ------------------------------------------------------------------

    def select_agents_for_isolation(
        self, agents: List[Agent], n: int
    ) -> List[Agent]:
        """
        Select agents to isolate based on the configured criteria.

        Returns up to n agents that are alive and not already isolated.
        """
        candidates = [a for a in agents if a.alive and not a.is_isolated]
        if not candidates or n <= 0:
            return []

        n = min(n, len(candidates))

        if self.selection_criteria == "adventurousness":
            candidates.sort(key=lambda a: a.genome["adventurousness"], reverse=True)
            return candidates[:n]

        elif self.selection_criteria == "affiliation_need":
            candidates.sort(key=lambda a: a.genome["affiliation_need"], reverse=True)
            return candidates[:n]

        elif self.selection_criteria == "best_fitness":
            candidates.sort(key=lambda a: a.compute_fitness(), reverse=True)
            return candidates[:n]

        elif self.selection_criteria == "worst_fitness":
            candidates.sort(key=lambda a: a.compute_fitness())
            return candidates[:n]

        elif self.selection_criteria == "random":
            rng = np.random.default_rng(42 + len(self.isolation_events))
            indices = rng.choice(len(candidates), size=n, replace=False)
            return [candidates[i] for i in indices]

        elif self.selection_criteria == "xenophobia":
            candidates.sort(key=lambda a: a.genome["xenophobia"], reverse=True)
            return candidates[:n]

        else:
            # Default to random
            rng = np.random.default_rng(42 + len(self.isolation_events))
            indices = rng.choice(len(candidates), size=n, replace=False)
            return [candidates[i] for i in indices]

    # ------------------------------------------------------------------
    # Isolation mechanics
    # ------------------------------------------------------------------

    def _apply_isolation(self, world: World, current_step: int) -> int:
        """
        Apply isolation to selected agents in the treatment world.

        Returns count of newly isolated agents.
        """
        living = world.get_living_agents()
        n_to_isolate = int(len(living) * self.isolation_fraction)
        if n_to_isolate <= 0:
            return 0
        selected = self.select_agents_for_isolation(living, n_to_isolate)

        count = 0
        rng = np.random.default_rng(current_step + len(self.isolation_events))
        for agent in selected:
            # Scatter to random edge positions (not all at one cell)
            env = world.environment
            iso_x = int(rng.integers(0, env.width))
            iso_y = int(rng.integers(0, env.height))
            agent.isolate(iso_x, iso_y)
            self._isolated_agents[agent.id] = current_step + self.isolation_duration

            self.isolation_events.append({
                "step": current_step,
                "agent_id": agent.id,
                "action": "isolated",
                "criteria": self.selection_criteria,
                "agent_fitness": round(agent.compute_fitness(), 4),
                "agent_energy": round(agent.energy, 2),
                "genome_adventurousness": round(agent.genome["adventurousness"], 4),
                "genome_affiliation": round(agent.genome["affiliation_need"], 4),
            })
            count += 1

        return count

    def _return_agents(self, world: World, current_step: int) -> int:
        """
        Return isolated agents whose duration has expired.

        Returns count of agents returned.
        """
        to_return = [
            aid for aid, return_step in self._isolated_agents.items()
            if current_step >= return_step
        ]

        count = 0
        for agent_id in to_return:
            agent = world.get_agent_by_id(agent_id)
            if agent is not None and agent.alive and agent.is_isolated:
                # Return to a random position near center
                env = world.environment
                rx, ry = env.get_random_empty_position()
                agent.return_from_isolation(rx, ry)

                self.isolation_events.append({
                    "step": current_step,
                    "agent_id": agent_id,
                    "action": "returned",
                    "agent_energy": round(agent.energy, 2),
                })
                count += 1

            del self._isolated_agents[agent_id]

        return count

    # ------------------------------------------------------------------
    # Run experiment
    # ------------------------------------------------------------------

    def run_single_generation(
        self, steps: int, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run one generation for both control and treatment worlds.

        Returns comparison metrics.
        """
        if self.control_world is None or self.treatment_world is None:
            self.setup()

        for step in range(1, steps + 1):
            # Step both worlds
            ctrl_metrics = self.control_world.step()
            treat_metrics = self.treatment_world.step()

            # Treatment: apply isolation on schedule
            isolated_count = 0
            returned_count = 0

            if step % self.isolation_frequency == 0:
                isolated_count = self._apply_isolation(
                    self.treatment_world, step
                )

            # Return agents whose time is up
            returned_count = self._return_agents(self.treatment_world, step)

            # Record
            self.control_history.append(ctrl_metrics)
            treat_metrics["isolated_this_step"] = isolated_count
            treat_metrics["returned_this_step"] = returned_count
            treat_metrics["currently_isolated"] = len(self._isolated_agents)
            self.treatment_history.append(treat_metrics)

            if verbose and step % self.config.logging.log_interval == 0:
                c_alive = ctrl_metrics["agents_alive"]
                t_alive = treat_metrics["agents_alive"]
                t_iso = len(self._isolated_agents)
                print(
                    f"  Step {step:>5d} | "
                    f"Ctrl: {c_alive:>3d} alive, E={ctrl_metrics.get('avg_energy',0):>5.1f} | "
                    f"Treat: {t_alive:>3d} alive, E={treat_metrics.get('avg_energy',0):>5.1f}, "
                    f"Iso: {t_iso:>2d}"
                )

            # Both dead → stop
            if (ctrl_metrics["agents_alive"] == 0 and
                    treat_metrics["agents_alive"] == 0):
                break

        return self._compare_generation()

    def run_experiment(
        self,
        num_generations: int,
        steps_per_generation: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full multi-generational isolation experiment.

        Returns comprehensive comparison results.
        """
        self.setup()

        if verbose:
            print(f"\n{'='*70}")
            print(f"  ISOLATION EXPERIMENT")
            print(f"  Criteria: {self.selection_criteria}")
            print(f"  Isolation: {self.isolation_fraction*100:.0f}% of population "
                  f"every {self.isolation_frequency} steps "
                  f"for {self.isolation_duration} steps")
            print(f"  Generations: {num_generations} x {steps_per_generation} steps")
            print(f"{'='*70}\n")

        for gen in range(num_generations):
            if verbose:
                print(f"--- Generation {gen} ---")

            # Run generation
            comparison = self.run_single_generation(
                steps=steps_per_generation, verbose=verbose,
            )

            # Record generation summaries
            self.control_gen_records.append(comparison["control"])
            self.treatment_gen_records.append(comparison["treatment"])

            if verbose:
                c = comparison["control"]
                t = comparison["treatment"]
                print(
                    f"  >> Ctrl: fit={c['avg_fitness']:.4f}, "
                    f"alive={c['alive_at_end']}, "
                    f"food={c['total_food_eaten']}"
                )
                print(
                    f"  >> Treat: fit={t['avg_fitness']:.4f}, "
                    f"alive={t['alive_at_end']}, "
                    f"food={t['total_food_eaten']}, "
                    f"isolations={t.get('total_isolations', 0)}"
                )
                print()

            # Evolve both worlds
            self.control_world.evolve()
            self.treatment_world.evolve()

            # Clear per-generation isolation tracking
            self._isolated_agents.clear()

        return self.get_results()

    # ------------------------------------------------------------------
    # Comparison & analysis
    # ------------------------------------------------------------------

    def _compare_generation(self) -> Dict[str, Any]:
        """Compare control and treatment for the current generation."""
        ctrl_living = self.control_world.get_living_agents()
        treat_living = self.treatment_world.get_living_agents()
        ctrl_all = self.control_world.agents
        treat_all = self.treatment_world.agents

        ctrl_summary = self._summarize_group(ctrl_all, ctrl_living, "control")
        treat_summary = self._summarize_group(treat_all, treat_living, "treatment")

        # Add isolation-specific metrics to treatment
        gen_events = [
            e for e in self.isolation_events
            if e.get("action") == "isolated"
        ]
        treat_summary["total_isolations"] = len(gen_events)

        return {
            "control": ctrl_summary,
            "treatment": treat_summary,
        }

    def _summarize_group(
        self, all_agents: List[Agent], living: List[Agent], label: str
    ) -> Dict[str, Any]:
        """Summarize a group's performance."""
        fitnesses = [a.compute_fitness() for a in all_agents]
        return {
            "label": label,
            "total_agents": len(all_agents),
            "alive_at_end": len(living),
            "survival_rate": round(len(living) / max(1, len(all_agents)), 4),
            "avg_fitness": round(float(np.mean(fitnesses)) if fitnesses else 0, 4),
            "best_fitness": round(max(fitnesses) if fitnesses else 0, 4),
            "fitness_std": round(float(np.std(fitnesses)) if fitnesses else 0, 4),
            "avg_energy": round(
                float(np.mean([a.energy for a in living])) if living else 0, 2
            ),
            "total_food_eaten": sum(a.total_food_eaten for a in all_agents),
            "total_offspring": sum(a.num_offspring for a in all_agents),
            "genome_diversity": round(self._compute_diversity(all_agents), 4),
        }

    def _compute_diversity(self, agents: List[Agent]) -> float:
        """Compute genome diversity for a group."""
        if len(agents) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                total += agents[i].genome.distance(agents[j].genome)
                count += 1
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, Any]:
        """
        Get comprehensive experiment results.

        Returns a dictionary with:
          - config: experiment parameters
          - control_generations: per-gen records for control
          - treatment_generations: per-gen records for treatment
          - comparison: statistical comparison
          - isolation_events: log of all isolation/return events
        """
        comparison = self._compute_comparison()

        return {
            "config": {
                "isolation_fraction": self.isolation_fraction,
                "isolation_duration": self.isolation_duration,
                "isolation_frequency": self.isolation_frequency,
                "selection_criteria": self.selection_criteria,
                "population_size": self.config.agents.population_size,
            },
            "control_generations": self.control_gen_records,
            "treatment_generations": self.treatment_gen_records,
            "comparison": comparison,
            "total_isolation_events": len([
                e for e in self.isolation_events if e["action"] == "isolated"
            ]),
            "total_return_events": len([
                e for e in self.isolation_events if e["action"] == "returned"
            ]),
        }

    def _compute_comparison(self) -> Dict[str, Any]:
        """Compute statistical comparison between control and treatment."""
        if not self.control_gen_records or not self.treatment_gen_records:
            return {"error": "No data to compare"}

        ctrl_fits = [r["avg_fitness"] for r in self.control_gen_records]
        treat_fits = [r["avg_fitness"] for r in self.treatment_gen_records]
        ctrl_survival = [r["survival_rate"] for r in self.control_gen_records]
        treat_survival = [r["survival_rate"] for r in self.treatment_gen_records]
        ctrl_diversity = [r["genome_diversity"] for r in self.control_gen_records]
        treat_diversity = [r["genome_diversity"] for r in self.treatment_gen_records]
        ctrl_food = [r["total_food_eaten"] for r in self.control_gen_records]
        treat_food = [r["total_food_eaten"] for r in self.treatment_gen_records]

        def safe_mean(lst):
            return round(float(np.mean(lst)), 4) if lst else 0.0

        def safe_last(lst):
            return round(lst[-1], 4) if lst else 0.0

        return {
            "fitness": {
                "control_avg": safe_mean(ctrl_fits),
                "treatment_avg": safe_mean(treat_fits),
                "control_final": safe_last(ctrl_fits),
                "treatment_final": safe_last(treat_fits),
                "difference": round(safe_mean(treat_fits) - safe_mean(ctrl_fits), 4),
                "isolation_impact": (
                    "positive" if safe_mean(treat_fits) > safe_mean(ctrl_fits)
                    else "negative" if safe_mean(treat_fits) < safe_mean(ctrl_fits)
                    else "neutral"
                ),
            },
            "survival": {
                "control_avg": safe_mean(ctrl_survival),
                "treatment_avg": safe_mean(treat_survival),
                "difference": round(
                    safe_mean(treat_survival) - safe_mean(ctrl_survival), 4
                ),
            },
            "diversity": {
                "control_avg": safe_mean(ctrl_diversity),
                "treatment_avg": safe_mean(treat_diversity),
                "control_final": safe_last(ctrl_diversity),
                "treatment_final": safe_last(treat_diversity),
                "difference": round(
                    safe_mean(treat_diversity) - safe_mean(ctrl_diversity), 4
                ),
            },
            "food": {
                "control_total": sum(ctrl_food),
                "treatment_total": sum(treat_food),
                "difference": sum(treat_food) - sum(ctrl_food),
            },
        }

    def print_summary(self) -> None:
        """Print a human-readable experiment summary."""
        results = self.get_results()
        comp = results["comparison"]

        print(f"\n{'='*70}")
        print(f"  ISOLATION EXPERIMENT RESULTS")
        print(f"  Criteria: {results['config']['selection_criteria']}")
        print(f"  Isolation: {results['config']['isolation_fraction']*100:.0f}% "
              f"every {results['config']['isolation_frequency']} steps "
              f"for {results['config']['isolation_duration']} steps")
        print(f"  Total isolations: {results['total_isolation_events']}")
        print(f"  Total returns: {results['total_return_events']}")
        print(f"{'='*70}")

        print(f"\n  FITNESS COMPARISON:")
        f = comp["fitness"]
        print(f"    Control avg:    {f['control_avg']:.4f}")
        print(f"    Treatment avg:  {f['treatment_avg']:.4f}")
        print(f"    Difference:     {f['difference']:+.4f} ({f['isolation_impact']})")

        print(f"\n  SURVIVAL COMPARISON:")
        s = comp["survival"]
        print(f"    Control avg:    {s['control_avg']:.4f}")
        print(f"    Treatment avg:  {s['treatment_avg']:.4f}")
        print(f"    Difference:     {s['difference']:+.4f}")

        print(f"\n  DIVERSITY COMPARISON:")
        d = comp["diversity"]
        print(f"    Control avg:    {d['control_avg']:.4f}")
        print(f"    Treatment avg:  {d['treatment_avg']:.4f}")
        print(f"    Difference:     {d['difference']:+.4f}")

        print(f"\n  FOOD COMPARISON:")
        fd = comp["food"]
        print(f"    Control total:  {fd['control_total']}")
        print(f"    Treatment total:{fd['treatment_total']}")
        print(f"    Difference:     {fd['difference']:+d}")
        print(f"{'='*70}")

    def __repr__(self) -> str:
        return (
            f"IsolationExperiment("
            f"criteria={self.selection_criteria}, "
            f"frac={self.isolation_fraction}, "
            f"dur={self.isolation_duration}, "
            f"freq={self.isolation_frequency}, "
            f"gens={len(self.control_gen_records)})"
        )