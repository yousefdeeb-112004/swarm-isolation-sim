"""
World manager — orchestrates the simulation loop.

Responsibilities:
  - Owns the Environment and the agent population
  - Tracks global time step
  - Runs the observe → decide → act loop for all agents each step
  - Manages agent spawning, death, and population stats
  - Manages the random number generator for reproducibility
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.environment import Environment
from swarm_sim.agents.agent import Agent, Action
from swarm_sim.agents.genome import Genome
from swarm_sim.agents.interaction import InteractionManager
from swarm_sim.evolution.evolution import EvolutionManager


class World:
    """
    Top-level simulation world.

    Manages the Environment and the agent population. Each step runs
    the full agent lifecycle: observe -> decide -> act -> check death.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig.default()
        self.rng = np.random.default_rng(self.config.world.seed)

        # Core components
        self.environment = Environment(self.config, rng=self.rng)

        # Agent population
        self.agents: List[Agent] = []
        self.dead_agents: List[Agent] = []

        # Time tracking
        self.current_step: int = 0
        self.current_generation: int = 0
        self.max_steps: int = self.config.world.max_steps

        # Step history (lightweight metrics per step)
        self.step_history: List[Dict[str, Any]] = []

        # Multi-agent interaction manager (Section 6)
        self.interaction_manager = InteractionManager(
            communication_range=3,
            reproduction_range=2,
            reproduction_cost=40.0,
            child_energy=50.0,
            min_reproduction_age=50,
            communication_cooldown=5,
            rng=np.random.default_rng(self.rng.integers(0, 2**32)),
        )

        # Evolution manager (Section 7)
        self.evolution_manager = EvolutionManager(
            population_size=self.config.agents.population_size,
            elite_count=self.config.evolution.elitism_count,
            tournament_size=self.config.evolution.tournament_size,
            mutation_rate=self.config.evolution.mutation_rate,
            mutation_strength=self.config.evolution.mutation_strength,
            crossover_rate=self.config.evolution.crossover_rate,
            rng=np.random.default_rng(self.rng.integers(0, 2**32)),
        )

        # Spawn initial population
        self._spawn_initial_population()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _spawn_initial_population(self) -> None:
        """Create the initial agent population with random genomes."""
        Agent.reset_id_counter()
        n = self.config.agents.population_size
        initial_energy = self.config.agents.initial_energy

        for _ in range(n):
            x, y = self.environment.get_random_empty_position()
            genome = Genome.random(self.rng)
            agent = Agent(
                genome=genome,
                x=x, y=y,
                energy=initial_energy,
                rng=np.random.default_rng(self.rng.integers(0, 2**32)),
            )
            agent.init_belief_network(
                self.config.world.width,
                self.config.world.height,
                self.config.agents.sensor_range,
            )
            self.agents.append(agent)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """
        Advance the world by one time step.

        Pipeline per step:
          1. Collect all agent positions
          2. Advance environment (food regen, predator movement)
          3. For each living agent: observe -> decide -> act
          4. Remove dead agents
          5. Collect metrics
        """
        self.current_step += 1

        # 1. Gather living agent positions for environment
        living = [a for a in self.agents if a.alive]
        agent_positions = [(a.x, a.y) for a in living]

        # 2. Advance environment
        env_metrics = self.environment.step(agent_positions)

        # 3. Agent lifecycle: observe -> decide -> act
        step_food_eaten = 0
        step_agents_died = 0
        step_actions: Dict[str, int] = {}

        for agent in living:
            # Observe
            obs = self.environment.get_local_observation(
                agent.x, agent.y,
                sensor_range=self.config.agents.sensor_range,
                agent_positions=agent_positions,
            )
            agent.observe(obs)

            # Decide
            action = agent.decide()
            action_name = action.name
            step_actions[action_name] = step_actions.get(action_name, 0) + 1

            # Act
            result = agent.act(action, self.environment)
            if result["ate"]:
                step_food_eaten += 1

        # 4. Handle deaths
        newly_dead = [a for a in self.agents if not a.alive and a not in self.dead_agents]
        step_agents_died = len(newly_dead)
        self.dead_agents.extend(newly_dead)

        # 5. Multi-agent interactions (communication, reproduction)
        interaction_metrics = self.interaction_manager.process_interactions(
            agents=self.agents,
            environment=self.environment,
            current_step=self.current_step,
            world_width=self.config.world.width,
            world_height=self.config.world.height,
            sensor_range=self.config.agents.sensor_range,
        )

        # Add newly born agents to the population
        new_agents = interaction_metrics.get("new_agents", [])
        for child in new_agents:
            self.agents.append(child)

        # 6. Build metrics
        living_after = [a for a in self.agents if a.alive]
        metrics = {
            "step": self.current_step,
            "generation": self.current_generation,
            "agents_alive": len(living_after),
            "agents_died_this_step": step_agents_died,
            "agents_born_this_step": len(new_agents),
            "total_dead": len(self.dead_agents),
            "total_agents": len(self.agents),
            "food_eaten_this_step": step_food_eaten,
            "communications": interaction_metrics.get("communications", 0),
            "reproductions": interaction_metrics.get("reproductions", 0),
            "actions": step_actions,
            **env_metrics,
            **self.environment.get_stats(),
        }

        # Population stats
        if living_after:
            energies = [a.energy for a in living_after]
            ages = [a.age for a in living_after]
            metrics.update({
                "avg_energy": round(np.mean(energies), 2),
                "min_energy": round(min(energies), 2),
                "max_energy": round(max(energies), 2),
                "avg_age": round(np.mean(ages), 2),
                "avg_distance_between_agents": round(
                    self._avg_agent_distance(living_after), 2
                ),
            })
        else:
            metrics.update({
                "avg_energy": 0, "min_energy": 0, "max_energy": 0,
                "avg_age": 0, "avg_distance_between_agents": 0,
            })

        self.step_history.append(metrics)
        return metrics

    def run(self, steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run the simulation for a given number of steps.

        Stops early if all agents die.
        """
        steps = steps or self.max_steps
        results = []
        for _ in range(steps):
            metrics = self.step()
            results.append(metrics)
            if metrics["agents_alive"] == 0:
                break
        return results

    def reset(self) -> None:
        """Reset the world to initial state."""
        self.rng = np.random.default_rng(self.config.world.seed)
        self.environment = Environment(self.config, rng=self.rng)
        self.agents.clear()
        self.dead_agents.clear()
        self.current_step = 0
        self.current_generation = 0
        self.step_history.clear()
        self.interaction_manager = InteractionManager(
            communication_range=3,
            reproduction_range=2,
            reproduction_cost=40.0,
            child_energy=50.0,
            min_reproduction_age=50,
            communication_cooldown=5,
            rng=np.random.default_rng(self.rng.integers(0, 2**32)),
        )
        self._spawn_initial_population()

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(self) -> Dict[str, Any]:
        """
        Evaluate current generation and create the next one.

        Steps:
          1. Score all agents (living + dead)
          2. Record generation stats
          3. Select parents and create new genomes
          4. Reset environment
          5. Spawn new population with evolved genomes
        """
        # 1. Score all agents
        all_agents = self.agents  # includes living and dead
        scored = self.evolution_manager.evaluate_generation(all_agents)

        # 2. Record generation
        gen_record = self.evolution_manager.record_generation(
            generation=self.current_generation,
            scored_agents=scored,
            steps_survived=self.current_step,
        )

        # 3. Create next generation genomes
        new_genomes = self.evolution_manager.create_next_generation(scored)

        # 4. Reset environment
        self.environment = Environment(
            self.config,
            rng=np.random.default_rng(self.rng.integers(0, 2**32)),
        )

        # 5. Spawn new population
        self.agents.clear()
        self.dead_agents.clear()
        self.current_step = 0
        self.current_generation += 1

        # Reset interaction manager
        self.interaction_manager = InteractionManager(
            communication_range=3,
            reproduction_range=2,
            reproduction_cost=40.0,
            child_energy=50.0,
            min_reproduction_age=50,
            communication_cooldown=5,
            rng=np.random.default_rng(self.rng.integers(0, 2**32)),
        )

        Agent.reset_id_counter()
        initial_energy = self.config.agents.initial_energy
        for genome in new_genomes:
            x, y = self.environment.get_random_empty_position()
            agent = Agent(
                genome=genome,
                x=x, y=y,
                energy=initial_energy,
                rng=np.random.default_rng(self.rng.integers(0, 2**32)),
            )
            agent.init_belief_network(
                self.config.world.width,
                self.config.world.height,
                self.config.agents.sensor_range,
            )
            agent.generation = self.current_generation
            self.agents.append(agent)

        return gen_record

    def run_evolution(
        self,
        num_generations: int,
        steps_per_generation: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple generations of evolution.

        Each generation runs for `steps_per_generation` steps (or until
        all agents die), then evolves to the next generation.

        Returns list of generation records.
        """
        steps_per_gen = steps_per_generation or self.max_steps
        generation_records = []

        for gen in range(num_generations):
            # Run the current generation
            for _ in range(steps_per_gen):
                metrics = self.step()
                if metrics["agents_alive"] == 0:
                    break

            # Evolve
            record = self.evolve()
            generation_records.append(record)

            if verbose:
                print(
                    f"  Gen {record['generation']:>3d} | "
                    f"Steps: {record['steps_survived']:>4d} | "
                    f"Alive: {record['alive_at_end']:>3d} | "
                    f"BestFit: {record['best_fitness']:.4f} | "
                    f"AvgFit: {record['avg_fitness']:.4f} | "
                    f"Diversity: {record['genome_diversity']:.4f} | "
                    f"Food: {record['total_food_eaten']:>3d}"
                )

        return generation_records

    # ------------------------------------------------------------------
    # Agent queries
    # ------------------------------------------------------------------

    def get_living_agents(self) -> List[Agent]:
        """Return all currently living agents."""
        return [a for a in self.agents if a.alive]

    def get_agent_by_id(self, agent_id: int) -> Optional[Agent]:
        """Find an agent by its ID."""
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    # ------------------------------------------------------------------
    # Population analysis
    # ------------------------------------------------------------------

    def _avg_agent_distance(self, agents: List[Agent]) -> float:
        """Compute average pairwise Manhattan distance between agents."""
        if len(agents) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                total += abs(agents[i].x - agents[j].x) + abs(agents[i].y - agents[j].y)
                count += 1
        return total / count if count > 0 else 0.0

    def get_genome_diversity(self) -> float:
        """
        Compute average genetic distance across all living agents.
        Returns value in [0, 1]. Higher = more diverse.
        """
        living = self.get_living_agents()
        if len(living) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(living)):
            for j in range(i + 1, len(living)):
                total += living[i].genome.distance(living[j].genome)
                count += 1
        return total / count if count > 0 else 0.0

    def get_population_stats(self) -> Dict[str, Any]:
        """Return detailed population statistics."""
        living = self.get_living_agents()
        if not living:
            return {"alive": 0, "dead": len(self.dead_agents)}

        return {
            "alive": len(living),
            "dead": len(self.dead_agents),
            "avg_energy": round(np.mean([a.energy for a in living]), 2),
            "avg_age": round(np.mean([a.age for a in living]), 2),
            "avg_fitness": round(np.mean([a.compute_fitness() for a in living]), 4),
            "genome_diversity": round(self.get_genome_diversity(), 4),
            "avg_adventurousness": round(
                np.mean([a.genome["adventurousness"] for a in living]), 4
            ),
            "avg_affiliation_need": round(
                np.mean([a.genome["affiliation_need"] for a in living]), 4
            ),
            "total_food_eaten": sum(a.total_food_eaten for a in living),
            "total_offspring": sum(a.num_offspring for a in living),
            **self.interaction_manager.get_stats(),
        }

    # ------------------------------------------------------------------
    # State summary
    # ------------------------------------------------------------------

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a summary of the current world state."""
        return {
            "step": self.current_step,
            "generation": self.current_generation,
            "environment": self.environment.get_stats(),
            "population": self.get_population_stats(),
        }

    def __repr__(self) -> str:
        alive = len(self.get_living_agents())
        return (
            f"World(step={self.current_step}, "
            f"gen={self.current_generation}, "
            f"agents={alive}/{len(self.agents)}, "
            f"env={self.environment})"
        )
