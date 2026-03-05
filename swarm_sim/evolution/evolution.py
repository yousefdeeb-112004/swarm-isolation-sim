"""
Genetic Algorithm & Evolution — generational selection and population management.

Manages the evolutionary cycle:
  1. Evaluate fitness of all agents (living + dead from this generation)
  2. Select parents via tournament selection (fitness-proportional)
  3. Create next generation via crossover + mutation
  4. Reset environment and spawn new population
  5. Track evolutionary metrics across generations

The GA operates at the generation level, triggered when:
  - All agents die, OR
  - Max steps per generation reached

Key design: evolution preserves the top-performing genomes (elitism)
while creating diversity through crossover and mutation.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from swarm_sim.agents.genome import Genome
from swarm_sim.agents.agent import Agent


class EvolutionManager:
    """
    Manages the evolutionary cycle across generations.

    Tracks fitness history, performs selection, and creates new generations.
    """

    def __init__(
        self,
        population_size: int = 50,
        elite_count: int = 5,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.15,
        crossover_rate: float = 0.7,
        rng: Optional[np.random.Generator] = None,
    ):
        self.population_size = population_size
        self.elite_count = min(elite_count, population_size)
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.rng = rng or np.random.default_rng()

        # Generation tracking
        self.current_generation: int = 0
        self.generation_history: List[Dict[str, Any]] = []

        # Best genome ever seen
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = 0.0

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate_generation(self, agents: List[Agent]) -> List[Tuple[Agent, float]]:
        """
        Evaluate fitness for all agents in the current generation.

        Returns list of (agent, fitness) sorted by fitness descending.
        """
        scored = []
        for agent in agents:
            fitness = agent.compute_fitness()
            scored.append((agent, fitness))

        # Sort by fitness (best first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Track best ever
        if scored and scored[0][1] > self.best_fitness:
            self.best_fitness = scored[0][1]
            self.best_genome = scored[0][0].genome

        return scored

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def tournament_select(
        self, scored_agents: List[Tuple[Agent, float]]
    ) -> Genome:
        """
        Select a parent genome via tournament selection.

        Picks `tournament_size` random candidates and returns the best.
        """
        if not scored_agents:
            return Genome.random(self.rng)

        indices = self.rng.choice(
            len(scored_agents),
            size=min(self.tournament_size, len(scored_agents)),
            replace=False,
        )
        candidates = [scored_agents[i] for i in indices]
        winner = max(candidates, key=lambda x: x[1])
        return winner[0].genome

    def fitness_proportional_select(
        self, scored_agents: List[Tuple[Agent, float]]
    ) -> Genome:
        """
        Select a parent genome via fitness-proportional (roulette wheel) selection.
        """
        if not scored_agents:
            return Genome.random(self.rng)

        fitnesses = np.array([f for _, f in scored_agents])
        # Shift to positive
        min_f = fitnesses.min()
        if min_f < 0:
            fitnesses = fitnesses - min_f + 0.01
        total = fitnesses.sum()
        if total <= 0:
            # All zero fitness — random
            idx = self.rng.integers(0, len(scored_agents))
        else:
            probs = fitnesses / total
            idx = self.rng.choice(len(scored_agents), p=probs)

        return scored_agents[idx][0].genome

    # ------------------------------------------------------------------
    # Next generation creation
    # ------------------------------------------------------------------

    def create_next_generation(
        self, scored_agents: List[Tuple[Agent, float]]
    ) -> List[Genome]:
        """
        Create genomes for the next generation.

        Strategy:
          1. Elitism: top N genomes pass through unchanged
          2. Crossover: remaining slots filled by tournament-selected
             parents producing children via crossover + mutation
          3. Random immigrants: small fraction of purely random genomes
             to maintain diversity

        Returns list of Genome objects for the new population.
        """
        new_genomes: List[Genome] = []

        if not scored_agents:
            # No survivors — random population
            return [Genome.random(self.rng) for _ in range(self.population_size)]

        # 1. Elitism — preserve top genomes
        for i in range(min(self.elite_count, len(scored_agents))):
            new_genomes.append(scored_agents[i][0].genome)

        # 2. Random immigrants (5% of population)
        n_immigrants = max(1, self.population_size // 20)
        for _ in range(n_immigrants):
            new_genomes.append(Genome.random(self.rng))

        # 3. Fill remaining with crossover offspring
        while len(new_genomes) < self.population_size:
            if self.rng.random() < self.crossover_rate and len(scored_agents) >= 2:
                # Crossover
                parent_a = self.tournament_select(scored_agents)
                parent_b = self.tournament_select(scored_agents)
                child = Genome.from_parents(
                    parent_a, parent_b,
                    mutation_rate=self.mutation_rate,
                    mutation_strength=self.mutation_strength,
                    rng=self.rng,
                )
            else:
                # Mutation only (clone + mutate)
                parent = self.tournament_select(scored_agents)
                child = Genome.from_parents(
                    parent, parent,
                    mutation_rate=self.mutation_rate * 1.5,
                    mutation_strength=self.mutation_strength * 1.5,
                    rng=self.rng,
                )
            new_genomes.append(child)

        return new_genomes[:self.population_size]

    # ------------------------------------------------------------------
    # Generation recording
    # ------------------------------------------------------------------

    def record_generation(
        self,
        generation: int,
        scored_agents: List[Tuple[Agent, float]],
        steps_survived: int,
    ) -> Dict[str, Any]:
        """
        Record statistics for the completed generation.
        """
        fitnesses = [f for _, f in scored_agents]
        genomes = [a.genome for a, _ in scored_agents]

        # Compute genome diversity
        diversity = 0.0
        if len(genomes) >= 2:
            total = 0.0
            count = 0
            for i in range(len(genomes)):
                for j in range(i + 1, len(genomes)):
                    total += genomes[i].distance(genomes[j])
                    count += 1
            diversity = total / count if count > 0 else 0.0

        # Average gene values
        avg_genes = {}
        if genomes:
            for gene_name in genomes[0].genes:
                vals = [g[gene_name] for g in genomes]
                avg_genes[gene_name] = round(float(np.mean(vals)), 4)

        record = {
            "generation": generation,
            "steps_survived": steps_survived,
            "population_size": len(scored_agents),
            "alive_at_end": sum(1 for a, _ in scored_agents if a.alive),
            "best_fitness": round(max(fitnesses) if fitnesses else 0, 4),
            "avg_fitness": round(float(np.mean(fitnesses)) if fitnesses else 0, 4),
            "worst_fitness": round(min(fitnesses) if fitnesses else 0, 4),
            "fitness_std": round(float(np.std(fitnesses)) if fitnesses else 0, 4),
            "genome_diversity": round(diversity, 4),
            "avg_genes": avg_genes,
            "total_food_eaten": sum(a.total_food_eaten for a, _ in scored_agents),
            "total_offspring": sum(a.num_offspring for a, _ in scored_agents),
        }

        self.generation_history.append(record)
        self.current_generation = generation
        return record

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_fitness_trend(self) -> Dict[str, List[float]]:
        """Return fitness trends across generations."""
        return {
            "generations": [r["generation"] for r in self.generation_history],
            "best_fitness": [r["best_fitness"] for r in self.generation_history],
            "avg_fitness": [r["avg_fitness"] for r in self.generation_history],
            "diversity": [r["genome_diversity"] for r in self.generation_history],
        }

    def get_gene_trend(self, gene_name: str) -> List[float]:
        """Return the average value of a gene across generations."""
        return [
            r["avg_genes"].get(gene_name, 0.0)
            for r in self.generation_history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return overall evolution statistics."""
        return {
            "generations_completed": len(self.generation_history),
            "best_fitness_ever": round(self.best_fitness, 4),
            "latest_avg_fitness": (
                round(self.generation_history[-1]["avg_fitness"], 4)
                if self.generation_history else 0.0
            ),
            "latest_diversity": (
                round(self.generation_history[-1]["genome_diversity"], 4)
                if self.generation_history else 0.0
            ),
        }

    def __repr__(self) -> str:
        return (
            f"EvolutionManager(gen={self.current_generation}, "
            f"best={self.best_fitness:.4f}, "
            f"history={len(self.generation_history)})"
        )
