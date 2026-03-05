"""
Multi-Agent Interaction — communication, reproduction, and social dynamics.

Handles all agent-to-agent interactions:
  1. Communication: nearby agents share belief maps (social learning)
  2. Reproduction: agents with enough energy produce offspring
  3. Competition: agents on the same food cell compete
  4. Social bonding: tracking familiarity between agents

Interactions are managed by the InteractionManager, which is called
by the World each step after all agents have acted.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set

from swarm_sim.agents.agent import Agent, Action
from swarm_sim.agents.genome import Genome


class InteractionManager:
    """
    Manages all agent-to-agent interactions each simulation step.

    Called by World after the observe-decide-act loop.
    """

    def __init__(
        self,
        communication_range: int = 3,
        reproduction_range: int = 2,
        reproduction_cost: float = 40.0,
        child_energy: float = 50.0,
        min_reproduction_age: int = 50,
        communication_cooldown: int = 5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.communication_range = communication_range
        self.reproduction_range = reproduction_range
        self.reproduction_cost = reproduction_cost
        self.child_energy = child_energy
        self.min_reproduction_age = min_reproduction_age
        self.communication_cooldown = communication_cooldown
        self.rng = rng or np.random.default_rng()

        # Social memory: tracks who has interacted with whom
        # Key: (agent_id_a, agent_id_b) -> interaction count
        self.interaction_history: Dict[Tuple[int, int], int] = {}

        # Cooldown tracking: agent_id -> step when last communicated
        self._comm_cooldowns: Dict[int, int] = {}

        # Step-level stats
        self.total_communications: int = 0
        self.total_reproductions: int = 0
        self.total_competitions: int = 0

    # ------------------------------------------------------------------
    # Main entry: process all interactions for one step
    # ------------------------------------------------------------------

    def process_interactions(
        self,
        agents: List[Agent],
        environment: Any,
        current_step: int,
        world_width: int,
        world_height: int,
        sensor_range: int,
    ) -> Dict[str, Any]:
        """
        Process all interactions for the current step.

        Returns metrics dict with counts of each interaction type.
        """
        living = [a for a in agents if a.alive]
        if len(living) < 2:
            return {"communications": 0, "reproductions": 0, "competitions": 0, "new_agents": []}

        # Build spatial index for fast neighbor lookup
        positions = {a.id: (a.x, a.y) for a in living}

        step_comms = 0
        step_repros = 0
        step_comps = 0
        new_agents: List[Agent] = []

        # Find all nearby pairs (within max interaction range)
        max_range = max(self.communication_range, self.reproduction_range)
        pairs = self._find_nearby_pairs(living, max_range)

        # 1. Communication
        for a, b in pairs:
            if self._can_communicate(a, b, current_step):
                self._communicate(a, b, current_step)
                step_comms += 1

        # 2. Reproduction
        for a, b in pairs:
            dist = abs(a.x - b.x) + abs(a.y - b.y)
            if dist <= self.reproduction_range:
                child = self._try_reproduce(
                    a, b, environment, world_width, world_height, sensor_range
                )
                if child is not None:
                    new_agents.append(child)
                    step_repros += 1

        # Update totals
        self.total_communications += step_comms
        self.total_reproductions += step_repros
        self.total_competitions += step_comps

        return {
            "communications": step_comms,
            "reproductions": step_repros,
            "competitions": step_comps,
            "new_agents": new_agents,
        }

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def _find_nearby_pairs(
        self, agents: List[Agent], max_range: int
    ) -> List[Tuple[Agent, Agent]]:
        """Find all pairs of agents within max_range Manhattan distance."""
        pairs = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                dist = abs(a.x - b.x) + abs(a.y - b.y)
                if dist <= max_range:
                    pairs.append((a, b))
        return pairs

    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------

    def _can_communicate(self, a: Agent, b: Agent, current_step: int) -> bool:
        """Check if two agents can communicate this step."""
        dist = abs(a.x - b.x) + abs(a.y - b.y)
        if dist > self.communication_range:
            return False

        # Cooldown check
        last_a = self._comm_cooldowns.get(a.id, -999)
        last_b = self._comm_cooldowns.get(b.id, -999)
        if (current_step - last_a < self.communication_cooldown or
                current_step - last_b < self.communication_cooldown):
            return False

        # Xenophobia check: highly xenophobic agents refuse communication
        # with genetically distant strangers
        if a.belief_network is not None and b.belief_network is not None:
            genetic_dist = a.genome.distance(b.genome)
            avg_xeno = (a.genome["xenophobia"] + b.genome["xenophobia"]) / 2
            if self.rng.random() < avg_xeno * genetic_dist:
                return False

        return True

    def _communicate(self, a: Agent, b: Agent, current_step: int) -> None:
        """
        Two agents share their belief maps (social learning).

        Each agent's beliefs are blended with the other's, weighted
        by their relative confidence and affiliation.
        """
        if a.belief_network is None or b.belief_network is None:
            return

        # Communication weight based on affiliation
        weight_a = a.genome["affiliation_need"] * 0.3
        weight_b = b.genome["affiliation_need"] * 0.3

        # Merge beliefs bidirectionally
        a.belief_network.merge_beliefs(b.belief_network, weight=weight_a)
        b.belief_network.merge_beliefs(a.belief_network, weight=weight_b)

        # Update cooldowns
        self._comm_cooldowns[a.id] = current_step
        self._comm_cooldowns[b.id] = current_step

        # Track interaction
        pair_key = (min(a.id, b.id), max(a.id, b.id))
        self.interaction_history[pair_key] = self.interaction_history.get(pair_key, 0) + 1

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def _try_reproduce(
        self,
        a: Agent,
        b: Agent,
        environment: Any,
        world_width: int,
        world_height: int,
        sensor_range: int,
    ) -> Optional[Agent]:
        """
        Attempt reproduction between two agents.

        Conditions:
          - Both have enough energy (above their genome's threshold)
          - Both are old enough
          - Neither is isolated
          - Random chance based on energy surplus
        """
        # Check conditions
        if a.is_isolated or b.is_isolated:
            return None
        if a.age < self.min_reproduction_age or b.age < self.min_reproduction_age:
            return None
        if a.energy < a.genome["reproduction_threshold"]:
            return None
        if b.energy < b.genome["reproduction_threshold"]:
            return None

        # Probability based on energy surplus
        surplus_a = (a.energy - a.genome["reproduction_threshold"]) / a.max_energy
        surplus_b = (b.energy - b.genome["reproduction_threshold"]) / b.max_energy
        prob = min(0.1, (surplus_a + surplus_b) / 2)

        if self.rng.random() > prob:
            return None

        # Find spawn position near parents
        spawn_x, spawn_y = self._find_spawn_position(a, b, environment)
        if spawn_x is None:
            return None

        # Create child
        child = Agent.from_parents(
            a, b,
            x=spawn_x, y=spawn_y,
            energy=self.child_energy,
            rng=np.random.default_rng(self.rng.integers(0, 2**32)),
        )

        # Initialize child's cognitive systems
        child.init_belief_network(world_width, world_height, sensor_range)

        # Deduct energy from parents
        a.energy -= self.reproduction_cost
        b.energy -= self.reproduction_cost

        return child

    def _find_spawn_position(
        self, a: Agent, b: Agent, environment: Any
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find a valid empty cell near both parents."""
        mid_x = (a.x + b.x) // 2
        mid_y = (a.y + b.y) // 2

        # Try positions in expanding radius
        for radius in range(0, 4):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = mid_x + dx, mid_y + dy
                    if environment.is_valid_position(nx, ny):
                        return nx, ny

        return None, None

    # ------------------------------------------------------------------
    # Familiarity queries
    # ------------------------------------------------------------------

    def get_familiarity(self, agent_a_id: int, agent_b_id: int) -> int:
        """Get the number of times two agents have interacted."""
        key = (min(agent_a_id, agent_b_id), max(agent_a_id, agent_b_id))
        return self.interaction_history.get(key, 0)

    def get_social_network(self, agents: List[Agent]) -> Dict[int, List[int]]:
        """
        Build a social network graph: agent_id -> list of connected agent_ids.

        Two agents are connected if they have interacted at least once.
        """
        network: Dict[int, List[int]] = {a.id: [] for a in agents if a.alive}
        for (id_a, id_b), count in self.interaction_history.items():
            if count > 0:
                if id_a in network:
                    network[id_a].append(id_b)
                if id_b in network:
                    network[id_b].append(id_a)
        return network

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return interaction statistics."""
        return {
            "total_communications": self.total_communications,
            "total_reproductions": self.total_reproductions,
            "total_competitions": self.total_competitions,
            "unique_interactions": len(self.interaction_history),
            "avg_interaction_count": (
                np.mean(list(self.interaction_history.values()))
                if self.interaction_history else 0.0
            ),
        }

    def __repr__(self) -> str:
        return (
            f"InteractionManager(comms={self.total_communications}, "
            f"repros={self.total_reproductions}, "
            f"pairs={len(self.interaction_history)})"
        )
