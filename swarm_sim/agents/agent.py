"""
Agent module — the core entity that lives, moves, eats, and dies.

Each agent has:
  - A Genome (fixed at birth)
  - Position, energy, age
  - An isolation flag and history
  - A memory buffer for past experiences
  - Stubs for Bayesian network, neural network, and RL policy (future sections)

In Section 2, agents use a simple genome-driven heuristic for movement.
Sections 3–5 will replace this with Bayesian beliefs, NN inner states, and RL.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from enum import IntEnum

from swarm_sim.agents.genome import Genome
from swarm_sim.agents.bayesian import BeliefNetwork
from swarm_sim.agents.neural import (
    InnerStateNetwork, build_vitals_vector, compute_heuristic_inner_state,
    INNER_STATE_NAMES, NUM_INNER_STATES,
)
from swarm_sim.agents.policy import RLPolicy


# ---------------------------------------------------------------------------
# Actions an agent can take
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Available agent actions."""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    STAY = 4
    EAT = 5
    COMMUNICATE = 6     # placeholder for Section 6
    REPRODUCE = 7       # placeholder for Section 6


ACTION_DELTAS = {
    Action.MOVE_UP:    (0, -1),
    Action.MOVE_DOWN:  (0,  1),
    Action.MOVE_LEFT:  (-1, 0),
    Action.MOVE_RIGHT: (1,  0),
    Action.STAY:       (0,  0),
}

NUM_MOVEMENT_ACTIONS = 5  # UP, DOWN, LEFT, RIGHT, STAY


# ---------------------------------------------------------------------------
# Memory entry
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory of a past experience."""
    step: int
    position: Tuple[int, int]
    action: int
    energy_change: float
    nearby_agents: int
    nearby_food: int
    nearby_predators: int


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class Agent:
    """
    A single agent in the swarm.

    Lifecycle:
        1. Born with a Genome (random or inherited)
        2. Each step: observe → decide → act → update state
        3. Dies when energy <= 0 or age >= lifespan
    """

    _next_id: int = 0  # Class-level counter for unique IDs

    def __init__(
        self,
        genome: Optional[Genome] = None,
        x: int = 0,
        y: int = 0,
        energy: float = 100.0,
        agent_id: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        # Identity
        if agent_id is not None:
            self.id = agent_id
        else:
            self.id = Agent._next_id
            Agent._next_id += 1

        # Genome
        self.genome = genome or Genome.random(rng)

        # Position
        self.x = x
        self.y = y

        # Vitals
        self.energy = energy
        self.max_energy: float = 200.0
        self.age: int = 0
        self.alive: bool = True
        self.cause_of_death: Optional[str] = None

        # Isolation state
        self.is_isolated: bool = False
        self.isolation_steps: int = 0
        self.times_isolated: int = 0
        self.steps_since_return: int = -1   # -1 means never isolated

        # Social tracking
        self.num_offspring: int = 0
        self.parent_ids: Tuple[Optional[int], Optional[int]] = (None, None)
        self.generation: int = 0

        # Stats tracking
        self.total_food_eaten: int = 0
        self.total_steps_alive: int = 0
        self.total_energy_gained: float = 0.0
        self.total_energy_lost: float = 0.0

        # Memory buffer (genome controls capacity)
        mem_capacity = self.genome["memory_size"]
        self.memory: deque = deque(maxlen=mem_capacity)

        # Last observation (cached for decision-making)
        self._last_observation: Optional[Dict[str, Any]] = None

        # Bayesian belief network (Section 3)
        self.belief_network: Optional[BeliefNetwork] = None
        self._world_width: int = 100
        self._world_height: int = 100
        self._sensor_range: int = 7

        # Neural network inner state model (Section 4)
        self.inner_net: Optional[InnerStateNetwork] = None

        # RL policy (Section 5)
        self.policy: Optional[RLPolicy] = None

        # Inner state vector (hunger, fear, curiosity) — Section 4 will use NN
        # For now, computed from simple heuristics
        self.inner_state: Dict[str, float] = {
            "hunger": 0.0,      # Higher when low energy
            "fear": 0.0,        # Higher when predators nearby
            "curiosity": 0.0,   # Higher with adventurousness gene
            "loneliness": 0.0,  # Higher when isolated
            "aggression": 0.0,  # Higher with xenophobia + competition
        }

        # RNG
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the global agent ID counter (useful for tests)."""
        cls._next_id = 0

    @classmethod
    def from_parents(
        cls,
        parent_a: "Agent",
        parent_b: "Agent",
        x: int,
        y: int,
        energy: float,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> "Agent":
        """Create a child agent via sexual reproduction."""
        rng = rng or np.random.default_rng()
        child_genome = Genome.from_parents(
            parent_a.genome, parent_b.genome,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            rng=rng,
        )
        child = cls(genome=child_genome, x=x, y=y, energy=energy, rng=rng)
        child.parent_ids = (parent_a.id, parent_b.id)
        child.generation = max(parent_a.generation, parent_b.generation) + 1

        parent_a.num_offspring += 1
        parent_b.num_offspring += 1

        return child

    # ------------------------------------------------------------------
    # Belief network setup
    # ------------------------------------------------------------------

    def init_belief_network(
        self, world_width: int, world_height: int, sensor_range: int
    ) -> None:
        """Initialize the Bayesian belief network and neural network for this agent."""
        self._world_width = world_width
        self._world_height = world_height
        self._sensor_range = sensor_range
        self.belief_network = BeliefNetwork(
            world_width=world_width,
            world_height=world_height,
            region_size=5,
            prior_food=0.15,
            prior_danger=0.05,
            prior_agents=0.1,
            decay_rate=0.02,
            observation_strength=0.5 + 0.3 * self.genome["plasticity"] / 0.1,
        )

        # Initialize neural network for inner state
        belief_vec_size = self.belief_network.get_belief_vector(0, 0, radius=2).shape[0]
        self.inner_net = InnerStateNetwork(
            belief_size=belief_vec_size,
            hidden_size=self.genome["nn_hidden_size"],
            learning_rate=self.genome["plasticity"],
            rng=self.rng,
        )

        # Initialize RL policy
        self.policy = RLPolicy(
            epsilon=self.genome["exploration_rate"],
            gamma=0.95,
            hidden_size=self.genome["nn_hidden_size"] // 2,
            learning_rate=min(self.genome["plasticity"], 0.02),
            affiliation_need=self.genome["affiliation_need"],
            adventurousness=self.genome["adventurousness"],
            rng=self.rng,
        )

    # ------------------------------------------------------------------
    # Per-step lifecycle
    # ------------------------------------------------------------------

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Receive local observation from the environment.

        Parameters
        ----------
        observation : dict
            From Environment.get_local_observation(), containing:
            grid_patch, food_positions, obstacle_positions,
            predator_positions, agent_positions
        """
        self._last_observation = observation
        self._update_inner_state(observation)

        # Update Bayesian beliefs
        if self.belief_network is not None:
            self.belief_network.update(
                self.x, self.y, observation, self._sensor_range
            )

        # Compute inner state via neural network
        if self.inner_net is not None and self.belief_network is not None:
            self._update_inner_state_nn(observation)

    def decide(self) -> Action:
        """
        Choose an action using the RL policy (with heuristic overrides for safety).

        The RL policy selects actions via epsilon-greedy Q-learning.
        Critical survival behaviors override RL to prevent early deaths
        while the policy is still learning.
        """
        if self._last_observation is None:
            return Action.STAY

        obs = self._last_observation

        # --- Safety override: Flee from predators ---
        if obs["predator_positions"] and self.inner_state.get("fear", 0) > 0.3:
            return self._flee_from_predators(obs["predator_positions"])

        # --- Safety override: Eat when hungry and on food ---
        if self._is_on_food(obs) and self.inner_state.get("hunger", 0) > 0.3:
            return Action.EAT

        # --- Safety override: Move toward visible food when hungry ---
        if obs["food_positions"] and self.inner_state.get("hunger", 0) > 0.5:
            return self._move_toward_nearest(obs["food_positions"])

        # --- RL Policy selection (blended with heuristic early on) ---
        if self.policy is not None:
            # Heuristic blend: use heuristic more during early life
            rl_maturity = min(1.0, self.age / 100.0)
            if self.rng.random() < rl_maturity:
                action_idx = self.policy.select_action(self.inner_state, obs)
                return Action(min(action_idx, len(Action) - 1))
            else:
                return self._heuristic_decide(obs)

        # --- Fallback: belief-guided heuristic ---
        return self._heuristic_decide(obs)

    def _heuristic_decide(self, obs: Dict[str, Any]) -> Action:
        """Fallback heuristic decision-making (used if no RL policy)."""
        # Eat if on food and hungry
        if self._is_on_food(obs) and self.inner_state["hunger"] > 0.3:
            return Action.EAT

        # Move toward observed food if hungry
        if obs["food_positions"] and self.inner_state["hunger"] > 0.2:
            return self._move_toward_nearest(obs["food_positions"])

        # Belief-guided food search
        if self.belief_network is not None and self.inner_state["hunger"] > 0.3:
            dx, dy = self.belief_network.get_best_food_direction(self.x, self.y)
            if dx != 0 or dy != 0:
                return self._direction_to_action(dx, dy)

        # Explore
        if self.rng.random() < self.genome["adventurousness"]:
            return self._random_move()

        # Social seeking
        if obs["agent_positions"] and self.rng.random() < self.genome["affiliation_need"]:
            return self._move_toward_nearest(obs["agent_positions"])

        if self.rng.random() < 0.5:
            return self._random_move()
        return Action.STAY

    def act(self, action: Action, env: Any) -> Dict[str, Any]:
        """
        Execute the chosen action in the environment.

        Parameters
        ----------
        action : Action
            The action to perform.
        env : Environment
            The environment to interact with.

        Returns
        -------
        dict : result of the action
            - energy_change: float
            - moved: bool
            - ate: bool
        """
        result = {"energy_change": 0.0, "moved": False, "ate": False}

        if not self.alive:
            return result

        if action == Action.EAT:
            gained = env.consume_food(self.x, self.y)
            if gained > 0:
                self.energy = min(self.energy + gained, self.max_energy)
                result["energy_change"] += gained
                result["ate"] = True
                self.total_food_eaten += 1
                self.total_energy_gained += gained

        elif action in ACTION_DELTAS:
            dx, dy = ACTION_DELTAS[action]
            nx, ny = self.x + dx, self.y + dy
            if env.is_valid_position(nx, ny):
                self.x, self.y = nx, ny
                result["moved"] = True

        # Metabolism cost (reduced during isolation — agents are inactive)
        metabolism = env.config.agents.energy_per_step
        if self.is_isolated:
            metabolism *= 0.5  # Half metabolism cost while isolated
        self.energy += metabolism
        result["energy_change"] += metabolism
        self.total_energy_lost += abs(metabolism)

        # Predator collision (skip if isolated — agent is in separate zone)
        if not self.is_isolated:
            damage = env.check_predator_collision(self.x, self.y)
            if damage > 0:
                self.energy -= damage
                result["energy_change"] -= damage
                self.total_energy_lost += damage

        # Age
        self.age += 1
        self.total_steps_alive += 1

        # Isolation tracking
        if self.is_isolated:
            self.isolation_steps += 1
        if self.steps_since_return >= 0:
            self.steps_since_return += 1

        # Store memory
        obs = self._last_observation or {}
        self.memory.append(MemoryEntry(
            step=self.age,
            position=(self.x, self.y),
            action=int(action),
            energy_change=result["energy_change"],
            nearby_agents=len(obs.get("agent_positions", [])),
            nearby_food=len(obs.get("food_positions", [])),
            nearby_predators=len(obs.get("predator_positions", [])),
        ))

        # Check death
        self._check_death()

        # RL learning step
        if self.policy is not None and self._last_observation is not None:
            obs = self._last_observation
            reward = self.policy.compute_reward(
                energy_change=result["energy_change"],
                ate=result["ate"],
                nearby_agents=len(obs.get("agent_positions", [])),
                nearby_predators=len(obs.get("predator_positions", [])),
                agent_x=self.x,
                agent_y=self.y,
                is_isolated=self.is_isolated,
                alive=self.alive,
            )
            self.policy.learn(
                reward=reward,
                next_inner_state=self.inner_state,
                next_observation=obs,
                done=not self.alive,
            )
            # Gradual epsilon decay
            self.policy.decay_epsilon(decay_rate=0.9995, min_epsilon=0.05)

        return result

    # ------------------------------------------------------------------
    # Inner state (heuristic for Section 2, NN in Section 4)
    # ------------------------------------------------------------------

    def _update_inner_state(self, obs: Dict[str, Any]) -> None:
        """Compute inner state from observations and vitals (heuristic fallback)."""
        # Hunger: inversely proportional to energy
        self.inner_state["hunger"] = max(0.0, 1.0 - (self.energy / self.max_energy))

        # Fear: based on nearby predators
        n_preds = len(obs.get("predator_positions", []))
        self.inner_state["fear"] = min(1.0, n_preds * 0.5)

        # Curiosity: driven by genome + low fear
        self.inner_state["curiosity"] = (
            self.genome["adventurousness"] * (1.0 - self.inner_state["fear"])
        )

        # Loneliness: based on isolation and lack of nearby agents
        n_agents = len(obs.get("agent_positions", []))
        base_loneliness = max(0.0, 1.0 - n_agents * 0.2)
        isolation_boost = 0.3 if self.is_isolated else 0.0
        self.inner_state["loneliness"] = min(
            1.0,
            base_loneliness * self.genome["affiliation_need"] + isolation_boost
        )

        # Aggression: driven by xenophobia and competition
        self.inner_state["aggression"] = (
            self.genome["xenophobia"]
            * min(1.0, n_agents * 0.2)
            * (0.5 + 0.5 * self.inner_state["hunger"])
        )

    def _update_inner_state_nn(self, obs: Dict[str, Any]) -> None:
        """Compute inner state via neural network and train it online."""
        # Build inputs
        belief_vec = self.belief_network.get_belief_vector(self.x, self.y, radius=2)
        vitals = build_vitals_vector(
            energy=self.energy,
            max_energy=self.max_energy,
            age=self.age,
            lifespan=self.genome["lifespan"],
            is_isolated=self.is_isolated,
            adventurousness=self.genome["adventurousness"],
            affiliation_need=self.genome["affiliation_need"],
            xenophobia=self.genome["xenophobia"],
            plasticity=self.genome["plasticity"],
            exploration_rate=self.genome["exploration_rate"],
        )

        # Forward pass
        nn_output = self.inner_net.forward(belief_vec, vitals)

        # Compute heuristic target for supervised learning
        target = compute_heuristic_inner_state(
            energy=self.energy,
            max_energy=self.max_energy,
            nearby_predators=len(obs.get("predator_positions", [])),
            nearby_food=len(obs.get("food_positions", [])),
            nearby_agents=len(obs.get("agent_positions", [])),
            is_isolated=self.is_isolated,
            adventurousness=self.genome["adventurousness"],
            affiliation_need=self.genome["affiliation_need"],
            xenophobia=self.genome["xenophobia"],
        )

        # Train toward heuristic (the NN learns to generalize)
        self.inner_net.learn(target)

        # Blend NN output with heuristic (NN gains more influence over time)
        nn_weight = min(0.8, self.inner_net.total_updates / 200.0)
        blended = (1.0 - nn_weight) * target + nn_weight * nn_output

        # Update inner state dict
        for i, name in enumerate(INNER_STATE_NAMES):
            self.inner_state[name] = float(np.clip(blended[i], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Decision helpers (heuristic)
    # ------------------------------------------------------------------

    def _flee_from_predators(self, predator_positions: List[Tuple[int, int]]) -> Action:
        """Move away from the nearest predator."""
        if not predator_positions:
            return Action.STAY

        # Average predator direction
        avg_dx = sum(p[0] for p in predator_positions) / len(predator_positions)
        avg_dy = sum(p[1] for p in predator_positions) / len(predator_positions)

        # Move in opposite direction
        if abs(avg_dx) >= abs(avg_dy):
            return Action.MOVE_LEFT if avg_dx > 0 else Action.MOVE_RIGHT
        else:
            return Action.MOVE_UP if avg_dy > 0 else Action.MOVE_DOWN

    def _move_toward_nearest(self, positions: List[Tuple[int, int]]) -> Action:
        """Move toward the nearest position in the list."""
        if not positions:
            return Action.STAY

        # Find closest by Manhattan distance
        closest = min(positions, key=lambda p: abs(p[0]) + abs(p[1]))
        dx, dy = closest

        if dx == 0 and dy == 0:
            return Action.STAY

        if abs(dx) >= abs(dy):
            return Action.MOVE_RIGHT if dx > 0 else Action.MOVE_LEFT
        else:
            return Action.MOVE_DOWN if dy > 0 else Action.MOVE_UP

    def _is_on_food(self, obs: Dict[str, Any]) -> bool:
        """Check if agent is standing on food (relative (0,0) in observation)."""
        return (0, 0) in obs.get("food_positions", [])

    def _random_move(self) -> Action:
        """Pick a random movement action."""
        return Action(self.rng.integers(0, NUM_MOVEMENT_ACTIONS))

    def _direction_to_action(self, dx: int, dy: int) -> Action:
        """Convert a (dx, dy) direction into a movement Action."""
        if dx == 0 and dy == 0:
            return Action.STAY
        if abs(dx) >= abs(dy):
            return Action.MOVE_RIGHT if dx > 0 else Action.MOVE_LEFT
        else:
            return Action.MOVE_DOWN if dy > 0 else Action.MOVE_UP

    # ------------------------------------------------------------------
    # Death
    # ------------------------------------------------------------------

    def _check_death(self) -> None:
        """Check if the agent should die."""
        if self.energy <= 0:
            self.alive = False
            self.cause_of_death = "starvation"
        elif self.age >= self.genome["lifespan"]:
            self.alive = False
            self.cause_of_death = "old_age"

    def kill(self, cause: str = "external") -> None:
        """Force-kill the agent."""
        self.alive = False
        self.cause_of_death = cause

    # ------------------------------------------------------------------
    # Isolation
    # ------------------------------------------------------------------

    def isolate(self, new_x: int, new_y: int) -> None:
        """Teleport agent to isolation zone."""
        self.x = new_x
        self.y = new_y
        self.is_isolated = True
        self.isolation_steps = 0
        self.times_isolated += 1

    def return_from_isolation(self, new_x: int, new_y: int) -> None:
        """Return agent to the swarm."""
        self.x = new_x
        self.y = new_y
        self.is_isolated = False
        self.steps_since_return = 0

    # ------------------------------------------------------------------
    # Fitness (for evolution — detailed in Section 7)
    # ------------------------------------------------------------------

    def compute_fitness(self) -> float:
        """
        Compute a fitness score for this agent.

        Components:
          - Survival time (40%)
          - Food eaten (30%)
          - Offspring produced (20%)
          - Energy remaining (10%)
        """
        survival_score = self.total_steps_alive / max(self.genome["lifespan"], 1)
        food_score = min(1.0, self.total_food_eaten / 20.0)
        offspring_score = min(1.0, self.num_offspring / 3.0)
        energy_score = self.energy / self.max_energy if self.energy > 0 else 0.0

        return (
            0.4 * survival_score
            + 0.3 * food_score
            + 0.2 * offspring_score
            + 0.1 * energy_score
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return full agent state as a dictionary for logging."""
        state = {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "energy": round(self.energy, 2),
            "age": self.age,
            "alive": self.alive,
            "cause_of_death": self.cause_of_death,
            "generation": self.generation,
            "is_isolated": self.is_isolated,
            "isolation_steps": self.isolation_steps,
            "times_isolated": self.times_isolated,
            "num_offspring": self.num_offspring,
            "total_food_eaten": self.total_food_eaten,
            "inner_state": dict(self.inner_state),
            "genome": self.genome.to_dict(),
            "fitness": round(self.compute_fitness(), 4),
        }
        if self.belief_network is not None:
            state["beliefs"] = self.belief_network.get_global_summary()
        if self.inner_net is not None:
            state["nn_stats"] = self.inner_net.get_weight_stats()
        if self.policy is not None:
            state["rl_stats"] = self.policy.get_stats()
        return state

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "alive" if self.alive else f"dead({self.cause_of_death})"
        return (
            f"Agent(id={self.id}, pos=({self.x},{self.y}), "
            f"energy={self.energy:.1f}, age={self.age}, {status})"
        )