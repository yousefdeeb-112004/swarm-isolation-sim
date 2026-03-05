"""
Bayesian Belief Network — probabilistic beliefs about the environment.

Each agent maintains a belief map over its local neighborhood, estimating
the probability of food, danger, and other agents in nearby regions.
Beliefs are updated using Bayesian inference when new observations arrive.

Architecture:
  - The world is divided into a coarse grid of "regions" (e.g., 5x5 cells each)
  - Each region has belief probabilities for: food, danger, agents
  - Observations within sensor range update beliefs via Bayes' rule
  - Unobserved regions decay toward a prior over time
  - The belief vector output feeds into the Neural Network (Section 4)

This is a lightweight discrete Bayesian network — no pgmpy dependency needed.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class BeliefNetwork:
    """
    Probabilistic belief model for a single agent.

    The agent divides the world into coarse regions and maintains
    probability estimates for what exists in each region.

    Attributes
    ----------
    world_width, world_height : int
        Full world dimensions.
    region_size : int
        How many cells per region side (e.g., 5 means 5x5 cell regions).
    grid_w, grid_h : int
        Number of regions in each dimension.
    food_belief : np.ndarray (grid_h, grid_w)
        P(food exists) for each region, in [0, 1].
    danger_belief : np.ndarray (grid_h, grid_w)
        P(danger exists) — predators or obstacles.
    agent_belief : np.ndarray (grid_h, grid_w)
        P(other agents present) for each region.
    confidence : np.ndarray (grid_h, grid_w)
        How confident the agent is about each region [0, 1].
        Decays over time for unobserved regions.
    """

    def __init__(
        self,
        world_width: int = 100,
        world_height: int = 100,
        region_size: int = 5,
        prior_food: float = 0.15,
        prior_danger: float = 0.05,
        prior_agents: float = 0.1,
        decay_rate: float = 0.02,
        observation_strength: float = 0.8,
    ):
        self.world_width = world_width
        self.world_height = world_height
        self.region_size = region_size

        # Coarse grid dimensions
        self.grid_w = max(1, (world_width + region_size - 1) // region_size)
        self.grid_h = max(1, (world_height + region_size - 1) // region_size)

        # Priors
        self.prior_food = prior_food
        self.prior_danger = prior_danger
        self.prior_agents = prior_agents

        # Learning parameters
        self.decay_rate = decay_rate
        self.observation_strength = observation_strength

        # Belief grids — initialized to priors
        self.food_belief = np.full((self.grid_h, self.grid_w), prior_food, dtype=np.float64)
        self.danger_belief = np.full((self.grid_h, self.grid_w), prior_danger, dtype=np.float64)
        self.agent_belief = np.full((self.grid_h, self.grid_w), prior_agents, dtype=np.float64)

        # Confidence: how recently each region was observed
        self.confidence = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)

        # Step counter for time-based decay
        self.last_observed = np.full(
            (self.grid_h, self.grid_w), -1, dtype=np.int32
        )
        self.current_step = 0

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _cell_to_region(self, x: int, y: int) -> Tuple[int, int]:
        """Convert cell coordinates to region coordinates."""
        rx = min(x // self.region_size, self.grid_w - 1)
        ry = min(y // self.region_size, self.grid_h - 1)
        return int(rx), int(ry)

    def _region_center_cell(self, rx: int, ry: int) -> Tuple[int, int]:
        """Get the center cell of a region (for distance calculations)."""
        cx = rx * self.region_size + self.region_size // 2
        cy = ry * self.region_size + self.region_size // 2
        return min(cx, self.world_width - 1), min(cy, self.world_height - 1)

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------

    def update(
        self,
        agent_x: int,
        agent_y: int,
        observation: Dict[str, Any],
        sensor_range: int,
    ) -> None:
        """
        Update beliefs based on a new observation.

        Parameters
        ----------
        agent_x, agent_y : int
            Agent's current cell position.
        observation : dict
            From Environment.get_local_observation():
            - food_positions: list of (rx, ry) relative positions
            - predator_positions: list of (rx, ry) relative
            - obstacle_positions: list of (rx, ry) relative
            - agent_positions: list of (rx, ry) relative
        sensor_range : int
            Agent's sensor range in cells.
        """
        self.current_step += 1

        # Decay all beliefs toward priors (farther = faster decay)
        self._decay_beliefs()

        # Determine which regions the agent can observe
        observed_regions = self._get_observed_regions(agent_x, agent_y, sensor_range)

        # Count observations per observed region
        food_obs = self._count_in_regions(
            agent_x, agent_y, observation.get("food_positions", [])
        )
        pred_obs = self._count_in_regions(
            agent_x, agent_y, observation.get("predator_positions", [])
        )
        obs_obs = self._count_in_regions(
            agent_x, agent_y, observation.get("obstacle_positions", [])
        )
        agent_obs = self._count_in_regions(
            agent_x, agent_y, observation.get("agent_positions", [])
        )

        # Merge predator + obstacle into danger
        danger_obs = {}
        for key in set(list(pred_obs.keys()) + list(obs_obs.keys())):
            danger_obs[key] = pred_obs.get(key, 0) + obs_obs.get(key, 0)

        # Apply Bayesian updates to observed regions
        cells_per_region = self.region_size * self.region_size
        strength = self.observation_strength

        for (rx, ry) in observed_regions:
            # Food belief
            n_food = food_obs.get((rx, ry), 0)
            likelihood_food = min(1.0, n_food / max(cells_per_region * 0.1, 1))
            self.food_belief[ry, rx] = self._bayesian_update(
                self.food_belief[ry, rx], likelihood_food, strength
            )

            # Danger belief
            n_danger = danger_obs.get((rx, ry), 0)
            likelihood_danger = min(1.0, n_danger / max(cells_per_region * 0.05, 1))
            self.danger_belief[ry, rx] = self._bayesian_update(
                self.danger_belief[ry, rx], likelihood_danger, strength
            )

            # Agent belief
            n_agents = agent_obs.get((rx, ry), 0)
            likelihood_agents = min(1.0, n_agents / max(cells_per_region * 0.05, 1))
            self.agent_belief[ry, rx] = self._bayesian_update(
                self.agent_belief[ry, rx], likelihood_agents, strength
            )

            # Update confidence and last-observed
            self.confidence[ry, rx] = min(1.0, self.confidence[ry, rx] + 0.3)
            self.last_observed[ry, rx] = self.current_step

    @staticmethod
    def _bayesian_update(prior: float, likelihood: float, strength: float) -> float:
        """
        Simple Bayesian update: blend prior toward observed evidence.

        Uses a weighted average approach for stability:
            posterior = prior * (1 - strength) + likelihood * strength

        This avoids numerical issues with pure Bayes rule on sparse observations.
        """
        posterior = prior * (1.0 - strength) + likelihood * strength
        return float(np.clip(posterior, 0.001, 0.999))

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def _decay_beliefs(self) -> None:
        """Decay beliefs toward priors for regions not recently observed."""
        # Regions observed long ago or never have their beliefs pulled toward prior
        time_since = np.where(
            self.last_observed >= 0,
            self.current_step - self.last_observed,
            self.current_step + 1
        ).astype(np.float64)

        # Decay factor: older observations decay faster
        decay_factor = np.clip(self.decay_rate * time_since, 0.0, 1.0)

        self.food_belief += (self.prior_food - self.food_belief) * decay_factor
        self.danger_belief += (self.prior_danger - self.danger_belief) * decay_factor
        self.agent_belief += (self.prior_agents - self.agent_belief) * decay_factor

        # Decay confidence
        self.confidence *= (1.0 - self.decay_rate)

    # ------------------------------------------------------------------
    # Region helpers
    # ------------------------------------------------------------------

    def _get_observed_regions(
        self, agent_x: int, agent_y: int, sensor_range: int
    ) -> List[Tuple[int, int]]:
        """Get all regions that fall within the agent's sensor range."""
        # The agent can see cells from (x-r, y-r) to (x+r, y+r)
        x_min = max(0, agent_x - sensor_range)
        y_min = max(0, agent_y - sensor_range)
        x_max = min(self.world_width - 1, agent_x + sensor_range)
        y_max = min(self.world_height - 1, agent_y + sensor_range)

        # Convert to region range
        rx_min, ry_min = self._cell_to_region(x_min, y_min)
        rx_max, ry_max = self._cell_to_region(x_max, y_max)

        regions = []
        for ry in range(ry_min, ry_max + 1):
            for rx in range(rx_min, rx_max + 1):
                regions.append((rx, ry))
        return regions

    def _count_in_regions(
        self,
        agent_x: int,
        agent_y: int,
        relative_positions: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], int]:
        """
        Count how many items fall into each region.

        relative_positions are (dx, dy) relative to agent position.
        """
        counts: Dict[Tuple[int, int], int] = {}
        for dx, dy in relative_positions:
            abs_x = agent_x + dx
            abs_y = agent_y + dy
            if 0 <= abs_x < self.world_width and 0 <= abs_y < self.world_height:
                rx, ry = self._cell_to_region(abs_x, abs_y)
                counts[(rx, ry)] = counts.get((rx, ry), 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Belief queries
    # ------------------------------------------------------------------

    def get_belief_at(self, x: int, y: int) -> Dict[str, float]:
        """Get beliefs for the region containing cell (x, y)."""
        rx, ry = self._cell_to_region(x, y)
        return {
            "food": float(self.food_belief[ry, rx]),
            "danger": float(self.danger_belief[ry, rx]),
            "agents": float(self.agent_belief[ry, rx]),
            "confidence": float(self.confidence[ry, rx]),
        }

    def get_best_food_direction(self, agent_x: int, agent_y: int) -> Tuple[int, int]:
        """
        Return the (dx, dy) direction toward the region with highest food belief.

        Returns (0, 0) if the best region is the current one.
        """
        current_rx, current_ry = self._cell_to_region(agent_x, agent_y)

        # Weight food belief by confidence, penalize danger
        score = self.food_belief * (0.5 + 0.5 * self.confidence) - 0.3 * self.danger_belief

        best_ry, best_rx = np.unravel_index(np.argmax(score), score.shape)

        if best_rx == current_rx and best_ry == current_ry:
            return (0, 0)

        # Direction toward best region center
        target_x, target_y = self._region_center_cell(int(best_rx), int(best_ry))
        dx = int(np.sign(target_x - agent_x))
        dy = int(np.sign(target_y - agent_y))
        return (dx, dy)

    def get_safest_direction(self, agent_x: int, agent_y: int) -> Tuple[int, int]:
        """
        Return (dx, dy) direction toward the region with lowest danger belief.
        """
        current_rx, current_ry = self._cell_to_region(agent_x, agent_y)

        # Invert danger: lower is better. Add small bonus for high confidence.
        safety = (1.0 - self.danger_belief) * (0.5 + 0.5 * self.confidence)

        best_ry, best_rx = np.unravel_index(np.argmax(safety), safety.shape)

        if best_rx == current_rx and best_ry == current_ry:
            return (0, 0)

        target_x, target_y = self._region_center_cell(int(best_rx), int(best_ry))
        dx = int(np.sign(target_x - agent_x))
        dy = int(np.sign(target_y - agent_y))
        return (dx, dy)

    def get_social_direction(self, agent_x: int, agent_y: int) -> Tuple[int, int]:
        """
        Return (dx, dy) direction toward the region with highest agent belief.
        """
        current_rx, current_ry = self._cell_to_region(agent_x, agent_y)
        score = self.agent_belief * (0.5 + 0.5 * self.confidence)

        best_ry, best_rx = np.unravel_index(np.argmax(score), score.shape)

        if best_rx == current_rx and best_ry == current_ry:
            return (0, 0)

        target_x, target_y = self._region_center_cell(int(best_rx), int(best_ry))
        dx = int(np.sign(target_x - agent_x))
        dy = int(np.sign(target_y - agent_y))
        return (dx, dy)

    # ------------------------------------------------------------------
    # Belief vector for NN input
    # ------------------------------------------------------------------

    def get_belief_vector(self, agent_x: int, agent_y: int, radius: int = 2) -> np.ndarray:
        """
        Produce a compact belief vector for neural network input.

        Samples beliefs from a (2*radius+1) x (2*radius+1) neighborhood
        of regions around the agent's current region.

        Returns
        -------
        np.ndarray of shape (n_features,) where n_features = (2r+1)^2 * 4
            4 channels: food, danger, agents, confidence per region
        """
        cx, cy = self._cell_to_region(agent_x, agent_y)
        patch_size = 2 * radius + 1
        features = np.zeros((patch_size, patch_size, 4), dtype=np.float32)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ry = cy + dy
                rx = cx + dx
                py = dy + radius
                px = dx + radius

                if 0 <= rx < self.grid_w and 0 <= ry < self.grid_h:
                    features[py, px, 0] = self.food_belief[ry, rx]
                    features[py, px, 1] = self.danger_belief[ry, rx]
                    features[py, px, 2] = self.agent_belief[ry, rx]
                    features[py, px, 3] = self.confidence[ry, rx]
                else:
                    # Out of world bounds: high danger, no food
                    features[py, px, 0] = 0.0
                    features[py, px, 1] = 1.0
                    features[py, px, 2] = 0.0
                    features[py, px, 3] = 0.0

        return features.flatten()

    def get_global_summary(self) -> Dict[str, float]:
        """Return global statistics about beliefs."""
        return {
            "avg_food_belief": float(np.mean(self.food_belief)),
            "max_food_belief": float(np.max(self.food_belief)),
            "avg_danger_belief": float(np.mean(self.danger_belief)),
            "max_danger_belief": float(np.max(self.danger_belief)),
            "avg_agent_belief": float(np.mean(self.agent_belief)),
            "avg_confidence": float(np.mean(self.confidence)),
        }

    # ------------------------------------------------------------------
    # Social learning: absorb beliefs from another agent
    # ------------------------------------------------------------------

    def merge_beliefs(self, other: "BeliefNetwork", weight: float = 0.3) -> None:
        """
        Incorporate another agent's beliefs (e.g., via communication).

        Parameters
        ----------
        other : BeliefNetwork
            The other agent's belief network.
        weight : float
            How much to weight the other's beliefs (0 = ignore, 1 = fully adopt).
        """
        # Only merge regions where the other has higher confidence
        other_better = other.confidence > self.confidence

        blend = weight * other_better.astype(np.float64)

        self.food_belief += (other.food_belief - self.food_belief) * blend
        self.danger_belief += (other.danger_belief - self.danger_belief) * blend
        self.agent_belief += (other.agent_belief - self.agent_belief) * blend
        self.confidence = np.maximum(self.confidence, other.confidence * weight)

    # ------------------------------------------------------------------
    # Reset / clone
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all beliefs to priors."""
        self.food_belief[:] = self.prior_food
        self.danger_belief[:] = self.prior_danger
        self.agent_belief[:] = self.prior_agents
        self.confidence[:] = 0.0
        self.last_observed[:] = -1
        self.current_step = 0

    def clone(self) -> "BeliefNetwork":
        """Create a deep copy of this belief network."""
        new = BeliefNetwork(
            world_width=self.world_width,
            world_height=self.world_height,
            region_size=self.region_size,
            prior_food=self.prior_food,
            prior_danger=self.prior_danger,
            prior_agents=self.prior_agents,
            decay_rate=self.decay_rate,
            observation_strength=self.observation_strength,
        )
        new.food_belief = self.food_belief.copy()
        new.danger_belief = self.danger_belief.copy()
        new.agent_belief = self.agent_belief.copy()
        new.confidence = self.confidence.copy()
        new.last_observed = self.last_observed.copy()
        new.current_step = self.current_step
        return new

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BeliefNetwork(regions={self.grid_w}x{self.grid_h}, "
            f"avg_food={np.mean(self.food_belief):.3f}, "
            f"avg_danger={np.mean(self.danger_belief):.3f}, "
            f"avg_confidence={np.mean(self.confidence):.3f})"
        )
