"""
Environment module — the 2D world where agents live.

Manages:
  - Grid representation (what occupies each cell)
  - Food sources: placement, depletion, regeneration, clustering
  - Obstacles: static impassable cells
  - Predators: mobile dangers that patrol and chase agents
  - Sensor queries: what an agent can see within its range
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional, Dict, Any

from swarm_sim.core.config import SimulationConfig


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

class CellType(IntEnum):
    """What can occupy a cell on the grid."""
    EMPTY = 0
    FOOD = 1
    OBSTACLE = 2
    PREDATOR = 3
    # Agents are tracked separately (multiple can share a cell)


# ---------------------------------------------------------------------------
# Predator entity
# ---------------------------------------------------------------------------

@dataclass
class Predator:
    """A simple predator that patrols near its spawn and chases nearby agents."""

    id: int
    x: int
    y: int
    spawn_x: int
    spawn_y: int
    speed: int = 1
    detection_range: int = 8
    patrol_radius: int = 15
    energy_damage: int = 30

    def patrol_step(self, rng: np.random.Generator, width: int, height: int,
                    obstacle_mask: np.ndarray) -> None:
        """Move randomly within patrol_radius of spawn, avoiding obstacles."""
        dx, dy = rng.integers(-self.speed, self.speed + 1, size=2)
        nx = int(np.clip(self.x + dx, 0, width - 1))
        ny = int(np.clip(self.y + dy, 0, height - 1))

        # Stay within patrol radius and avoid obstacles
        dist_to_spawn = abs(nx - self.spawn_x) + abs(ny - self.spawn_y)
        if dist_to_spawn <= self.patrol_radius and not obstacle_mask[ny, nx]:
            self.x, self.y = nx, ny

    def chase_step(self, target_x: int, target_y: int, width: int, height: int,
                   obstacle_mask: np.ndarray) -> None:
        """Move one step toward a target agent position."""
        dx = int(np.sign(target_x - self.x))
        dy = int(np.sign(target_y - self.y))
        nx = int(np.clip(self.x + dx, 0, width - 1))
        ny = int(np.clip(self.y + dy, 0, height - 1))

        if not obstacle_mask[ny, nx]:
            self.x, self.y = nx, ny


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """
    2D grid world containing food, obstacles, and predators.

    The grid uses a numpy array of shape (height, width) storing CellType values.
    Agents are *not* stored on the grid; instead their positions are queried
    externally. This decouples the environment from agent logic.
    """

    def __init__(self, config: SimulationConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.width = config.world.width
        self.height = config.world.height
        self.rng = rng or np.random.default_rng(config.world.seed)

        # Core grid: CellType per cell
        self.grid: np.ndarray = np.full(
            (self.height, self.width), CellType.EMPTY, dtype=np.int8
        )

        # Convenience masks (updated every step)
        self._obstacle_mask: np.ndarray = np.zeros(
            (self.height, self.width), dtype=bool
        )

        # Predator list
        self.predators: List[Predator] = []

        # Food tracking
        self.total_food: int = 0

        # Stats
        self.food_consumed_this_step: int = 0
        self.food_regenerated_this_step: int = 0

        # Build the world
        self._place_obstacles()
        self._place_food()
        self._place_predators()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _random_empty_cell(self) -> Tuple[int, int]:
        """Return a random (x, y) that is currently EMPTY."""
        while True:
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            if self.grid[y, x] == CellType.EMPTY:
                return x, y

    def _place_obstacles(self) -> None:
        """Place static obstacles randomly on the grid."""
        count = self.config.environment.obstacles.count
        for _ in range(count):
            x, y = self._random_empty_cell()
            self.grid[y, x] = CellType.OBSTACLE
            self._obstacle_mask[y, x] = True

    def _place_food(self) -> None:
        """Place initial food sources with optional clustering."""
        cfg = self.config.environment.food
        placed = 0

        while placed < cfg.initial_count:
            # Decide: cluster near existing food or place randomly
            if placed > 0 and self.rng.random() < cfg.cluster_probability:
                # Pick a random existing food cell and place nearby
                food_positions = np.argwhere(self.grid == CellType.FOOD)
                if len(food_positions) > 0:
                    idx = self.rng.integers(0, len(food_positions))
                    fy, fx = food_positions[idx]
                    offset_x = int(self.rng.integers(-cfg.cluster_radius, cfg.cluster_radius + 1))
                    offset_y = int(self.rng.integers(-cfg.cluster_radius, cfg.cluster_radius + 1))
                    nx = int(np.clip(fx + offset_x, 0, self.width - 1))
                    ny = int(np.clip(fy + offset_y, 0, self.height - 1))
                    if self.grid[ny, nx] == CellType.EMPTY:
                        self.grid[ny, nx] = CellType.FOOD
                        placed += 1
                    continue

            x, y = self._random_empty_cell()
            self.grid[y, x] = CellType.FOOD
            placed += 1

        self.total_food = placed

    def _place_predators(self) -> None:
        """Spawn predators at random empty locations."""
        cfg = self.config.environment.predators
        for i in range(cfg.count):
            x, y = self._random_empty_cell()
            pred = Predator(
                id=i, x=x, y=y, spawn_x=x, spawn_y=y,
                speed=cfg.speed,
                detection_range=cfg.detection_range,
                patrol_radius=cfg.patrol_radius,
                energy_damage=cfg.energy_damage,
            )
            self.predators.append(pred)
            # Note: predator positions are tracked in the Predator objects,
            # not permanently on the grid (they move).

    # ------------------------------------------------------------------
    # Per-step updates
    # ------------------------------------------------------------------

    def step(self, agent_positions: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Advance the environment by one time step.

        Parameters
        ----------
        agent_positions : list of (x, y), optional
            Current positions of all living agents (used by predators for chasing).

        Returns
        -------
        dict with step metrics:
            - food_regenerated: int
            - predator_positions: list of (x, y)
        """
        self.food_consumed_this_step = 0
        self.food_regenerated_this_step = 0

        self._regenerate_food()
        self._update_predators(agent_positions or [])

        return {
            "food_regenerated": self.food_regenerated_this_step,
            "total_food": self.total_food,
            "predator_positions": [(p.x, p.y) for p in self.predators],
        }

    def _regenerate_food(self) -> None:
        """Probabilistically spawn new food on empty cells."""
        cfg = self.config.environment.food

        if self.total_food >= cfg.max_food:
            return

        # For performance: sample a subset of empty cells rather than iterating all
        empty_cells = np.argwhere(self.grid == CellType.EMPTY)
        if len(empty_cells) == 0:
            return

        # Each empty cell has regeneration_rate chance of spawning food
        spawn_mask = self.rng.random(len(empty_cells)) < cfg.regeneration_rate
        candidates = empty_cells[spawn_mask]

        budget = cfg.max_food - self.total_food
        if len(candidates) > budget:
            indices = self.rng.choice(len(candidates), size=budget, replace=False)
            candidates = candidates[indices]

        for (cy, cx) in candidates:
            self.grid[cy, cx] = CellType.FOOD
            self.total_food += 1
            self.food_regenerated_this_step += 1

    def _update_predators(self, agent_positions: List[Tuple[int, int]]) -> None:
        """Move each predator: chase nearest agent if in range, else patrol."""
        agent_arr = np.array(agent_positions) if agent_positions else np.empty((0, 2))

        for pred in self.predators:
            target = self._nearest_agent_in_range(pred, agent_arr)
            if target is not None:
                pred.chase_step(
                    target[0], target[1],
                    self.width, self.height,
                    self._obstacle_mask,
                )
            else:
                pred.patrol_step(self.rng, self.width, self.height, self._obstacle_mask)

    def _nearest_agent_in_range(
        self, pred: Predator, agent_arr: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Find the closest agent within predator's detection range."""
        if len(agent_arr) == 0:
            return None

        dists = np.abs(agent_arr[:, 0] - pred.x) + np.abs(agent_arr[:, 1] - pred.y)
        in_range = dists <= pred.detection_range

        if not np.any(in_range):
            return None

        closest_idx = int(np.argmin(np.where(in_range, dists, np.inf)))
        return int(agent_arr[closest_idx, 0]), int(agent_arr[closest_idx, 1])

    # ------------------------------------------------------------------
    # Agent interaction queries
    # ------------------------------------------------------------------

    def consume_food(self, x: int, y: int) -> int:
        """
        Agent at (x, y) attempts to eat food.

        Returns energy gained (0 if no food at that cell).
        """
        if self.grid[y, x] == CellType.FOOD:
            self.grid[y, x] = CellType.EMPTY
            self.total_food -= 1
            self.food_consumed_this_step += 1
            return self.config.environment.food.energy_value
        return 0

    def is_obstacle(self, x: int, y: int) -> bool:
        """Check if a cell is blocked by an obstacle."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True  # Out of bounds treated as obstacle
        return self._obstacle_mask[y, x]

    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if an agent can stand on this cell."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return not self._obstacle_mask[y, x]

    def get_local_observation(
        self,
        x: int,
        y: int,
        sensor_range: int,
        agent_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Return everything visible within sensor_range of position (x, y).

        Returns
        -------
        dict:
            grid_patch : np.ndarray of shape (2*range+1, 2*range+1)
                Cell types in the local area (padded with OBSTACLE for edges).
            food_positions : list of (rx, ry) — relative coords of food cells
            obstacle_positions : list of (rx, ry)
            predator_positions : list of (rx, ry) — predators in range
            agent_positions : list of (rx, ry) — other agents in range
        """
        r = sensor_range
        patch_size = 2 * r + 1

        # Build padded patch (out-of-bounds = OBSTACLE)
        patch = np.full((patch_size, patch_size), CellType.OBSTACLE, dtype=np.int8)

        # Compute bounds
        x_min = x - r
        y_min = y - r
        x_max = x + r + 1
        y_max = y + r + 1

        # Source bounds (grid)
        sx_min = max(x_min, 0)
        sy_min = max(y_min, 0)
        sx_max = min(x_max, self.width)
        sy_max = min(y_max, self.height)

        # Destination bounds (patch)
        dx_min = sx_min - x_min
        dy_min = sy_min - y_min
        dx_max = dx_min + (sx_max - sx_min)
        dy_max = dy_min + (sy_max - sy_min)

        patch[dy_min:dy_max, dx_min:dx_max] = self.grid[sy_min:sy_max, sx_min:sx_max]

        # Extract relative positions of interesting items
        food_positions = []
        obstacle_positions = []
        for ry in range(patch_size):
            for rx in range(patch_size):
                if patch[ry, rx] == CellType.FOOD:
                    food_positions.append((rx - r, ry - r))
                elif patch[ry, rx] == CellType.OBSTACLE:
                    # Only report obstacles within the actual grid, not padding
                    if (0 <= x_min + rx < self.width and 0 <= y_min + ry < self.height):
                        obstacle_positions.append((rx - r, ry - r))

        # Predators in range
        predator_rel = []
        for pred in self.predators:
            if abs(pred.x - x) <= r and abs(pred.y - y) <= r:
                predator_rel.append((pred.x - x, pred.y - y))

        # Other agents in range
        agent_rel = []
        if agent_positions:
            for (ax, ay) in agent_positions:
                if (ax, ay) != (x, y) and abs(ax - x) <= r and abs(ay - y) <= r:
                    agent_rel.append((ax - x, ay - y))

        return {
            "grid_patch": patch,
            "food_positions": food_positions,
            "obstacle_positions": obstacle_positions,
            "predator_positions": predator_rel,
            "agent_positions": agent_rel,
        }

    def check_predator_collision(self, x: int, y: int) -> int:
        """
        Check if any predator occupies the same cell as (x, y).

        Returns total energy damage (could be hit by multiple predators).
        """
        damage = 0
        for pred in self.predators:
            if pred.x == x and pred.y == y:
                damage += pred.energy_damage
        return damage

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_random_empty_position(self) -> Tuple[int, int]:
        """Public wrapper for finding a random empty cell."""
        return self._random_empty_cell()

    def get_distant_empty_position(self, from_x: int, from_y: int,
                                    min_distance: int) -> Tuple[int, int]:
        """Find a random empty cell at least min_distance away from (from_x, from_y)."""
        for _ in range(1000):
            x, y = self._random_empty_cell()
            dist = abs(x - from_x) + abs(y - from_y)
            if dist >= min_distance:
                return x, y
        # Fallback: return whatever we can get
        return self._random_empty_cell()

    def get_food_count(self) -> int:
        return self.total_food

    def get_grid_snapshot(self) -> np.ndarray:
        """Return a copy of the grid for visualization/logging."""
        return self.grid.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Return current environment statistics."""
        return {
            "total_food": self.total_food,
            "food_consumed_this_step": self.food_consumed_this_step,
            "food_regenerated_this_step": self.food_regenerated_this_step,
            "num_obstacles": int(np.sum(self._obstacle_mask)),
            "num_predators": len(self.predators),
            "predator_positions": [(p.x, p.y) for p in self.predators],
        }

    def __repr__(self) -> str:
        return (
            f"Environment(size={self.width}x{self.height}, "
            f"food={self.total_food}, "
            f"obstacles={int(np.sum(self._obstacle_mask))}, "
            f"predators={len(self.predators)})"
        )
