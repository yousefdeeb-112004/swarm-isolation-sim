"""
Tests for Section 1: Environment & World Foundation.

Covers:
  - Configuration loading (YAML + defaults)
  - Environment initialization (food, obstacles, predators)
  - Food consumption and regeneration
  - Obstacle collision checks
  - Predator movement (patrol & chase)
  - Sensor observations (local neighborhood)
  - World step loop
"""

import numpy as np
import pytest
from pathlib import Path

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.environment import Environment, CellType, Predator
from swarm_sim.core.world import World


# ===================================================================
# Configuration Tests
# ===================================================================

class TestConfig:

    def test_default_config_creation(self):
        cfg = SimulationConfig.default()
        assert cfg.world.width == 100
        assert cfg.world.height == 100
        assert cfg.environment.food.initial_count == 150
        assert cfg.agents.population_size == 50

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
world:
  width: 50
  height: 50
  seed: 99
environment:
  food:
    initial_count: 30
  obstacles:
    count: 10
  predators:
    count: 2
agents:
  population_size: 20
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        cfg = SimulationConfig.from_yaml(yaml_file)
        assert cfg.world.width == 50
        assert cfg.world.seed == 99
        assert cfg.environment.food.initial_count == 30
        assert cfg.environment.obstacles.count == 10
        assert cfg.agents.population_size == 20

    def test_missing_yaml_raises(self):
        with pytest.raises(FileNotFoundError):
            SimulationConfig.from_yaml("/nonexistent/path.yaml")


# ===================================================================
# Environment Initialization Tests
# ===================================================================

class TestEnvironmentInit:

    def test_grid_shape(self, small_env):
        assert small_env.grid.shape == (20, 20)

    def test_food_count_matches_config(self, small_config, small_env):
        food_on_grid = int(np.sum(small_env.grid == CellType.FOOD))
        assert food_on_grid == small_config.environment.food.initial_count

    def test_obstacle_count_matches_config(self, small_config, small_env):
        obs_on_grid = int(np.sum(small_env.grid == CellType.OBSTACLE))
        assert obs_on_grid == small_config.environment.obstacles.count

    def test_predator_count_matches_config(self, small_config, small_env):
        assert len(small_env.predators) == small_config.environment.predators.count

    def test_no_overlap_food_obstacle(self, small_env):
        """Food and obstacles should never share the same cell."""
        food_mask = small_env.grid == CellType.FOOD
        obs_mask = small_env.grid == CellType.OBSTACLE
        assert not np.any(food_mask & obs_mask)

    def test_repr(self, small_env):
        r = repr(small_env)
        assert "Environment" in r
        assert "20x20" in r

    def test_reproducibility(self, small_config):
        """Same seed should produce identical environments."""
        env1 = Environment(small_config)
        env2 = Environment(small_config)
        assert np.array_equal(env1.grid, env2.grid)


# ===================================================================
# Food Tests
# ===================================================================

class TestFood:

    def test_consume_food_at_food_cell(self, small_env, small_config):
        """Eating food at a food cell should return energy and clear cell."""
        food_pos = np.argwhere(small_env.grid == CellType.FOOD)
        assert len(food_pos) > 0, "No food found on grid"

        fy, fx = food_pos[0]
        energy = small_env.consume_food(int(fx), int(fy))
        assert energy == small_config.environment.food.energy_value
        assert small_env.grid[fy, fx] == CellType.EMPTY

    def test_consume_food_at_empty_cell(self, small_env):
        """Eating at an empty cell should return 0."""
        empty_pos = np.argwhere(small_env.grid == CellType.EMPTY)
        assert len(empty_pos) > 0
        ey, ex = empty_pos[0]
        assert small_env.consume_food(int(ex), int(ey)) == 0

    def test_food_total_decreases_on_consume(self, small_env):
        food_before = small_env.total_food
        food_pos = np.argwhere(small_env.grid == CellType.FOOD)
        fy, fx = food_pos[0]
        small_env.consume_food(int(fx), int(fy))
        assert small_env.total_food == food_before - 1

    def test_food_regeneration(self, small_config):
        """After consuming all food, regeneration should create new food."""
        small_config.environment.food.regeneration_rate = 0.5  # High rate
        env = Environment(small_config)

        # Consume all food
        food_positions = np.argwhere(env.grid == CellType.FOOD)
        for fy, fx in food_positions:
            env.consume_food(int(fx), int(fy))
        assert env.total_food == 0

        # Run a step — should regenerate some food
        env.step()
        assert env.total_food > 0

    def test_food_respects_max_cap(self, small_config):
        """Food should not exceed max_food."""
        small_config.environment.food.regeneration_rate = 1.0  # Extreme rate
        small_config.environment.food.max_food = 20
        env = Environment(small_config)

        for _ in range(50):
            env.step()

        assert env.total_food <= small_config.environment.food.max_food


# ===================================================================
# Obstacle Tests
# ===================================================================

class TestObstacles:

    def test_is_obstacle(self, small_env):
        obs_pos = np.argwhere(small_env.grid == CellType.OBSTACLE)
        assert len(obs_pos) > 0
        oy, ox = obs_pos[0]
        assert small_env.is_obstacle(int(ox), int(oy))

    def test_is_not_obstacle_at_empty(self, small_env):
        empty_pos = np.argwhere(small_env.grid == CellType.EMPTY)
        ey, ex = empty_pos[0]
        assert not small_env.is_obstacle(int(ex), int(ey))

    def test_out_of_bounds_is_obstacle(self, small_env):
        assert small_env.is_obstacle(-1, 0)
        assert small_env.is_obstacle(0, -1)
        assert small_env.is_obstacle(999, 0)

    def test_is_valid_position(self, small_env):
        empty_pos = np.argwhere(small_env.grid == CellType.EMPTY)
        ey, ex = empty_pos[0]
        assert small_env.is_valid_position(int(ex), int(ey))
        assert not small_env.is_valid_position(-1, 0)


# ===================================================================
# Predator Tests
# ===================================================================

class TestPredators:

    def test_predator_patrol_stays_in_bounds(self, small_env):
        """Predators should never leave the grid."""
        for _ in range(100):
            small_env.step()
        for pred in small_env.predators:
            assert 0 <= pred.x < small_env.width
            assert 0 <= pred.y < small_env.height

    def test_predator_chases_agent(self, small_config):
        """A predator near an agent should move toward it."""
        small_config.environment.predators.count = 1
        small_config.environment.predators.detection_range = 50  # See everything
        env = Environment(small_config)

        pred = env.predators[0]
        # Place "agent" far from predator
        agent_x = (pred.x + 5) % small_config.world.width
        agent_y = (pred.y + 5) % small_config.world.height

        old_dist = abs(pred.x - agent_x) + abs(pred.y - agent_y)
        env.step(agent_positions=[(agent_x, agent_y)])
        new_dist = abs(pred.x - agent_x) + abs(pred.y - agent_y)

        assert new_dist <= old_dist  # Should have moved closer or stayed

    def test_predator_collision_damage(self, small_config):
        small_config.environment.predators.count = 1
        env = Environment(small_config)
        pred = env.predators[0]

        damage = env.check_predator_collision(pred.x, pred.y)
        assert damage == small_config.environment.predators.energy_damage

    def test_no_damage_when_far(self, small_env):
        """No damage at a position with no predator."""
        # Find a position far from all predators
        for ey, ex in np.argwhere(small_env.grid == CellType.EMPTY):
            near_pred = any(
                abs(p.x - int(ex)) + abs(p.y - int(ey)) == 0
                for p in small_env.predators
            )
            if not near_pred:
                assert small_env.check_predator_collision(int(ex), int(ey)) == 0
                break


# ===================================================================
# Sensor / Observation Tests
# ===================================================================

class TestObservation:

    def test_observation_returns_expected_keys(self, small_env):
        obs = small_env.get_local_observation(10, 10, sensor_range=4)
        assert "grid_patch" in obs
        assert "food_positions" in obs
        assert "obstacle_positions" in obs
        assert "predator_positions" in obs
        assert "agent_positions" in obs

    def test_observation_patch_shape(self, small_env):
        r = 4
        obs = small_env.get_local_observation(10, 10, sensor_range=r)
        expected = 2 * r + 1
        assert obs["grid_patch"].shape == (expected, expected)

    def test_observation_at_corner(self, small_env):
        """Observations at grid corners should still work (padded)."""
        obs = small_env.get_local_observation(0, 0, sensor_range=4)
        assert obs["grid_patch"].shape == (9, 9)

    def test_observation_includes_nearby_agents(self, small_env):
        obs = small_env.get_local_observation(
            10, 10, sensor_range=4,
            agent_positions=[(10, 10), (11, 11), (100, 100)],
        )
        # (11,11) is within range, (100,100) is not, (10,10) is self
        assert (1, 1) in obs["agent_positions"]
        assert len(obs["agent_positions"]) == 1  # only the nearby one


# ===================================================================
# World Tests
# ===================================================================

class TestWorld:

    def test_world_creation(self, small_world):
        assert small_world.current_step == 0
        assert small_world.environment is not None

    def test_world_step_increments_time(self, small_world):
        small_world.step()
        assert small_world.current_step == 1

    def test_world_run_multiple_steps(self, small_world):
        results = small_world.run(steps=10)
        assert len(results) == 10
        assert small_world.current_step == 10
        assert results[-1]["step"] == 10

    def test_world_reset(self, small_world):
        small_world.run(steps=5)
        small_world.reset()
        assert small_world.current_step == 0
        assert len(small_world.step_history) == 0

    def test_world_state_summary(self, small_world):
        summary = small_world.get_state_summary()
        assert "step" in summary
        assert "generation" in summary
        assert "environment" in summary

    def test_world_repr(self, small_world):
        r = repr(small_world)
        assert "World" in r

    def test_world_deterministic(self, small_config):
        """Two worlds with same config should produce identical results."""
        w1 = World(small_config)
        w2 = World(small_config)

        r1 = w1.run(steps=20)
        r2 = w2.run(steps=20)

        for a, b in zip(r1, r2):
            assert a["total_food"] == b["total_food"]

    def test_distant_position(self, small_world):
        """get_distant_empty_position should return a far-away cell."""
        x, y = small_world.environment.get_distant_empty_position(0, 0, min_distance=10)
        assert abs(x) + abs(y) >= 10
