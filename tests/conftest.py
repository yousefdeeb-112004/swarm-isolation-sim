"""Shared test fixtures."""

import numpy as np

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.environment import Environment
from swarm_sim.core.world import World
from swarm_sim.agents.genome import Genome
from swarm_sim.agents.agent import Agent


def make_small_config() -> SimulationConfig:
    """A small-world config for fast tests."""
    cfg = SimulationConfig.default()
    cfg.world.width = 20
    cfg.world.height = 20
    cfg.world.seed = 123
    cfg.environment.food.initial_count = 15
    cfg.environment.food.max_food = 30
    cfg.environment.obstacles.count = 5
    cfg.environment.predators.count = 2
    cfg.agents.population_size = 10
    cfg.agents.sensor_range = 4
    cfg.agents.initial_energy = 100
    return cfg
