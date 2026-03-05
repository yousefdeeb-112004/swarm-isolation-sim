"""Core simulation components: environment, world, configuration."""

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.environment import Environment, CellType, Predator
from swarm_sim.core.world import World

__all__ = [
    "SimulationConfig",
    "Environment",
    "CellType",
    "Predator",
    "World",
]
