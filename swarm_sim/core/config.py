"""
Configuration dataclasses for the simulation.

Loads from YAML and provides typed, validated access to all parameters.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FoodConfig:
    initial_count: int = 150
    energy_value: int = 20
    regeneration_rate: float = 0.02
    max_food: int = 200
    cluster_probability: float = 0.6
    cluster_radius: int = 5


@dataclass
class ObstacleConfig:
    count: int = 30


@dataclass
class PredatorConfig:
    count: int = 5
    energy_damage: int = 30
    speed: int = 1
    detection_range: int = 8
    patrol_radius: int = 15


@dataclass
class EnvironmentConfig:
    food: FoodConfig = field(default_factory=FoodConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    predators: PredatorConfig = field(default_factory=PredatorConfig)


@dataclass
class WorldConfig:
    width: int = 100
    height: int = 100
    max_steps: int = 1000
    seed: Optional[int] = 42


@dataclass
class AgentConfig:
    population_size: int = 50
    sensor_range: int = 7
    initial_energy: int = 100
    energy_per_step: int = -1
    max_energy: int = 200


@dataclass
class EvolutionConfig:
    mutation_rate: float = 0.05
    mutation_strength: float = 0.1
    crossover_rate: float = 0.7
    selection_method: str = "tournament"
    tournament_size: int = 3
    elitism_count: int = 2


@dataclass
class ExperimentConfig:
    isolation_duration: int = 100
    isolation_distance: int = 80
    num_generations: int = 50
    isolation_frequency: int = 5
    selection_criteria: str = "adventurousness"


@dataclass
class LoggingConfig:
    log_interval: int = 10
    export_format: str = "csv"
    output_dir: str = "data/exports"
    verbose: bool = True


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Root configuration container for the entire simulation."""

    world: WorldConfig = field(default_factory=WorldConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    # Factory: load from YAML
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, data: dict) -> "SimulationConfig":
        """Recursively build config from a nested dictionary."""
        env_raw = data.get("environment", {})
        env_cfg = EnvironmentConfig(
            food=FoodConfig(**env_raw.get("food", {})),
            obstacles=ObstacleConfig(**env_raw.get("obstacles", {})),
            predators=PredatorConfig(**env_raw.get("predators", {})),
        )

        return cls(
            world=WorldConfig(**data.get("world", {})),
            environment=env_cfg,
            agents=AgentConfig(**data.get("agents", {})),
            evolution=EvolutionConfig(**data.get("evolution", {})),
            experiment=ExperimentConfig(**data.get("experiment", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    # ------------------------------------------------------------------
    # Factory: defaults
    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "SimulationConfig":
        """Return configuration with all default values."""
        return cls()
