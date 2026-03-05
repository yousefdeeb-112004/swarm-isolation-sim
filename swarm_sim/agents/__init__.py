"""Agent architecture: genome, beliefs, neural network, RL policy, interactions."""

from swarm_sim.agents.genome import Genome, GENE_SPEC, GENE_NAMES
from swarm_sim.agents.agent import Agent, Action, MemoryEntry
from swarm_sim.agents.bayesian import BeliefNetwork
from swarm_sim.agents.neural import (
    InnerStateNetwork, build_vitals_vector, compute_heuristic_inner_state,
    INNER_STATE_NAMES, NUM_INNER_STATES,
)
from swarm_sim.agents.policy import RLPolicy, QNetwork, STATE_SIZE, NUM_ACTIONS
from swarm_sim.agents.interaction import InteractionManager

__all__ = [
    "Genome", "GENE_SPEC", "GENE_NAMES",
    "Agent", "Action", "MemoryEntry",
    "BeliefNetwork",
    "InnerStateNetwork", "build_vitals_vector", "compute_heuristic_inner_state",
    "INNER_STATE_NAMES", "NUM_INNER_STATES",
    "RLPolicy", "QNetwork", "STATE_SIZE", "NUM_ACTIONS",
    "InteractionManager",
]
