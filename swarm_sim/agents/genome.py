"""
Genome module — heritable genetic encoding for agents.

Each genome is a dictionary of genes that control innate parameters.
Genes are fixed during an agent's lifetime but passed to offspring
via crossover and mutation (handled in the evolution module).

Gene Definitions:
  - adventurousness    : probability to voluntarily leave the swarm [0, 1]
  - affiliation_need   : reward multiplier for being near others [0, 1]
  - xenophobia         : negative reaction to strangers [0, 1]
  - plasticity         : learning rate for neural network [0.001, 0.1]
  - exploration_rate   : epsilon for RL exploration [0, 1]
  - nn_hidden_size     : neurons in hidden layers [4, 64] (int)
  - memory_size        : capacity for storing experiences [10, 200] (int)
  - reproduction_threshold : energy required to reproduce [50, 180]
  - lifespan           : maximum age in steps [100, 2000] (int)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


# ---------------------------------------------------------------------------
# Gene specification — defines valid range and type for each gene
# ---------------------------------------------------------------------------

GENE_SPEC: Dict[str, Dict[str, Any]] = {
    "adventurousness": {
        "min": 0.0, "max": 1.0, "dtype": float, "default": 0.3,
        "description": "Probability to voluntarily leave the swarm",
    },
    "affiliation_need": {
        "min": 0.0, "max": 1.0, "dtype": float, "default": 0.5,
        "description": "Reward multiplier for proximity to other agents",
    },
    "xenophobia": {
        "min": 0.0, "max": 1.0, "dtype": float, "default": 0.2,
        "description": "Negative reaction intensity toward strangers",
    },
    "plasticity": {
        "min": 0.001, "max": 0.1, "dtype": float, "default": 0.01,
        "description": "Learning rate for neural network",
    },
    "exploration_rate": {
        "min": 0.0, "max": 1.0, "dtype": float, "default": 0.3,
        "description": "Epsilon for RL exploration vs exploitation",
    },
    "nn_hidden_size": {
        "min": 4, "max": 64, "dtype": int, "default": 16,
        "description": "Number of neurons in NN hidden layers",
    },
    "memory_size": {
        "min": 10, "max": 200, "dtype": int, "default": 50,
        "description": "Capacity for storing past experiences",
    },
    "reproduction_threshold": {
        "min": 50.0, "max": 180.0, "dtype": float, "default": 100.0,
        "description": "Minimum energy required to reproduce",
    },
    "lifespan": {
        "min": 100, "max": 2000, "dtype": int, "default": 500,
        "description": "Maximum age in simulation steps",
    },
}

GENE_NAMES: List[str] = list(GENE_SPEC.keys())


# ---------------------------------------------------------------------------
# Genome class
# ---------------------------------------------------------------------------

class Genome:
    """
    Immutable genetic encoding for a single agent.

    Attributes
    ----------
    genes : dict[str, float | int]
        Mapping of gene name → value. Values are clipped to valid ranges.
    """

    __slots__ = ("genes",)

    def __init__(self, genes: Optional[Dict[str, Any]] = None):
        """
        Create a genome from an explicit gene dict.

        Parameters
        ----------
        genes : dict, optional
            Gene values. Missing genes filled with defaults.
            Values are clipped to their valid ranges.
        """
        self.genes: Dict[str, Any] = {}
        base = genes or {}

        for name, spec in GENE_SPEC.items():
            raw = base.get(name, spec["default"])
            self.genes[name] = self._clip_gene(name, raw)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def get_spec() -> Dict[str, Tuple[float, float]]:
        """Return gene name -> (min, max) mapping."""
        return {name: (spec["min"], spec["max"]) for name, spec in GENE_SPEC.items()}

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> "Genome":
        """Create a genome with uniformly random gene values."""
        rng = rng or np.random.default_rng()
        genes = {}
        for name, spec in GENE_SPEC.items():
            if spec["dtype"] == int:
                genes[name] = int(rng.integers(spec["min"], spec["max"] + 1))
            else:
                genes[name] = float(rng.uniform(spec["min"], spec["max"]))
        return cls(genes)

    @classmethod
    def default(cls) -> "Genome":
        """Create a genome with all default values."""
        return cls()

    @classmethod
    def from_parents(
        cls,
        parent_a: "Genome",
        parent_b: "Genome",
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> "Genome":
        """
        Create a child genome via sexual reproduction.

        Parameters
        ----------
        parent_a, parent_b : Genome
            The two parents.
        crossover_rate : float
            Probability that crossover occurs (vs cloning parent_a).
        mutation_rate : float
            Per-gene probability of mutation.
        mutation_strength : float
            Magnitude of mutation perturbation (fraction of gene range).
        rng : np.random.Generator, optional
        """
        rng = rng or np.random.default_rng()
        child_genes = {}

        # Decide: crossover or clone
        do_crossover = rng.random() < crossover_rate

        if do_crossover:
            # Uniform crossover: each gene randomly from either parent
            for name in GENE_NAMES:
                if rng.random() < 0.5:
                    child_genes[name] = parent_a.genes[name]
                else:
                    child_genes[name] = parent_b.genes[name]
        else:
            # Clone parent_a
            child_genes = dict(parent_a.genes)

        # Mutation
        for name in GENE_NAMES:
            if rng.random() < mutation_rate:
                spec = GENE_SPEC[name]
                gene_range = spec["max"] - spec["min"]
                perturbation = rng.normal(0, mutation_strength * gene_range)
                child_genes[name] = child_genes[name] + perturbation

        return cls(child_genes)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, gene_name: str) -> Any:
        return self.genes[gene_name]

    def __contains__(self, gene_name: str) -> bool:
        return gene_name in self.genes

    def get(self, gene_name: str, default: Any = None) -> Any:
        return self.genes.get(gene_name, default)

    # ------------------------------------------------------------------
    # Comparison & analysis
    # ------------------------------------------------------------------

    def distance(self, other: "Genome") -> float:
        """
        Compute normalized genetic distance to another genome.

        Returns a value in [0, 1] where 0 = identical, 1 = maximally different.
        Each gene contributes equally (normalized by its range).
        """
        total = 0.0
        for name, spec in GENE_SPEC.items():
            gene_range = spec["max"] - spec["min"]
            if gene_range > 0:
                diff = abs(self.genes[name] - other.genes[name]) / gene_range
                total += diff
        return total / len(GENE_SPEC)

    def to_vector(self) -> np.ndarray:
        """
        Convert genome to a normalized numpy vector in [0, 1].

        Useful for neural network input or distance calculations.
        """
        vec = np.zeros(len(GENE_NAMES), dtype=np.float32)
        for i, name in enumerate(GENE_NAMES):
            spec = GENE_SPEC[name]
            gene_range = spec["max"] - spec["min"]
            if gene_range > 0:
                vec[i] = (self.genes[name] - spec["min"]) / gene_range
            else:
                vec[i] = 0.0
        return vec

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict copy of the genes."""
        return dict(self.genes)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_gene(name: str, value: Any) -> Any:
        """Clip a gene value to its valid range and correct type."""
        spec = GENE_SPEC[name]
        if spec["dtype"] == int:
            return int(np.clip(int(round(value)), spec["min"], spec["max"]))
        else:
            return float(np.clip(float(value), spec["min"], spec["max"]))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in self.genes.items()]
        return f"Genome({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Genome):
            return NotImplemented
        return self.genes == other.genes
