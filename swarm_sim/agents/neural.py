"""
Neural Network Inner State Model — subjective experience of the agent.

Takes the belief vector from the Bayesian network + internal vitals
(energy, age, isolation flag) and produces an "inner state" vector
representing the agent's subjective experience: hunger, fear, curiosity,
loneliness, and aggression.

Architecture:
  Input  = belief_vector (100 floats) + vitals (normalized energy, age, isolation, genome traits)
  Hidden = genome-controlled hidden size (nn_hidden_size gene), 1-2 layers
  Output = inner state vector (5 floats, each in [0, 1])

This is a lightweight numpy-only MLP — no PyTorch/TensorFlow dependency.
The network learns online via simple backpropagation from a reward signal,
with learning rate controlled by the genome's plasticity gene.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def sigmoid_derivative(output: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its output."""
    return output * (1.0 - output)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(output: np.ndarray) -> np.ndarray:
    return 1.0 - output ** 2


# ---------------------------------------------------------------------------
# Inner State Labels
# ---------------------------------------------------------------------------

INNER_STATE_NAMES = ["hunger", "fear", "curiosity", "loneliness", "aggression"]
NUM_INNER_STATES = len(INNER_STATE_NAMES)

# Vitals appended to belief vector as NN input
NUM_VITALS = 8  # energy_norm, age_norm, is_isolated, adventurousness,
                # affiliation_need, xenophobia, plasticity, exploration_rate


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class InnerStateNetwork:
    """
    A small MLP that maps (beliefs + vitals) -> inner state vector.

    The network structure is:
        input (belief_size + NUM_VITALS)
        -> hidden layer 1 (hidden_size, tanh activation)
        -> hidden layer 2 (hidden_size // 2, tanh activation)
        -> output (NUM_INNER_STATES, sigmoid activation -> [0, 1])

    Learning:
        The network is trained online via simple backpropagation.
        A target inner state is computed from a heuristic (ground truth),
        and the network learns to approximate it. Over time, the network
        generalizes and may produce different inner states than the heuristic
        in novel situations (e.g., isolation).

    Parameters are controlled by the genome:
        - hidden_size from nn_hidden_size gene
        - learning_rate from plasticity gene
    """

    def __init__(
        self,
        belief_size: int = 100,
        hidden_size: int = 16,
        learning_rate: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.belief_size = belief_size
        self.input_size = belief_size + NUM_VITALS
        self.hidden1_size = max(4, hidden_size)
        self.hidden2_size = max(4, hidden_size // 2)
        self.output_size = NUM_INNER_STATES
        self.learning_rate = learning_rate

        # Xavier initialization
        self.W1 = self.rng.normal(
            0, np.sqrt(2.0 / (self.input_size + self.hidden1_size)),
            (self.input_size, self.hidden1_size)
        ).astype(np.float32)
        self.b1 = np.zeros(self.hidden1_size, dtype=np.float32)

        self.W2 = self.rng.normal(
            0, np.sqrt(2.0 / (self.hidden1_size + self.hidden2_size)),
            (self.hidden1_size, self.hidden2_size)
        ).astype(np.float32)
        self.b2 = np.zeros(self.hidden2_size, dtype=np.float32)

        self.W3 = self.rng.normal(
            0, np.sqrt(2.0 / (self.hidden2_size + self.output_size)),
            (self.hidden2_size, self.output_size)
        ).astype(np.float32)
        self.b3 = np.zeros(self.output_size, dtype=np.float32)

        # Cache for backprop
        self._input: Optional[np.ndarray] = None
        self._h1: Optional[np.ndarray] = None
        self._h2: Optional[np.ndarray] = None
        self._output: Optional[np.ndarray] = None

        # Training stats
        self.total_updates: int = 0
        self.cumulative_loss: float = 0.0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, belief_vector: np.ndarray, vitals: np.ndarray) -> np.ndarray:
        """
        Compute inner state from beliefs and vitals.

        Parameters
        ----------
        belief_vector : np.ndarray of shape (belief_size,)
            From BeliefNetwork.get_belief_vector()
        vitals : np.ndarray of shape (NUM_VITALS,)
            Normalized agent vitals.

        Returns
        -------
        np.ndarray of shape (NUM_INNER_STATES,)
            Values in [0, 1] for each inner state dimension.
        """
        # Concatenate input
        x = np.concatenate([belief_vector, vitals]).astype(np.float32)

        # Ensure correct size (pad or truncate if needed)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        elif len(x) > self.input_size:
            x = x[:self.input_size]

        # Hidden layer 1
        z1 = x @ self.W1 + self.b1
        h1 = tanh(z1)

        # Hidden layer 2
        z2 = h1 @ self.W2 + self.b2
        h2 = tanh(z2)

        # Output layer (sigmoid for [0, 1] range)
        z3 = h2 @ self.W3 + self.b3
        output = sigmoid(z3)

        # Cache for backprop
        self._input = x
        self._h1 = h1
        self._h2 = h2
        self._output = output

        return output

    # ------------------------------------------------------------------
    # Learning (online backpropagation)
    # ------------------------------------------------------------------

    def learn(self, target: np.ndarray) -> float:
        """
        Update weights via backpropagation toward a target inner state.

        Parameters
        ----------
        target : np.ndarray of shape (NUM_INNER_STATES,)
            The "correct" inner state (from heuristic).

        Returns
        -------
        float : MSE loss before the update
        """
        if self._output is None or self._input is None:
            return 0.0

        target = target.astype(np.float32)

        # MSE loss
        error = self._output - target
        loss = float(np.mean(error ** 2))

        # Output layer gradient
        d_output = error * sigmoid_derivative(self._output)  # (NUM_INNER_STATES,)

        # Hidden layer 2 gradient
        d_h2 = (d_output @ self.W3.T) * tanh_derivative(self._h2)  # (hidden2_size,)

        # Hidden layer 1 gradient
        d_h1 = (d_h2 @ self.W2.T) * tanh_derivative(self._h1)  # (hidden1_size,)

        # Weight updates
        lr = self.learning_rate

        self.W3 -= lr * np.outer(self._h2, d_output)
        self.b3 -= lr * d_output

        self.W2 -= lr * np.outer(self._h1, d_h2)
        self.b2 -= lr * d_h2

        self.W1 -= lr * np.outer(self._input, d_h1)
        self.b1 -= lr * d_h1

        self.total_updates += 1
        self.cumulative_loss += loss

        return loss

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_avg_loss(self) -> float:
        """Return average loss over all updates."""
        if self.total_updates == 0:
            return 0.0
        return self.cumulative_loss / self.total_updates

    def get_weight_stats(self) -> Dict[str, float]:
        """Return statistics about the network weights."""
        all_weights = np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3,
        ])
        return {
            "mean_weight": float(np.mean(all_weights)),
            "std_weight": float(np.std(all_weights)),
            "max_abs_weight": float(np.max(np.abs(all_weights))),
            "total_params": len(all_weights),
            "total_updates": self.total_updates,
            "avg_loss": self.get_avg_loss(),
        }

    def clone(self) -> "InnerStateNetwork":
        """Create a deep copy of this network."""
        new = InnerStateNetwork(
            belief_size=self.belief_size,
            hidden_size=self.hidden1_size,
            learning_rate=self.learning_rate,
            rng=self.rng,
        )
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        new.W3 = self.W3.copy()
        new.b3 = self.b3.copy()
        new.total_updates = self.total_updates
        new.cumulative_loss = self.cumulative_loss
        return new

    def __repr__(self) -> str:
        return (
            f"InnerStateNetwork(in={self.input_size}, "
            f"h1={self.hidden1_size}, h2={self.hidden2_size}, "
            f"out={self.output_size}, lr={self.learning_rate:.4f}, "
            f"updates={self.total_updates})"
        )


# ---------------------------------------------------------------------------
# Helper: build vitals vector from agent state
# ---------------------------------------------------------------------------

def build_vitals_vector(
    energy: float,
    max_energy: float,
    age: int,
    lifespan: int,
    is_isolated: bool,
    adventurousness: float,
    affiliation_need: float,
    xenophobia: float,
    plasticity: float,
    exploration_rate: float,
) -> np.ndarray:
    """
    Build a normalized vitals vector for NN input.

    Returns np.ndarray of shape (NUM_VITALS,) with values in [0, 1].
    """
    return np.array([
        energy / max(max_energy, 1.0),                  # energy_norm
        min(age / max(lifespan, 1), 1.0),               # age_norm
        1.0 if is_isolated else 0.0,                    # isolation flag
        adventurousness,                                 # genome trait
        affiliation_need,                                # genome trait
        xenophobia,                                      # genome trait
        plasticity / 0.1,                                # normalized plasticity
        exploration_rate,                                # genome trait
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: heuristic target for supervised learning
# ---------------------------------------------------------------------------

def compute_heuristic_inner_state(
    energy: float,
    max_energy: float,
    nearby_predators: int,
    nearby_food: int,
    nearby_agents: int,
    is_isolated: bool,
    adventurousness: float,
    affiliation_need: float,
    xenophobia: float,
) -> np.ndarray:
    """
    Compute a heuristic target inner state for training the NN.

    Returns np.ndarray of shape (NUM_INNER_STATES,) in [0, 1]:
        [hunger, fear, curiosity, loneliness, aggression]
    """
    hunger = max(0.0, 1.0 - energy / max(max_energy, 1.0))
    fear = min(1.0, nearby_predators * 0.5)
    curiosity = adventurousness * (1.0 - fear) * (0.3 + 0.7 * (1.0 - hunger))
    loneliness_base = max(0.0, 1.0 - nearby_agents * 0.25)
    loneliness = min(1.0, loneliness_base * affiliation_need + (0.3 if is_isolated else 0.0))
    aggression = xenophobia * min(1.0, nearby_agents * 0.2) * (0.5 + 0.5 * hunger)

    return np.array([hunger, fear, curiosity, loneliness, aggression], dtype=np.float32)
