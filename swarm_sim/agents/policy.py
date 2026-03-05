"""
Reinforcement Learning Policy — action selection via Q-learning.

Uses the agent's inner state vector (from the Neural Network) as the
state representation and learns Q-values for each action. The policy
selects actions using epsilon-greedy exploration, where epsilon is
controlled by the genome's exploration_rate gene.

Architecture:
  State  = inner state vector (5 floats: hunger, fear, curiosity, loneliness, aggression)
           + situational features (nearby food count, nearby predators, nearby agents, on_food flag)
  Actions = 8 possible actions (MOVE_UP/DOWN/LEFT/RIGHT, STAY, EAT, COMMUNICATE, REPRODUCE)
  Q-function = small numpy MLP mapping state -> Q-values for each action

Reward components (genome-weighted):
  - Survival: +1 per step alive
  - Food:     +energy_gained from eating
  - Social:   +bonus for being near other agents (scaled by affiliation_need)
  - Exploration: +bonus for visiting new regions (scaled by adventurousness)
  - Danger:   -penalty for being near predators
  - Starvation: -penalty when energy is critically low

The RL policy replaces the heuristic decide() in the Agent.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from swarm_sim.agents.neural import sigmoid, NUM_INNER_STATES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ACTIONS = 8  # matches Action enum (0-7)

# State features: inner_state (5) + situational (4) = 9
NUM_SITUATIONAL = 4  # nearby_food_norm, nearby_preds_norm, nearby_agents_norm, on_food
STATE_SIZE = NUM_INNER_STATES + NUM_SITUATIONAL  # 9


# ---------------------------------------------------------------------------
# Q-Network (small MLP)
# ---------------------------------------------------------------------------

class QNetwork:
    """
    A small MLP that maps state -> Q-values for each action.

    Architecture:
        input (STATE_SIZE=9) -> hidden (hidden_size, ReLU) -> output (NUM_ACTIONS=8)

    Trained via TD(0) Q-learning update.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        learning_rate: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.input_size = STATE_SIZE
        self.hidden_size = max(4, hidden_size)
        self.output_size = NUM_ACTIONS
        self.learning_rate = learning_rate

        # Xavier init
        self.W1 = self.rng.normal(
            0, np.sqrt(2.0 / (self.input_size + self.hidden_size)),
            (self.input_size, self.hidden_size)
        ).astype(np.float32)
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)

        self.W2 = self.rng.normal(
            0, np.sqrt(2.0 / (self.hidden_size + self.output_size)),
            (self.hidden_size, self.output_size)
        ).astype(np.float32)
        self.b2 = np.zeros(self.output_size, dtype=np.float32)

        # Cache
        self._input: Optional[np.ndarray] = None
        self._hidden: Optional[np.ndarray] = None
        self._output: Optional[np.ndarray] = None

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions given a state vector."""
        x = state.astype(np.float32)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        elif len(x) > self.input_size:
            x = x[:self.input_size]

        z1 = x @ self.W1 + self.b1
        h = np.maximum(0, z1)  # ReLU

        q_values = h @ self.W2 + self.b2

        # NaN safety
        if not np.all(np.isfinite(q_values)):
            self._reset_weights()
            q_values = np.zeros(self.output_size, dtype=np.float32)
            h = np.zeros(self.hidden_size, dtype=np.float32)

        self._input = x
        self._hidden = h
        self._output = q_values

        return q_values

    def update(self, action: int, td_target: float) -> float:
        """
        Update weights for a single (state, action) pair via TD error.

        Parameters
        ----------
        action : int
            The action that was taken.
        td_target : float
            The target Q-value: reward + gamma * max(Q(s', a'))

        Returns
        -------
        float : absolute TD error
        """
        if self._output is None or self._input is None:
            return 0.0

        # Clamp td_target to prevent explosion
        td_target = float(np.clip(td_target, -20.0, 20.0))

        current_q = self._output[action]
        if not np.isfinite(current_q):
            self._reset_weights()
            return 0.0

        td_error = td_target - current_q
        td_error = float(np.clip(td_error, -5.0, 5.0))  # Clip gradient
        lr = self.learning_rate

        # Gradient for output layer (only the selected action)
        d_output = np.zeros(self.output_size, dtype=np.float32)
        d_output[action] = -td_error

        # Hidden layer gradient
        d_hidden = (d_output @ self.W2.T) * (self._hidden > 0).astype(np.float32)

        # Gradient clipping
        grad_norm = np.sqrt(np.sum(d_hidden**2) + np.sum(d_output**2) + 1e-8)
        if grad_norm > 1.0:
            d_hidden /= grad_norm
            d_output /= grad_norm

        # Weight updates (gradient descent)
        self.W2 -= lr * np.outer(self._hidden, d_output)
        self.b2 -= lr * d_output
        self.W1 -= lr * np.outer(self._input, d_hidden)
        self.b1 -= lr * d_hidden

        # Weight clamping to prevent explosion
        self.W1 = np.clip(self.W1, -5.0, 5.0)
        self.W2 = np.clip(self.W2, -5.0, 5.0)
        self.b1 = np.clip(self.b1, -5.0, 5.0)
        self.b2 = np.clip(self.b2, -5.0, 5.0)

        return float(abs(td_error))

    def _reset_weights(self) -> None:
        """Reset weights if NaN detected."""
        self.W1 = self.rng.normal(
            0, 0.1, (self.input_size, self.hidden_size)
        ).astype(np.float32)
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)
        self.W2 = self.rng.normal(
            0, 0.1, (self.hidden_size, self.output_size)
        ).astype(np.float32)
        self.b2 = np.zeros(self.output_size, dtype=np.float32)

    def clone(self) -> "QNetwork":
        new = QNetwork(
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            rng=self.rng,
        )
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        return new


# ---------------------------------------------------------------------------
# RL Policy
# ---------------------------------------------------------------------------

class RLPolicy:
    """
    Epsilon-greedy Q-learning policy for action selection.

    Attributes
    ----------
    epsilon : float
        Exploration rate (from genome's exploration_rate gene).
    gamma : float
        Discount factor for future rewards.
    q_net : QNetwork
        The Q-value function approximator.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        gamma: float = 0.95,
        hidden_size: int = 16,
        learning_rate: float = 0.01,
        affiliation_need: float = 0.5,
        adventurousness: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.epsilon = epsilon
        self.gamma = gamma
        self.affiliation_need = affiliation_need
        self.adventurousness = adventurousness

        self.q_net = QNetwork(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            rng=self.rng,
        )

        # Experience tracking
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_energy: float = 0.0

        # Stats
        self.total_decisions: int = 0
        self.total_explorations: int = 0
        self.total_exploitations: int = 0
        self.cumulative_reward: float = 0.0
        self.cumulative_td_error: float = 0.0

        # Visited regions (for exploration bonus)
        self._visited_regions: set = set()

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def build_state(
        self,
        inner_state: Dict[str, float],
        observation: Dict[str, Any],
    ) -> np.ndarray:
        """
        Build the state vector from inner state + situational features.

        Returns np.ndarray of shape (STATE_SIZE,)
        """
        # Inner state (5 floats)
        inner = np.array([
            inner_state.get("hunger", 0.0),
            inner_state.get("fear", 0.0),
            inner_state.get("curiosity", 0.0),
            inner_state.get("loneliness", 0.0),
            inner_state.get("aggression", 0.0),
        ], dtype=np.float32)

        # Situational features
        n_food = len(observation.get("food_positions", []))
        n_preds = len(observation.get("predator_positions", []))
        n_agents = len(observation.get("agent_positions", []))
        on_food = 1.0 if (0, 0) in observation.get("food_positions", []) else 0.0

        situational = np.array([
            min(1.0, n_food / 10.0),
            min(1.0, n_preds / 3.0),
            min(1.0, n_agents / 5.0),
            on_food,
        ], dtype=np.float32)

        return np.concatenate([inner, situational])

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        inner_state: Dict[str, float],
        observation: Dict[str, Any],
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        Returns action index (0-7, matching Action enum).
        """
        state = self.build_state(inner_state, observation)

        self.total_decisions += 1

        if self.rng.random() < self.epsilon:
            # Exploration: random action
            action = int(self.rng.integers(0, NUM_ACTIONS))
            self.total_explorations += 1
        else:
            # Exploitation: pick best Q-value
            q_values = self.q_net.forward(state)

            # Mask invalid actions (COMMUNICATE and REPRODUCE need conditions)
            # For now, allow all actions; invalid ones just waste a step
            action = int(np.argmax(q_values))
            self.total_exploitations += 1

        # Store for learning
        self._prev_state = state
        self._prev_action = action

        return action

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        energy_change: float,
        ate: bool,
        nearby_agents: int,
        nearby_predators: int,
        agent_x: int,
        agent_y: int,
        is_isolated: bool,
        alive: bool,
    ) -> float:
        """
        Compute the composite reward signal.

        Components:
          - Survival: small constant for staying alive
          - Food: bonus for eating
          - Social: bonus for being near agents (genome-weighted)
          - Exploration: bonus for new regions
          - Danger: penalty for predator proximity
          - Death: large penalty
        """
        reward = 0.0

        # Survival bonus
        if alive:
            reward += 0.1

        # Food reward
        if ate:
            reward += 2.0
        elif energy_change > 0:
            reward += energy_change / 20.0

        # Energy management
        if energy_change < -5:
            reward -= 0.5  # Large energy loss (predator hit)

        # Social reward (genome-weighted)
        if nearby_agents > 0 and not is_isolated:
            reward += self.affiliation_need * min(1.0, nearby_agents * 0.3)

        # Exploration reward
        region = (agent_x // 5, agent_y // 5)
        if region not in self._visited_regions:
            self._visited_regions.add(region)
            reward += self.adventurousness * 0.5

        # Danger penalty
        if nearby_predators > 0:
            reward -= nearby_predators * 0.3

        # Death penalty
        if not alive:
            reward -= 5.0

        return reward

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def learn(
        self,
        reward: float,
        next_inner_state: Dict[str, float],
        next_observation: Dict[str, Any],
        done: bool = False,
    ) -> float:
        """
        Perform a Q-learning update.

        Parameters
        ----------
        reward : float
            Reward received after the action.
        next_inner_state : dict
            Inner state after the action.
        next_observation : dict
            Observation after the action.
        done : bool
            Whether the agent died (terminal state).

        Returns
        -------
        float : TD error magnitude
        """
        if self._prev_state is None or self._prev_action is None:
            return 0.0

        self.cumulative_reward += reward

        # Compute TD target
        if done:
            td_target = reward
        else:
            next_state = self.build_state(next_inner_state, next_observation)
            next_q_values = self.q_net.forward(next_state)
            td_target = reward + self.gamma * float(np.max(next_q_values))

        # Re-forward previous state (needed for backprop)
        self.q_net.forward(self._prev_state)

        # Update
        td_error = self.q_net.update(self._prev_action, td_target)
        self.cumulative_td_error += td_error

        return td_error

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self, decay_rate: float = 0.999, min_epsilon: float = 0.05) -> None:
        """Gradually reduce exploration rate."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return policy statistics."""
        return {
            "epsilon": round(self.epsilon, 4),
            "total_decisions": self.total_decisions,
            "explorations": self.total_explorations,
            "exploitations": self.total_exploitations,
            "exploration_pct": round(
                self.total_explorations / max(self.total_decisions, 1) * 100, 1
            ),
            "cumulative_reward": round(self.cumulative_reward, 2),
            "avg_reward": round(
                self.cumulative_reward / max(self.total_decisions, 1), 4
            ),
            "avg_td_error": round(
                self.cumulative_td_error / max(self.total_decisions, 1), 4
            ),
            "visited_regions": len(self._visited_regions),
        }

    def clone(self) -> "RLPolicy":
        new = RLPolicy(
            epsilon=self.epsilon,
            gamma=self.gamma,
            hidden_size=self.q_net.hidden_size,
            learning_rate=self.q_net.learning_rate,
            affiliation_need=self.affiliation_need,
            adventurousness=self.adventurousness,
            rng=self.rng,
        )
        new.q_net = self.q_net.clone()
        new._visited_regions = set(self._visited_regions)
        return new

    def __repr__(self) -> str:
        return (
            f"RLPolicy(eps={self.epsilon:.3f}, "
            f"decisions={self.total_decisions}, "
            f"reward={self.cumulative_reward:.1f})"
        )
