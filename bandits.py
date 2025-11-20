# bandits.py
# -----------------------------------------------------------------------------
# Multi-Armed Bandits: Agent Interfaces and Implementations
#
# This module implements several bandit agents that share a common interface:
#   - RandomAgent         : uniform random baseline
#   - EpsilonGreedyAgent  : E-greedy with optional linear decay and step-size
#   - UCBAgent            : UCB exploration bonus (Auer et al., 2002)
#   - GradientBandit      : preference-based (policy gradient) bandit
#
# Assumptions about the environment:
#   - The environment exposes a State object `s` with `s.actions`, a list of
#     Action objects. Each Action has a unique integer identifier `a.id`.
#   - The set of available actions can change over time (arms may appear/disappear).
#     Agents must lazily initialize any NEW arm that becomes available and should
#     simply ignore actions that are not currently available.
#
# Example Usage:
#   agent = EpsilonGreedyAgent(rng=np.random.RandomState(0))
#   arm_id = agent.select(state)
#   env_reward = env.step(arm_id)
#   agent.update(arm_id, env_reward, state)
#
# Testing:
#   pytest test_bandits_egreedy.py
#   pytest test_bandits_gb.py
#   pytest test_bandits_ucb.py
#   pytest test_bandits_contextual.py
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from sklearn.cluster import KMeans
from environments.base import Environment, State, Action


# -----------------------------------------------------------------------------
# Base Interface
# -----------------------------------------------------------------------------
class BanditAgent:
    """
    Abstract base class for multi-armed bandit agents keyed by integer arm IDs.

    Agents must implement two core methods:

    - select(self, s: State) -> int
        Return the ID of the chosen action (arm) for the current state `s`.
        Selection considers ONLY the actions in `s.actions`.

    - update(self, arm_id: int, reward: float, s: State) -> None
        Incorporate the observed `reward` after selecting `arm_id` in state `s`.
        Implementations typically update value estimates, counts, or preferences.

    Environment contract (assumed):
      - s.actions: List[Action] — actions available *at this timestep*.
      - Each Action `a` has an integer `a.id` used as the key for agent state.
      - The set of available actions may change over time (dynamic arms).

    Implementation note:
      Agents should lazily initialize per-arm statistics the first time an arm
      appears in `s.actions`. This must happen in the `select` method. Arms that
      are not in `s.actions` at selection time must be ignored when computing 
      argmax/softmax/etc.
    """

    def select(self, s: State) -> int:
        raise NotImplementedError

    def update(self, arm_id: int, reward: float, s: State) -> None:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Random Agent
# -----------------------------------------------------------------------------
class RandomAgent(BanditAgent):
    """
    Uniform random baseline.

    Selects an available action uniformly at random, ignoring all history.

    Attributes
    ----------
    rng : np.random.RandomState
        Random number generator.
    N : dict
        Total number of selections made for action (not required, just for basic bookkeeping).
    """

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.N = {}

    def select(self, s: State) -> int:
        """Select a random arm among currently available IDs."""
        if not isinstance(s, State): raise TypeError("s must be a State")

        # Lazy-init stats for any newly available arms
        for a in [a.id for a in s.actions]:
            if a not in self.N:
                self.N[a] = 0
        
        # Returns a random action
        return int(self.rng.choice([a.id for a in s.actions]))

    def update(self, arm_id: int, reward: float, s: State) -> None:
        """No-op: random agent does not learn from feedback."""
        if not isinstance(arm_id, int): raise TypeError("arm id must be int")
        if not isinstance(reward, float): raise TypeError("reward id must be float")
        if not isinstance(s, State): raise TypeError("s id must be State")
        if arm_id not in self.N or any([a.id not in self.N for a in s.actions]) or arm_id not in [a.id for a in s.actions]:
            raise ValueError(
                f"Unseen arm {arm_id}; call `select` before `update`."
            )
        self.N[arm_id] += 1

# -----------------------------------------------------------------------------
# Epsilon-Greedy Agent
# -----------------------------------------------------------------------------
class EpsilonGreedyAgent(BanditAgent):
    """
    E-greedy agent with constant step-size.

    Attributes
    ----------
    rng : np.random.RandomState
        Random number generator.
    epsilon : float
        Initial E for decay scheduling.
    alpha : Optional[float]
        Constant step-size for updates. If None, use sample-average.
    initial_value : float
        Initial Q[a] for unseen arms.
    Q : Dict[int, float]
        Value estimates per arm.
    N : Dict[int, int]
        Selection counts per arm.
    t : int
        Global timestep counter used for E-decay.
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        alpha: Optional[float] = None,
    ):
        # --- Type validation ---
        if not isinstance(rng, np.random.RandomState):
            raise TypeError(
                f"`rng` must be a numpy.random.RandomState instance, got {type(rng).__name__}"
            )
        if not isinstance(epsilon, (int, float)):
            raise TypeError(f"`epsilon` must be a float, got {type(epsilon).__name__}")
        if alpha is not None and not isinstance(alpha, (int, float)):
            raise TypeError(f"`alpha` must be numeric or None, got {type(alpha).__name__}")

        # --- Range validation ---
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"`epsilon` must be between 0 and 1 (inclusive), got {epsilon}")
        if alpha is not None and alpha <= 0:
            raise ValueError(f"`alpha` must be positive if provided, got {alpha}")

        # --- Assign fields ---
        self.rng = rng
        self.epsilon = float(epsilon)
        self.initial_value = float(initial_value)
        self.alpha = float(alpha) if alpha is not None else None

        # --- Internal state ---
        self.Q: Dict[int, float] = {}
        self.N: Dict[int, int] = {}
        self.t = 0


    def select(self, s: State) -> int:
        """Select an arm via E-greedy, initializing any new arms on the fly. Use argmax to select the greedy action; do not resolve tiebreaks via random selection."""
        if not isinstance(s, State):
            raise TypeError("s must be a State")

        available_ids = [a.id for a in s.actions]

        # Lazy-init any newly available arms
        for a_id in available_ids:
            if a_id not in self.Q:
                self.Q[a_id] = self.initial_value
                self.N[a_id] = 0

        # Exploration vs exploitation
        if self.rng.rand() < self.epsilon:
            return int(self.rng.choice(available_ids))

        # Greedy choice among currently available arms; Python's max is deterministic on ties
        best_id = max(available_ids, key=lambda aid: self.Q[aid])
        return int(best_id)

    def update(self, arm_id: int, reward: float, s: State) -> None:
        """Update Q[arm_id] from observed reward."""
        if not isinstance(arm_id, int): raise TypeError("arm id must be int")
        if not isinstance(reward, float): raise TypeError("reward id must be float")
        if not isinstance(s, State): raise TypeError("s id must be State")

        available_ids = [a.id for a in s.actions]
        if (
            arm_id not in self.Q
            or any(a_id not in self.Q for a_id in available_ids)
            or arm_id not in available_ids
        ):
            raise ValueError(
                f"Unseen arm {arm_id}; call `select` before `update`."
            )

        self.N[arm_id] += 1

        if self.alpha is None:
            # Sample-average update
            self.Q[arm_id] += (reward - self.Q[arm_id]) / self.N[arm_id]
        else:
            # Constant step-size update
            self.Q[arm_id] += self.alpha * (reward - self.Q[arm_id])


# -----------------------------------------------------------------------------
# UCB Agent
# -----------------------------------------------------------------------------
class UCBAgent(BanditAgent):
    """
    Upper Confidence Bound (UCB) agent.

    Attributes
    ----------
    c : float
        Exploration magnitude (non-negative). Larger c -> more exploration.
    initial_value : float
        Initial Q[a] for unseen arms.
    Q : Dict[int, float]
        Value estimates per arm.
    N : Dict[int, int]
        Selection counts per arm.
    t : int
        Global selection count (used in the log term).
    """

    def __init__(self, c: float = 2.0, initial_value: float = 0.0):
        # --- Type validation ---
        if not isinstance(c, (int, float)):
            raise TypeError(f"`c` must be numeric (float or int), got {type(c).__name__}")
        if not isinstance(initial_value, (int, float)):
            raise TypeError(
                f"`initial_value` must be numeric (float or int), got {type(initial_value).__name__}"
            )

        # --- Range validation ---
        if c < 0.0:
            raise ValueError(f"`c` must be non-negative, got {c}")
        # (No restriction on initial_value range — can be any real number)

        # --- Assign attributes ---
        self.c = float(c)
        self.initial_value = float(initial_value)

        # --- Initialize internal state ---
        self.Q: Dict[int, float] = {}
        self.N: Dict[int, int] = {}
        self.t = 0


    def select(self, s: State) -> int:
        """Select arm maximizing UCB index; ensure each new arm is tried once."""
        if not isinstance(s, State):
            raise TypeError("s must be a State")

        available_ids = [a.id for a in s.actions]

        # Lazy-init new arms
        for a_id in available_ids:
            if a_id not in self.Q:
                self.Q[a_id] = self.initial_value
                self.N[a_id] = 0

        # Force each unseen arm to be selected once (in available order)
        for a_id in available_ids:
            if self.N[a_id] == 0:
                return int(a_id)

        # Compute UCB index for available arms
        def ucb_val(a_id: int) -> float:
            return self.Q[a_id] + self.c * np.sqrt(np.log(max(1, self.t)) / self.N[a_id])

        best_id = max(available_ids, key=ucb_val)
        return int(best_id)


    def update(self, arm_id: int, reward: float, s: State) -> None:
        """Increment time, update counts, and apply sample-average value update."""
        if not isinstance(arm_id, int): raise TypeError("arm id must be int")
        if not isinstance(reward, float): raise TypeError("reward id must be float")
        if not isinstance(s, State): raise TypeError("s id must be State")

        available_ids = [a.id for a in s.actions]
        if (
            arm_id not in self.Q
            or any(a_id not in self.Q for a_id in available_ids)
            or arm_id not in available_ids
        ):
            raise ValueError(
                f"Unseen arm {arm_id}; call `select` before `update`."
            )

        self.t += 1
        self.N[arm_id] += 1
        # Sample-average update
        self.Q[arm_id] += (reward - self.Q[arm_id]) / self.N[arm_id]


# -----------------------------------------------------------------------------
# Gradient Bandit Agent (Preference-Based)
# -----------------------------------------------------------------------------
class GradientBandit(BanditAgent):
    """
    Preference-based (policy gradient) bandit (Sutton & Barto).

    Attributes
    ----------
    rng : np.random.RandomState
        Random number generator.
    alpha : float
        Learning rate for preference updates (alpha > 0).
    H : Dict[int, float]
        Preferences per arm.
    avg_reward : float
        Incremental average of observed rewards.
    t : int
        Timestep counter for the running average.
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        alpha: float = 0.1,
    ):
        # --- Type validation ---
        if not isinstance(rng, np.random.RandomState):
            raise TypeError(
                f"`rng` must be a numpy.random.RandomState instance, got {type(rng).__name__}"
            )
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"`alpha` must be numeric (float or int), got {type(alpha).__name__}")

        # --- Range validation ---
        if alpha <= 0:
            raise ValueError(f"`alpha` must be positive, got {alpha}")

        # --- Assign attributes ---
        self.alpha = float(alpha)
        self.H: Dict[int, float] = {}
        self.rng = rng
        self.avg_reward = 0.0
        self.t = 0
        self._eps = 1e-12  # numerical stability for softmax

    def _softmax(self, available_ids: List[int]) -> Dict[int, float]:
        """Return a dict of softmax probabilities over `available_ids` from H."""
        prefs = np.array([self.H[a] for a in available_ids], dtype=float)
        prefs = prefs - np.max(prefs)  # shift for numerical stability
        exp_p = np.exp(prefs)
        Z = float(np.sum(exp_p)) + self._eps
        probs = exp_p / Z
        return {a: float(p) for a, p in zip(available_ids, probs)}

    def select(self, s: State) -> int:
        """Sample an arm according to the softmax over preferences H[a]."""
        if not isinstance(s, State):
            raise TypeError("s must be a State")

        available_ids = [a.id for a in s.actions]

        # Lazy-init preferences for new arms
        for a_id in available_ids:
            if a_id not in self.H:
                self.H[a_id] = 0.0

        probs = self._softmax(available_ids)
        chosen = self.rng.choice(available_ids, p=[probs[a_id] for a_id in available_ids])
        return int(chosen)


    def update(self, arm_id: int, reward: float, s: State) -> None:
        """
        Apply the preference gradient update using the current set of available arms.
        """
        if not isinstance(arm_id, int): raise TypeError("arm id must be int")
        if not isinstance(reward, float): raise TypeError("reward id must be float")
        if not isinstance(s, State): raise TypeError("s id must be State")

        available_ids = [a.id for a in s.actions]
        if (
            arm_id not in self.H
            or any(a_id not in self.H for a_id in available_ids)
            or arm_id not in available_ids
        ):
            raise ValueError(
                f"Unseen arm {arm_id}; call `select` before `update`."
            )

        baseline = self.avg_reward
        probs = self._softmax(available_ids)

        # Update preferences only for currently available arms
        for a_id in available_ids:
            indicator = 1.0 if a_id == arm_id else 0.0
            self.H[a_id] += self.alpha * (reward - baseline) * (indicator - probs[a_id])

        # Update running average reward and timestep
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t


# -----------------------------
# Contextual Bandit Wrapper
# -----------------------------
@dataclass
class ContextualClusteredBandit(BanditAgent):
    """
    Wraps a non-contextual bandit agent with a user-clustered context.
    - Samples users from env, clusters their state vectors (env.state(user)).
    - Creates one independent bandit instance per cluster via base_agent_factory().
    - At selection time, assigns the current user to a cluster and delegates to that cluster's agent.

    Requirements on env:
    - env._sample_user(): returns a user object
    - env.state(user): returns a 1D numpy array (user feature/state)
    """
    env: Environment
    random_state: np.random.RandomState
    base_agent_factory: Callable[[np.random.RandomState], BanditAgent] = lambda x: RandomAgent(x)
    n_clusters: int = 2
    n_sample_contexts: int = 1000

    def __post_init__(self):
        if KMeans is None:
            raise ImportError(
                "scikit-learn is required for ContextualClusteredBandit. "
                "Please install scikit-learn."
            )

        # 1) sample users and collect state vectors
        X = self.env.sample_contexts(samples=self.n_sample_contexts)

        # 2) fit kmeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state.randint(91827))
        self.kmeans.fit(X)

        # 3) create one agent per cluster
        self.agents: Dict[int, BanditAgent] = {
            c: self.base_agent_factory(self.random_state) for c in range(self.n_clusters)
        }

        # keep last selected cluster to route updates if caller doesn't supply user again
        self._last_cluster: Optional[int] = None

        self.C = np.ascontiguousarray(self.kmeans.cluster_centers_, dtype=np.float32)  # (k, d)
        self.C_norm2 = np.einsum('ij,ij->i', self.C, self.C)                           # (k,)

    
    def _fast_predict(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        x = np.ascontiguousarray(x)
        # d^2(c_i, x) = ||c_i||^2 + ||x||^2 - 2 c_i·x
        x_norm2 = float(np.dot(x, x))
        dots = self.C @ x                           # (k,)
        d2 = self.C_norm2 + x_norm2 - 2.0 * dots    # (k,)
        return int(np.argmin(d2))


    def select(self, s: State) -> int:
        """
        Select an arm for the given user.
        If user is None, we attempt to call env.state(env._current_user) if available.
        For most reliability, pass the 'user' that will receive the recommendation.
        """
        if not isinstance(s, State):
            raise TypeError("s must be a State")
        if s.context is None:
            raise ValueError("User must be provided to select().")

        cluster = self._fast_predict(s.context)
        self._last_cluster = cluster
        return int(self.agents[cluster].select(s))

    def update(self, arm_id: int, reward: float, s: State) -> None:
        """
        Update the agent that handled the selection for the given user.
        If user is None, falls back to the last cluster used in select().
        """
        if not isinstance(arm_id, int): raise TypeError("arm id must be int")
        if not isinstance(reward, float): raise TypeError("reward id must be float")
        if not isinstance(s, State): raise TypeError("s id must be State")

        if s.context is None:
            if self._last_cluster is None:
                raise ValueError("User must be provided to select().")
            cluster = self._last_cluster
        else:
            cluster = self._fast_predict(s.context)
            self._last_cluster = cluster

        self.agents[cluster].update(arm_id, reward, s)
