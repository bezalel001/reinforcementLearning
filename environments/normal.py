from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from environments.base import Environment, State, Action

@dataclass
class MABTestAction(Action):
    """Represents a single available action (e.g., an arm in a multi-armed bandit)."""
    id: int  # Stable identity that only increases when new arms appear


@dataclass
class MABTestState(State):
    """Represents the environment state — here, just a catalog of available actions."""
    actions: List[MABTestAction]
    context: int
    steps: int


class MABTestEnvironment(Environment):
    """
    Multi-Armed Bandit (MAB) test environment with:
      - Optional non-stationary drift (Gaussian random walk of true means).
      - Random action turnover governed by a Poisson process per step.

    Notes:
      - The environment exposes exactly `n_actions` *slots*.
      - Each slot holds an action with a stable *ID* (monotonically increasing as new ones arrive).
      - On turnover, we remove some existing actions and add the same number of brand-new ones,
        chosen at random slots (no replacement within the same step).

    Attributes:
        rng: Random number generator.
        n_actions: Number of action slots.
        n_contexts: Number of discrete contexts.
        q_star: True reward means, shape (n_contexts, n_actions) for the current slot layout.
        action_ids: np.ndarray[int] length n_actions, mapping slot -> action ID.
        next_action_id: Next new ID to assign on turnover.
        stationary: If False, apply Gaussian drift to q_star each step.
        drift_std: Std-dev of the Gaussian drift when non-stationary.
        turnover_rate: Poisson rate lambda for expected number of replacements per step.
        replacement_std: Std-dev used to initialize a newly added action’s means.
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        n_actions: int = 10,
        n_contexts: int = 1,
        *,
        stationary: bool = True,
        drift_std: float = 0.01,
        turnover_rate: float = 0.0,   # lambda for Poisson(k)
        replacement_std: float = 1.0,
    ) -> None:
        if not isinstance(rng, np.random.RandomState):
            raise TypeError("rng must be an instance of np.random.RandomState.")

        if not (type(n_actions) is int):
            raise TypeError("`n_actions` must be an integer.")
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")

        if not (type(n_contexts) is int):
            raise TypeError("`n_contexts` must be an integer.")
        if n_contexts <= 0:
            raise ValueError("n_contexts must be positive.")

        if not isinstance(stationary, bool):
            raise TypeError("`stationary` must be a bool.")
        if drift_std < 0:
            raise ValueError("`drift_std` must be non-negative.")

        if turnover_rate < 0:
            raise ValueError("`turnover_rate` (Poisson lambda) must be non-negative.")
        if replacement_std <= 0:
            raise ValueError("`replacement_std` must be positive.")

        self.rng = rng
        self.n_actions = n_actions
        self.n_contexts = n_contexts
        self.stationary = stationary
        self.drift_std = drift_std
        self.turnover_rate = turnover_rate
        self.replacement_std = replacement_std

        # True means per (context, slot)
        self.q_star = rng.normal(loc=0.0, scale=1.0, size=(n_contexts, n_actions))

        # Slot -> action ID mapping; start with 0..n_actions-1, then grow monotonically
        self.action_ids = np.arange(n_actions, dtype=int)
        self.next_action_id = int(n_actions)

        self.current_context = rng.randint(0, n_contexts)
        self.t = 0

    def state(self) -> MABTestState:
        return MABTestState(
            actions=[MABTestAction(id=int(aid)) for aid in self.action_ids],
            context=self.current_context,
            steps=self.t
        )

    def sample_contexts(self, samples: int = 10_000) -> np.ndarray:
        """Return integer context IDs of shape (samples, 1)."""
        if samples <= 0:
            raise ValueError("samples must be positive.")
        return self.rng.randint(0, self.n_contexts, size=(samples, 1))

    def _apply_drift(self) -> None:
        """
        Apply mean-reverting Gaussian drift to q_star (non-stationary setting).

        This keeps q_star approximately distributed as N(0,1) over time,
        preventing unbounded growth in magnitude.
        """
        if not self.stationary:
            alpha = self.drift_std  # reuse drift_std as mean-reversion rate in (0,1)
            if not (0 < alpha < 1):
                raise ValueError("For mean-reverting drift, drift_std must be in (0, 1).")

            sigma = np.sqrt(2 * alpha - alpha ** 2)
            self.q_star = (1 - alpha) * self.q_star + self.rng.normal(
                loc=0.0, scale=sigma, size=self.q_star.shape
            )

    def _turnover_poisson(self) -> None:
        """
        Draw k ~ Poisson(lambda = turnover_rate). Replace k randomly chosen *distinct* slots
        (capped at n_actions) with brand-new actions (new IDs and new means).
        """
        if self.turnover_rate <= 0:
            return

        # Poisson draw for the number of replacements this step
        k = int(self.rng.poisson(lam=self.turnover_rate))
        if k <= 0:
            return

        k = min(k, self.n_actions)  # cap to available slots

        # Choose k distinct slots uniformly at random
        slots = self.rng.choice(self.n_actions, size=k, replace=False)

        # Replace each chosen slot with a new action (new ID + resampled means)
        for slot in np.atleast_1d(slots):
            # Sample new means across all contexts
            self.q_star[:, slot] = self.rng.normal(
                loc=0.0, scale=self.replacement_std, size=(self.n_contexts,)
            )
            # Assign a new, monotonically increasing ID
            self.action_ids[slot] = self.next_action_id
            self.next_action_id += 1

    def step(self, article_id: int) -> float:
        """
        Take one action and return a stochastic reward.

        NOTE: `article_id` is the **slot index** in [0, n_actions).
              The identity of that slot is available as `state().actions[article_id].id`.
        """
        if not (type(article_id) is int):
            raise TypeError("article_id must be an integer.")
        if not 0 <= article_id < self.n_actions:
            raise ValueError(f"article_id must be between 0 and {self.n_actions - 1}.")

        mean = self.q_star[self.current_context, article_id]
        reward = self.rng.normal(loc=mean, scale=0.05)

        # Advance context each step
        self.current_context = self.rng.randint(0, self.n_contexts)

        # Evolve the environment
        self._apply_drift()
        self._turnover_poisson()
        self.t += 1
        return float(reward)
