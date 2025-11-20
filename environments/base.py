from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Action:
    """Represents a single available action (e.g., an arm in a multi-armed bandit)."""
    id: int


@dataclass
class State:
    """Represents the environment state — here, just a catalog of available actions."""
    actions: List[Action]
    context: np.ndarray
    steps: int


class Environment:

    def step(self, action: Action) -> State:
        raise NotImplementedError

    def state(self) -> State:
        raise NotImplementedError

    def sample_contexts(self, samples: int = 10_000) -> np.ndarray:
        raise NotImplementedError