from __future__ import annotations

from typing import List, Tuple, Any

import numpy as np

from bandits import ContextualClusteredBandit
from environments.base import State


# -----------------------------------------
# Utilities
# -----------------------------------------


def rollout(agent, env, n_steps: int) -> List[Tuple[Any, int, float]]:
    """
    Run a bandit for n_steps in the given env and return a trajectory:
        [(state_snapshot, action, reward), ...]
    - Works with contextual and non-contextual agents.
    - Assumes:
        * env._sample_user() -> user object
        * env.catalog: iterable of articles with .id
        * env._score(user, article) -> float
        * env.state(user) or env.state() -> any (snapshot stored verbatim)
    """
    # map article id -> article object once
    s = env.state()
    traj: List[Tuple[State, int, float]] = []

    for _ in range(n_steps):
        arm = agent.select(s)
        # take a step
        reward = env.step(arm)
        # update agent (pass user if agent supports it)
        agent.update(arm, reward, s)
        traj.append((s, int(arm), float(reward)))
        s = env.state()

    return traj


def get_rewards(trajectory):
    """
    Extract click labels from trajectory entries.
    Supports (state, action, reward) or (state, action, reward, click).
    If only reward is present, it's treated as the click label.
    """
    rewards = []
    time = []
    for step in trajectory:
        rewards.append(step[2])
        time.append(step[0].steps)
    return np.asarray(rewards, dtype=float), np.asarray(time, dtype=float)


def eval_agent(agent, env, n_steps: int):
    traj = rollout(agent, env, n_steps=n_steps)
    r, time = get_rewards(traj)
    return {
        "trajectory": traj,
        "time": time,
        "rewards": r,
        "average_reward": float(np.mean(r))
    }


def run_multi_seeds(build_agent_env_fn, seeds: List[int], n_steps: int):
    """
    build_agent_env_fn(seed) -> (agent, env)
    Returns dict with stacked rewards and summary stats across seeds.
    """
    rewards = []
    for s in seeds:
        agent, env = build_agent_env_fn(s)
        res = eval_agent(agent, env, n_steps=n_steps)
        rewards.append(res["rewards"])
    rewards = np.vstack(rewards)  # shape: [S, T]
    return rewards


def experiment_factory(base_agent_factory, environment_factory, contextual=False, **context_config):
    def experiment_fun(seed):
        rng = np.random.RandomState(seed)
        env = environment_factory(rng)
        if contextual:
            agent = ContextualClusteredBandit(env, rng, base_agent_factory=base_agent_factory, **context_config)
        else:
            agent = base_agent_factory(rng)
        return agent, env
    return experiment_fun


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Mapping

def plot_reward_band(
    group_stats: Mapping[str, np.ndarray],
    users_per_minute: Optional[float] = None,
    quantiles: Tuple[float, float] = (0.25, 0.75),
    ylabel: str = "Reward",
    title: Optional[str] = None,
    legend_ncol: int = 4,
    figsize: Tuple[int, int] = (12, 4),
    linewidth: float = 2.0,
    band_alpha: float = 0.18,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot mean reward with an IQR band for multiple groups.

    Args:
        group_stats: dict like {name: rewards}, where rewards has shape (n_runs, T).
        users_per_minute: If provided, x-axis is time in minutes = step / users_per_minute.
                          If None, x-axis is step number (1..T).
        quantiles: Lower and upper quantiles for the band (default Q1-Q3).
        ylabel: Y-axis label.
        title: Plot title. If None, auto-includes the number of seeds/runs.
        legend_ncol: Number of columns in the legend.
        figsize: Figure size (ignored if an existing `ax` is passed).
        linewidth: Line width for the mean curves.
        band_alpha: Alpha for the quantile band.
        ax: Optional matplotlib Axes to draw on. If None, a new figure/axes is created.

    Returns:
        The matplotlib Axes with the plot.
    """
    if not group_stats:
        raise ValueError("group_stats is empty.")

    # Infer T and check shapes
    first_key = next(iter(group_stats))
    first_arr = np.asarray(group_stats[first_key])
    if first_arr.ndim != 2:
        raise ValueError(f"Rewards for '{first_key}' must be 2D (n_runs, T). Got shape {first_arr.shape}.")

    n_runs, T = first_arr.shape
    for k, arr in group_stats.items():
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != T:
            raise ValueError(f"All groups must have shape (n_runs, {T}). '{k}' has {arr.shape}.")

    # X-axis: time or steps
    steps = np.arange(1, T + 1, dtype=int)
    if users_per_minute is not None and users_per_minute > 0:
        xs = steps / float(users_per_minute)
        xlabel = "time (minutes)"
    else:
        xs = steps
        xlabel = "step"

    # Prepare axes
    created_fig = False
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        created_fig = True

    q_lo, q_hi = quantiles

    # Plot each group
    for name, rewards in group_stats.items():
        rewards = np.asarray(rewards)
        mean = rewards.mean(axis=0)
        vmin = np.quantile(rewards, q=q_lo, axis=0)
        vmax = np.quantile(rewards, q=q_hi, axis=0)

        ax.plot(xs, mean, label=name, linewidth=linewidth)
        ax.fill_between(xs, vmin, vmax, alpha=band_alpha)

    # Labels & cosmetics
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        title = f"Reward (mean line with Q{int(q_lo*100)}–Q{int(q_hi*100)} band) — {n_runs} seeds"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(ncol=legend_ncol)
    if created_fig:
        plt.tight_layout()

    return ax
