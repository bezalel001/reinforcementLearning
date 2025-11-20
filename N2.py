# """Detailed, line-by-line commented reconstruction of N2_hyperparameter_optimization.ipynb."""
#
# # %% [markdown] cell 0
# # <div class="alert alert-block alert-warning">
# # Submit this notebook via Moodle!
# # </div>
#
# # %% [code] cell 1
# try:
#     import pytest  # pytest is optional here but useful if we want to rerun tests from this notebook
# except ImportError:
#     pytest = None  # fall back gracefully when pytest is not installed in the runtime
# import matplotlib.pyplot as plt  # plotting library used for visual comparisons of reward curves
# import numpy as np  # numpy powers random seeding and numerical aggregates
# from typing import Any, Dict, List  # typing aids readability when dealing with nested dictionaries
#
# from bandits import RandomAgent, EpsilonGreedyAgent, UCBAgent, GradientBandit  # re-use the previously implemented agents
# from environments.recsys import RecsysEnvironment  # contextual recommendation environment used throughout N2
# from utils import experiment_factory, run_multi_seeds  # helper utilities for spawning agents/envs across seeds
# from utils import plot_reward_band  # renders mean/min-max reward envelopes for experiment groups
#
# # %% [code] cell 3 – import notebook-defined helpers (fallback included)
# try:
#     from ipynb.fs.full.N1_basic import (
#         eg_select, eg_update, ucb_select, ucb_update, gb_select, gb_update
#     )  # prefer importing the authoritative implementations from N1
# except Exception:  # pylint: disable=broad-except  # fallback so the notebook works even without ipynb import machinery
#     from environments.base import State  # need the State class for type checks inside the fallback definitions
#     import numpy as _np  # local alias to avoid clobbering the main numpy import
#
#     def eg_select(self, s: State) -> int:
#         if not isinstance(s, State):
#             raise TypeError("s must be a State")
#         available_ids = [a.id for a in s.actions]
#         for a_id in available_ids:
#             if a_id not in self.Q:
#                 self.Q[a_id] = self.initial_value
#                 self.N[a_id] = 0
#         if float(self.rng.rand()) < self.epsilon:
#             return int(self.rng.choice(available_ids))
#         best_q = max(self.Q[a] for a in available_ids)
#         for a_id in available_ids:
#             if self.Q[a_id] == best_q:
#                 return int(a_id)
#         raise RuntimeError("No arm chosen")
#
#     def eg_update(self, arm_id: int, reward: float, s: State) -> None:
#         if not isinstance(arm_id, int):
#             raise TypeError("arm id must be int")
#         if not isinstance(reward, float):
#             raise TypeError("reward id must be float")
#         if not isinstance(s, State):
#             raise TypeError("s id must be State")
#         available_ids = [a.id for a in s.actions]
#         if (
#             arm_id not in self.Q
#             or any(a_id not in self.Q for a_id in available_ids)
#             or arm_id not in available_ids
#         ):
#             raise ValueError("Unseen arm {arm_id}; call `select` before `update`.")
#         self.N[arm_id] += 1
#         if self.alpha is None:
#             self.Q[arm_id] += (reward - self.Q[arm_id]) / self.N[arm_id]
#         else:
#             self.Q[arm_id] += self.alpha * (reward - self.Q[arm_id])
#         self.t += 1
#
#     def ucb_select(self, s: State) -> int:
#         if not isinstance(s, State):
#             raise TypeError("s must be a State")
#         available_ids = [a.id for a in s.actions]
#         for a_id in available_ids:
#             if a_id not in self.Q:
#                 self.Q[a_id] = self.initial_value
#                 self.N[a_id] = 0
#         for a_id in available_ids:
#             if self.N[a_id] == 0:
#                 return int(a_id)
#         log_term = _np.log(max(self.t, 1))
#         best_arm, best_idx = None, None
#         for a_id in available_ids:
#             bonus = self.c * _np.sqrt(log_term / self.N[a_id])
#             idx = self.Q[a_id] + bonus
#             if best_idx is None or idx > best_idx:
#                 best_idx = idx
#                 best_arm = a_id
#         return int(best_arm)
#
#     def ucb_update(self, arm_id: int, reward: float, s: State) -> None:
#         if not isinstance(arm_id, int):
#             raise TypeError("arm id must be int")
#         if not isinstance(reward, float):
#             raise TypeError("reward id must be float")
#         if not isinstance(s, State):
#             raise TypeError("s id must be State")
#         available_ids = [a.id for a in s.actions]
#         if (
#             arm_id not in self.Q
#             or any(a_id not in self.Q for a_id in available_ids)
#             or arm_id not in available_ids
#         ):
#             raise ValueError("Unseen arm {arm_id}; call `select` before `update`.")
#         self.t += 1
#         self.N[arm_id] += 1
#         self.Q[arm_id] += (reward - self.Q[arm_id]) / self.N[arm_id]
#
#     def gb_select(self, s: State) -> int:
#         if not isinstance(s, State):
#             raise TypeError("s must be a State")
#         available_ids = [a.id for a in s.actions]
#         for a_id in available_ids:
#             if a_id not in self.H:
#                 self.H[a_id] = 0.0
#         probs = self._softmax(available_ids)
#         p_vec = [probs[a_id] for a_id in available_ids]
#         return int(self.rng.choice(available_ids, p=p_vec))
#
#     def gb_update(self, arm_id: int, reward: float, s: State) -> None:
#         if not isinstance(arm_id, int):
#             raise TypeError("arm id must be int")
#         if not isinstance(reward, float):
#             raise TypeError("reward id must be float")
#         if not isinstance(s, State):
#             raise TypeError("s id must be State")
#         available_ids = [a.id for a in s.actions]
#         if (
#             arm_id not in self.H
#             or any(a_id not in self.H for a_id in available_ids)
#             or arm_id not in available_ids
#         ):
#             raise ValueError("Unseen arm {arm_id}; call `select` before `update`.")
#         probs = self._softmax(available_ids)
#         baseline = self.avg_reward
#         for a_id in available_ids:
#             indicator = 1.0 if a_id == arm_id else 0.0
#             self.H[a_id] += self.alpha * (reward - baseline) * (indicator - probs[a_id])
#         self.t += 1
#         self.avg_reward += (reward - self.avg_reward) / self.t
#
# EpsilonGreedyAgent.select = eg_select  # attach imported/fallback methods to the actual agent classes
# EpsilonGreedyAgent.update = eg_update
# UCBAgent.select = ucb_select
# UCBAgent.update = ucb_update
# GradientBandit.select = gb_select
# GradientBandit.update = gb_update
#
# RUN_OPTIMIZATION = True  # notebook flag controlling whether the sweeps execute (set False during autograding)
#
# # %% [code] cell 10 – epsilon-greedy hyperparameter sweep
#
# def your_epsilon_greedy_optimization() -> Dict[str, Any]:  # wrapper keeps optimization isolated from autograder
#     """Coarse grid search over epsilon and optional constant step size."""
#     SEEDS = np.arange(30)  # evaluate each configuration over 30 different RNG seeds
#     N_STEPS = 2000  # run for roughly one simulated hour (30 users/min * 2000 steps ~ long horizon)
#     USERS_PER_MINUTE = 30  # throughput parameter for the environment
#     candidates = [  # predefined grid of (epsilon, alpha) combinations to compare
#         (0.05, None),
#         (0.1, None),
#         (0.05, 0.1),
#         (0.1, 0.1),
#         (0.2, 0.05),
#     ]
#     group_stats: Dict[str, Any] = {}  # store the reward traces for plotting
#     rand_rewards = run_multi_seeds(  # compute the random baseline once for reference
#         experiment_factory(
#             lambda rng: RandomAgent(rng=rng),  # random agent ignores context/history
#             lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),  # identical environment for fairness
#         ),
#         SEEDS,
#         n_steps=N_STEPS,
#     )
#     group_stats["random"] = rand_rewards  # record baseline stats for visualization
#     best_mean = -np.inf  # sentinel best score
#     best_params: Dict[str, Any] = {}  # dictionary that will store epsilon/alpha for the best run
#     for eps, alpha in candidates:  # iterate over each configuration in the coarse grid
#         label = f"$\\epsilon$–greedy (e={eps}, alpha={alpha})"  # label used in the plot legend
#         rewards = run_multi_seeds(
#             experiment_factory(
#                 lambda rng, e=eps, a=alpha: EpsilonGreedyAgent(rng=rng, epsilon=e, alpha=a),  # instantiate tuned agent
#                 lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),  # same environment each time
#             ),
#             SEEDS,
#             n_steps=N_STEPS,
#         )
#         group_stats[label] = rewards  # store reward traces for plotting
#         mean_reward = rewards.mean()  # aggregate mean reward across seeds/time
#         if mean_reward > best_mean:  # keep best-performing parameter set
#             best_mean = mean_reward
#             best_params = {"epsilon": eps, "alpha": alpha, "label": label}
#     plot_reward_band(group_stats, users_per_minute=USERS_PER_MINUTE)  # visualize learning curves for all configs
#     if best_params:  # print a friendly summary for the top performer
#         print(f"Best epsilon-greedy config: {best_params['label']} (mean reward {best_mean:.4f})")
#     return {"best_params": best_params, "best_mean": best_mean}  # return details for later reference
#
#
# if RUN_OPTIMIZATION:  # guard ensures autograder can skip heavy sweeps
#     your_epsilon_greedy_optimization()
#
# # %% [code] cell 14 – UCB hyperparameter sweep
#
# def your_UCB_optimization() -> Dict[str, Any]:  # wrapper for the UCB-specific grid search
#     """Grid search over the confidence parameter c."""
#     SEEDS = np.arange(30)
#     N_STEPS = 2000
#     USERS_PER_MINUTE = 30
#     c_values = [0.5, 1.0, 1.5, 2.0]  # the candidate exploration bonuses we will evaluate
#     groups = {
#         "random": experiment_factory(  # random baseline (same as before)
#             lambda rng: RandomAgent(rng=rng),
#             lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
#         )
#     }
#     best = {"name": None, "mean": -np.inf, "c": None}  # track the best c value encountered so far
#     for c in c_values:  # build experiment factories for each c value
#         label = f"UCB, c={c}"
#         groups[label] = experiment_factory(
#             lambda rng, cc=c: UCBAgent(c=cc),
#             lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
#         )
#     group_stats = {}  # rolling storage for actual evaluation outputs
#     for name, builder in groups.items():  # run each experiment
#         rewards = run_multi_seeds(builder, SEEDS, n_steps=N_STEPS)
#         group_stats[name] = rewards
#         mean_reward = rewards.mean()
#         if name.startswith("UCB") and mean_reward > best["mean"]:  # update best when this configuration wins
#             best.update({"name": name, "mean": mean_reward, "c": float(name.split('=')[1])})
#     plot_reward_band(group_stats, users_per_minute=USERS_PER_MINUTE)
#     print(f"Best UCB config: {best['name']} (mean reward {best['mean']:.4f})")
#     return best
#
#
# if RUN_OPTIMIZATION:
#     your_UCB_optimization()
#
# # %% [code] cell 18 – gradient bandit sweep
#
# def your_gradient_bandit_optimization() -> Dict[str, Any]:  # wrapper for alpha search
#     """Search over the preference learning rate alpha."""
#     SEEDS = np.arange(30)
#     N_STEPS = 2000
#     USERS_PER_MINUTE = 30
#     alphas = [0.02, 0.05, 0.1, 0.2]  # candidate learning rates
#     groups = {
#         "random": experiment_factory(
#             lambda rng: RandomAgent(rng=rng),
#             lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
#         )
#     }
#     best = {"name": None, "mean": -np.inf, "alpha": None}
#     for alpha in alphas:
#         label = f"GB, alpha={alpha}"
#         groups[label] = experiment_factory(
#             lambda rng, a=alpha: GradientBandit(rng=rng, alpha=a),
#             lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
#         )
#     group_stats = {}
#     for name, builder in groups.items():
#         rewards = run_multi_seeds(builder, SEEDS, n_steps=N_STEPS)
#         group_stats[name] = rewards
#         mean_reward = rewards.mean()
#         if name.startswith("GB") and mean_reward > best["mean"]:
#             best.update({"name": name, "mean": mean_reward, "alpha": float(name.split('=')[1])})
#     plot_reward_band(group_stats, users_per_minute=USERS_PER_MINUTE)
#     print(f"Best Gradient Bandit config: {best['name']} (mean reward {best['mean']:.4f})")
#     return best
#
#
# if RUN_OPTIMIZATION:
#     your_gradient_bandit_optimization()
#
# # %% [code] cell 22 – compare best configs head-to-head
# best_eps_params = {"epsilon": 0.05, "alpha": 0.1}  # winning epsilon-greedy parameters from the sweep
# best_ucb_c = 1.0  # best c from the UCB sweep
# best_gb_alpha = 0.05  # best alpha from the gradient bandit sweep
# SEEDS = np.arange(30)  # reuse same evaluation depth for comparability
# N_STEPS = 2000
# USERS_PER_MINUTE = 30
# groups: Dict[str, Any] = {}  # dictionary mapping legend labels to experiment factories
# groups[f"$\\epsilon$–greedy (e={best_eps_params['epsilon']}, alpha={best_eps_params['alpha']})"] = experiment_factory(
#     lambda rng: EpsilonGreedyAgent(
#         rng=rng,
#         epsilon=best_eps_params['epsilon'],
#         alpha=best_eps_params['alpha']
#     ),
#     lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
# )
# groups[f"UCB (c={best_ucb_c})"] = experiment_factory(
#     lambda rng: UCBAgent(c=best_ucb_c),
#     lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
# )
# groups[f"GB (alpha={best_gb_alpha})"] = experiment_factory(
#     lambda rng: GradientBandit(rng=rng, alpha=best_gb_alpha),
#     lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
# )
# groups["random"] = experiment_factory(
#     lambda rng: RandomAgent(rng=rng),
#     lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
# )
# if RUN_OPTIMIZATION:
#     group_stats = {  # evaluate each tuned configuration plus random baseline
#         name: run_multi_seeds(builder, SEEDS, n_steps=N_STEPS)
#         for name, builder in groups.items()
#     }
#     plot_reward_band(group_stats, users_per_minute=USERS_PER_MINUTE)
#
# # %% [code] cell 24 – confirm non-stationarity via constant step size
#
# def your_non_stationary_optimization() -> Dict[str, float]:
#     """Compare sample-average vs. constant step size for epsilon-greedy."""
#     SEEDS = np.arange(30)
#     N_STEPS = 2000
#     USERS_PER_MINUTE = 30
#     groups = {
#         "epsilon=0.05, sample-avg": experiment_factory(
#             lambda rng: EpsilonGreedyAgent(rng=rng, epsilon=0.05, alpha=None),
#             lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
#         ),
#         "epsilon=0.05, alpha=0.1": experiment_factory(
#             lambda rng: EpsilonGreedyAgent(rng=rng, epsilon=0.05, alpha=0.1),
#             lambda rng: RecsysEnvironment(rng=rng, users_per_minute=USERS_PER_MINUTE),
#         ),
#     }
#     group_stats = {
#         name: run_multi_seeds(builder, SEEDS, n_steps=N_STEPS)
#         for name, builder in groups.items()
#     }
#     plot_reward_band(group_stats, users_per_minute=USERS_PER_MINUTE)
#     return {name: stats.mean() for name, stats in group_stats.items()}
#
#
# if RUN_OPTIMIZATION:
#     your_non_stationary_optimization()
#
# # %% [markdown] cells 25-27 capture textual answers and are summarized in N2.md
#
# """
# The N2 plots show several comparisons, but the core idea is this: after sweeping various hyperparameters,
#   each agent (ε-greedy, UCB, Gradient Bandit) is rerun with its best setting alongside the random baseline in
#   the News Lab environment. The curves are mean rewards over 2000 steps (≈1 hour of simulated traffic), and
#   the shaded bands show variability across 30 seeds.
#
#   - All tuned agents clearly outperform Random, confirming they leverage feedback.
#   - ε-greedy with a small ε and constant step size stabilizes quickly but moves more slowly than UCB early on
#     because it still spends a fixed fraction of time exploring.
#   - UCB with c≈1.0 exhibits a modest initial spike (from the confidence bonus) and then keeps the highest
#     long-term average reward, showing it balances exploration and exploitation effectively in this setting.
#   - Gradient Bandit with α≈0.05 rises more gradually; it reacts well to changing articles but remains
#     slightly below the tuned UCB curve, indicating it benefits from careful step-size tuning but still
#     carries more variance.
#
#   There’s also a secondary plot comparing ε-greedy with and without a constant step size, and there the
#   constant-step curve stays higher later in the run, illustrating that the environment is non-stationary
#   (rewards drift), so recent data should be weighted more.
# """
# =====================================================================
# - Candidates: test c = {0.5, 1.0, 1.5, 2.0}; c scales the optimism bonus c * sqrt(log t / N[a]). Higher c explores more aggressively.
#   - Setup: for each candidate, build an experiment via experiment_factory, run the agent in RecsysEnvironment with 30 seeds and 2 000 steps
#     (about an hour of simulated traffic), and also keep a random baseline for reference.
#   - Evaluation: collect reward traces via run_multi_seeds, compute the mean reward per configuration, and store all curves in group_stats.
#   - Visualization: call plot_reward_band(group_stats) to compare the tuned curves; this shows how each c affects early spikes and long-run
#     averages.
#   - Result: track the best mean reward and print something like “Best UCB config: UCB, c=1.0 (mean reward …)”. In the provided run, c ≈ 1.0 gave
#     the most stable high reward—lower c under-explores, higher c spends too long on stale arms.

#   So 2.1.3 systematically sweeps c, plots the learning behavior, and picks the value that maximizes average reward within the coarse grid.

# =========================================================================================
# Task 2.1.4 – Hyperparameter Optimization for the Gradient Bandit

#   Objective: tune the learning rate α of the Gradient Bandit in the News Lab environment.

#   Implementation (function your_gradient_bandit_optimization()):

#   - Candidates: the grid [0.02, 0.05, 0.1, 0.2]. α controls how strongly the preferences H[a] respond to each
#     reward; smaller α = slower, steadier updates, larger α = faster but noisier adaptation.
#   - Experiment setup: same as earlier tasks—run each configuration for 2 000 steps with 30 random seeds and a
#     user rate of 30 per minute. A random-agent baseline is included for comparison.
#   - Execution: for every α, build an experiment_factory that creates GradientBandit(rng, alpha=α) and the
#     RecsysEnvironment. Use run_multi_seeds to collect reward traces, store them in group_stats, and compute
#     the mean reward.
#   - Visualization: feed all traces (including the random baseline) to plot_reward_band to inspect how
#     different α values affect convergence and variance.
#   - Selection: track the highest mean reward across the candidates. In the notebook run, α ≈ 0.05 provided
#     the best balance—small enough to keep variance down, large enough to adapt to changing article appeal.

#   In short, Task 2.1.4 runs a grid search over the gradient-bandit learning rate, logs and plots each result,
#   and reports the α that maximizes average reward so it can be used in later comparisons.

"""Task 2.3 – Non-stationarity of the Environment

  - Question: Is the News Lab environment non-stationary? In other words, do reward probabilities drift so
    that recent observations should be weighted more heavily?
  - Experiment (your_non_stationary_optimization): run two ε-greedy agents with the same ε (0.05) but
    different update rules—one uses the standard sample-average update (alpha=None), the other uses a
    constant step size (alpha=0.1). Because a constant step size discounts older data, it tends to perform
    better in non-stationary settings.
  - Result: the constant-step-size version achieves a slightly higher mean reward curve, indicating that
    the environment’s click probabilities change over time (e.g., new articles appear, old ones decay). The
    notebook’s markdown notes this observation explicitly.

  Task 2.4 – Hyperparameter Optimization in Real Life

  - Question: Without a simulator, how would you tune bandit hyperparameters?
  - Answer (markdown): rely on historical interaction logs and off-policy evaluation techniques (e.g.,
    Inverse Propensity Scoring or Doubly Robust estimators) to screen candidate settings offline, then
    validate promising hyperparameters using carefully throttled online A/B tests on a small portion of live
    traffic. This balances learning with user-experience safety when no simulator is available."""