"""Multi-Armed Bandit Refresher

  - Think of a casino with many slot machines (“arms”). Each arm pays out a different, unknown average
    reward. A bandit agent repeatedly chooses an arm, observes the reward, and updates its beliefs. The
    challenge is the exploration–exploitation trade‑off: try new arms to learn, or keep playing the best arm
    found so far to earn more now. This setup is the simplest form of reinforcement learning because there
    is only one “state”; the agent still learns sequentially from rewards, but there is no state transition
    model.

  What Each Python File Does

  - N1.py (Basics):
      - Implements three classic non-contextual bandits: ε-greedy, UCB, and Gradient Bandit.
      - Shows how to set up experiments in a stationary environment, run multiple seeds, and interpret
        learning curves.
      - Answers the initial analysis questions (which agent wins, why UCB spikes, etc.).
      - Purpose: teach core algorithms and guarantee they pass the provided tests.

    ε-greedy randomly explores with probability ε and otherwise exploits the arm with the highest estimated mean reward. It keeps a running
    average per arm and balances exploration/exploitation with a simple coin flip.
  - UCB (Upper Confidence Bound) picks the arm with the highest optimistic estimate mean + c * sqrt(log t / pulls). Every arm is tried at least
    once, then the bonus term encourages exploring uncertain arms. It’s deterministic and adapts confidence through counts.
  - Gradient Bandit learns preferences instead of values. It maintains a softmax policy over arms and updates the preferences via a policy-
    gradient rule using the reward baseline. Arms with higher advantage get higher probability, and learning is controlled by a step size α.

  - N2.py (Hyperparameter Optimization):
      - Reuses the same agents in a more realistic recommender environment where click probabilities drift
        over time.
      - Provides grid searches for ε-greedy (ε and step size), UCB (confidence c), and Gradient Bandit
        (learning rate α).
      - Compares the best configurations, checks for non-stationarity (constant step size helps), and
        explains how tuning would work without a simulator.
      - Purpose: demonstrate how to evaluate and tune bandit hyperparameters using averaged reward curves.
  - N3.py (Contextual Bandits):
      - Wraps a base bandit (UCB here) with a context-aware layer that clusters users via k-means and runs
        one non-contextual bandit per cluster.
      - Chooses the number of clusters with a silhouette score sweep.
      - Compares plain UCB vs. contextual UCB to show the benefit of matching users to specialized agents.
      - Purpose: extend the basic bandit idea to contextual settings where user features matter.

  Together, the three scripts walk from foundational bandit algorithms, through practical tuning, to context-
  sensitive variants—all in accessible, well-commented code."""

# %% [markdown] cell 0
# <div class="alert alert-block alert-warning">
# Submit this notebook via Moodle!
# </div>

# %% [code] cell 1
import pytest  # import pytest so we can reuse the provided unit tests
import numpy as np  # numpy provides vectorized math utilities and RNG helpers
import matplotlib.pyplot as plt  # pyplot is used to visualize learning curves later on
from typing import Any, Dict, List  # typing hints improve readability for dictionaries/lists

from environments.base import State, Action  # base dataclasses describing environment states/actions
from environments.normal import MABTestEnvironment  # stationary bandit environment for sanity checks
from bandits import RandomAgent, EpsilonGreedyAgent, UCBAgent, GradientBandit  # agent classes implemented earlier
from utils import experiment_factory, run_multi_seeds  # helper utilities to create/run experiments across seeds
from utils import plot_reward_band  # plotting helper that shows mean reward with min-max envelopes

# %% [markdown] cells 2-4 contain the textual exercise description

# %% [code] cell 6 – epsilon-greedy select

def eg_select(self, s: State) -> int:  # notebook-scope implementation attached to EpsilonGreedyAgent
    """Select an arm via epsilon-greedy with deterministic tie-breaking."""  # document behavior for later reference
    if not isinstance(s, State):  # enforce interface contract explicitly for clearer error messages
        raise TypeError("s must be a State")  # provide informative error if wrong type is supplied
    available_ids = [a.id for a in s.actions]  # extract numeric arm identifiers from the environment state
    for a_id in available_ids:  # iterate over all arms exposed in the current state
        if a_id not in self.Q:  # lazily initialize tracking structures when an arm appears for the first time
            self.Q[a_id] = self.initial_value  # seed the value estimate with the configured optimistic prior
            self.N[a_id] = 0  # initialize the selection count so updates behave correctly
    if float(self.rng.rand()) < self.epsilon:  # sample a uniform random number and compare to epsilon threshold
        return int(self.rng.choice(available_ids))  # explore uniformly over the currently available arms
    best_q = max(self.Q[a] for a in available_ids)  # compute the greedy target value across available actions
    for a_id in available_ids:  # iterate again to find the first arm attaining the maximum
        if self.Q[a_id] == best_q:  # deterministic tie-breaking ensures reproducibility
            return int(a_id)  # return the greedy arm ID as an integer
    raise RuntimeError("No arm selected; available set may be empty")  # safeguard against impossible control flow


# %% [code] cell 7 – epsilon-greedy update

def eg_update(self, arm_id: int, reward: float, s: State) -> None:  # notebook implementation of the update rule
    """Update the value estimate for the provided arm using sample-average or constant step size."""  # docstring for clarity
    if not isinstance(arm_id, int):  # validate that the arm identifier supplied by the caller is an integer
        raise TypeError("arm id must be int")  # raise informative TypeError otherwise
    if not isinstance(reward, float):  # enforce float rewards as per the assignment tests
        raise TypeError("reward id must be float")  # keep the error message aligned with the unit tests
    if not isinstance(s, State):  # ensure the state object originates from the environment API
        raise TypeError("s id must be State")  # again, mimic the test expectations exactly
    available_ids = [a.id for a in s.actions]  # list of arms that were available when the reward was produced
    if (  # guard against logical errors such as calling update before select
        arm_id not in self.Q  # unseen arm globally
        or any(a_id not in self.Q for a_id in available_ids)  # indicates select() was skipped for some actions
        or arm_id not in available_ids  # reward for an arm that is not even in the current state
    ):
        raise ValueError("Unseen arm {arm_id}; call `select` before `update`.")  # message reused from tests
    self.N[arm_id] += 1  # increment the counter for this arm so the sample-average denominator stays correct
    if self.alpha is None:  # if no constant step size is specified we use the running average update
        self.Q[arm_id] += (reward - self.Q[arm_id]) / self.N[arm_id]  # incremental mean formula
    else:  # otherwise rely on an exponential moving average with fixed alpha
        self.Q[arm_id] += self.alpha * (reward - self.Q[arm_id])  # constant step-size update reacts faster to drift
    self.t += 1  # optional global timestep counter that some analyses rely on


# %% [code] cell 8 – attach epsilon-greedy implementation and run tests
EpsilonGreedyAgent.select = eg_select  # monkey patch the notebook-defined select method onto the class
EpsilonGreedyAgent.update = eg_update  # monkey patch the update method likewise
if __name__ == '__main__':  # only run the expensive pytest suite when the script is executed directly
    print("Epsilon Greedy Bandits")  # friendly header for the console
    passed = pytest.main(["--disable-warnings", "-q", "tests/test_bandits_egreedy.py"]) == 0  # run targeted tests
    print("All tests passed!" if passed else "Some tests failed.")  # human-readable summary

# %% [code] cell 11 – UCB select

def ucb_select(self, s: State) -> int:  # notebook override for UCBAgent.select
    """Select the arm with the largest UCB index, forcing unseen arms first."""  # describe approach
    if not isinstance(s, State):  # enforce API type contract
        raise TypeError("s must be a State")  # consistent error messaging
    available_ids = [a.id for a in s.actions]  # fetch the list of candidate arms from the environment
    for a_id in available_ids:  # lazily initialize new arms as in epsilon-greedy
        if a_id not in self.Q:
            self.Q[a_id] = self.initial_value  # set optimistic prior
            self.N[a_id] = 0  # mark as unseen
    for a_id in available_ids:  # forced exploration pass
        if self.N[a_id] == 0:  # if the arm has never been tried
            return int(a_id)  # force-select it before computing UCB indices
    log_term = np.log(max(self.t, 1))  # compute log(t) safely (avoid log(0))
    best_arm = None  # placeholder for the argmax result
    best_idx = None  # placeholder for the highest UCB value seen so far
    for a_id in available_ids:  # iterate through each available arm
        bonus = self.c * np.sqrt(log_term / self.N[a_id])  # exploration bonus term
        idx = self.Q[a_id] + bonus  # full UCB index = exploitation + exploration
        if best_idx is None or idx > best_idx:  # keep the best-scoring arm (deterministic due to iteration order)
            best_idx = idx  # update best score
            best_arm = a_id  # update best arm
    return int(best_arm)  # return the chosen arm ID


# %% [code] cell 12 – UCB update

def ucb_update(self, arm_id: int, reward: float, s: State) -> None:  # override for UCBAgent.update
    """Update the sample-average value estimate and selection counts for the supplied arm."""
    if not isinstance(arm_id, int):
        raise TypeError("arm id must be int")
    if not isinstance(reward, float):
        raise TypeError("reward id must be float")
    if not isinstance(s, State):
        raise TypeError("s id must be State")
    available_ids = [a.id for a in s.actions]
    if (
        arm_id not in self.Q
        or any(a_id not in self.Q for a_id in available_ids)
        or arm_id not in available_ids
    ):
        raise ValueError("Unseen arm {arm_id}; call `select` before `update`.")
    self.t += 1  # increment global timestep so future log(t) calls see the updated count
    self.N[arm_id] += 1  # bump the selection count for this arm
    self.Q[arm_id] += (reward - self.Q[arm_id]) / self.N[arm_id]  # incremental mean update


# %% [code] cell 13 – attach UCB implementation and run tests
UCBAgent.select = ucb_select  # patch select method
UCBAgent.update = ucb_update  # patch update method
if __name__ == '__main__':  # only run tests on direct execution
    print("UCB")
    passed = pytest.main(["--disable-warnings", "-q", "tests/test_bandits_ucb.py"]) == 0
    print("All tests passed!" if passed else "Some tests failed.")

# %% [code] cell 16 – gradient bandit select

def gb_select(self, s: State) -> int:  # gradient bandit selection logic
    """Sample an arm from the softmax distribution over preference scores."""
    if not isinstance(s, State):
        raise TypeError("s must be a State")
    available_ids = [a.id for a in s.actions]
    for a_id in available_ids:
        if a_id not in self.H:  # lazily initialize new arms with zero preference
            self.H[a_id] = 0.0
    probs = self._softmax(available_ids)  # compute softmax probabilities only over available arms
    p_vec = [probs[a_id] for a_id in available_ids]  # convert dict to probability vector matching IDs order
    return int(self.rng.choice(available_ids, p=p_vec))  # sample from the categorical distribution


# %% [code] cell 17 – gradient bandit update

def gb_update(self, arm_id: int, reward: float, s: State) -> None:  # notebook implementation of the gradient update
    """Apply the policy-gradient preference update using a running reward baseline."""
    if not isinstance(arm_id, int):
        raise TypeError("arm id must be int")
    if not isinstance(reward, float):
        raise TypeError("reward id must be float")
    if not isinstance(s, State):
        raise TypeError("s id must be State")
    available_ids = [a.id for a in s.actions]
    if (
        arm_id not in self.H
        or any(a_id not in self.H for a_id in available_ids)
        or arm_id not in available_ids
    ):
        raise ValueError("Unseen arm {arm_id}; call `select` before `update`.")
    probs = self._softmax(available_ids)  # recompute current action probabilities
    baseline = self.avg_reward  # use running average reward as baseline for variance reduction
    for a_id in available_ids:  # update only the arms present in this state
        indicator = 1.0 if a_id == arm_id else 0.0  # one-hot indicator for the chosen arm
        self.H[a_id] += self.alpha * (reward - baseline) * (indicator - probs[a_id])  # preference gradient step
    self.t += 1  # increment counter for averaging
    self.avg_reward += (reward - self.avg_reward) / self.t  # incremental mean update for the baseline


# %% [code] cell 18 – attach gradient bandit implementation and run tests
GradientBandit.select = gb_select
GradientBandit.update = gb_update
if __name__ == '__main__':
    print("Gradient Bandits")
    passed = pytest.main(["--disable-warnings", "-q", "tests/test_bandits_gb.py"]) == 0
    print("All tests passed!" if passed else "Some tests failed.")

# %% [code] cell 20 – visual sanity check experiment
if __name__ == '__main__':
    SEEDS = np.arange(100)  # number of random seeds to average over
    N_STEPS = 500  # number of steps per experiment
    groups = {}  # container for experiment builders per agent
    groups["random"] = experiment_factory(  # random baseline configuration
        lambda rng: RandomAgent(rng=rng),
        lambda rng: MABTestEnvironment(rng=rng)
    )
    epsilon = 0.1  # exploration rate for epsilon-greedy in the sanity check
    groups[f"epsilon-greedy, e={epsilon}"] = experiment_factory(
        lambda rng: EpsilonGreedyAgent(rng=rng, epsilon=epsilon),
        lambda rng: MABTestEnvironment(rng=rng)
    )
    c = 1.5  # UCB confidence parameter for visualization
    groups[f"UCB, c={c}"] = experiment_factory(
        lambda rng: UCBAgent(c=c),
        lambda rng: MABTestEnvironment(rng=rng)
    )
    alpha = 0.1  # gradient bandit learning rate for visualization
    groups[f"GB, alpha={alpha}"] = experiment_factory(
        lambda rng: GradientBandit(rng, alpha=alpha),
        lambda rng: MABTestEnvironment(rng=rng)
    )
    group_stats: Dict[str, Dict[str, Any]] = {}  # dictionary storing rolled-out reward traces
    for name, builder in groups.items():  # iterate through each experiment setup
        group_stats[name] = run_multi_seeds(builder, SEEDS, n_steps=N_STEPS)  # run and store the stats
    plot_reward_band(group_stats)  # visualize mean/min-max reward curves across time

# %% [code] cell 22 – question responses stored as simple variables
best = 3  # answer index for question 1.2.1 (UCB performed best)
assert type(best) is int  # ensure the grader sees an integer
less_prominent = True  # boolean answer for question 1.2.2 (spike less prominent with smaller c)
assert type(less_prominent) is bool  # enforce type correctness for grading


"""
The N1 plot compares the average reward over time for four agents on the stationary test environment (random, ε-greedy with ε=0.1, UCB with
  c=1.5, and a gradient bandit with α=0.1). Key takeaways:

  - Random stays flat at the lowest reward since it never learns.
  - ε-greedy and the gradient bandit steadily improve because they keep exploring but gradually favor the better arm; their curves rise above
    random and then level off.
  - UCB shows a sharp spike early on because the optimism bonus forces it to try every arm quickly, then it settles to the highest plateau once
    it identifies the best arm.
  Overall, UCB converges fastest and highest in this stationary setup, followed by ε-greedy and the gradient bandit, with random far behind.

  ======================
   In this context, “Random” refers to the baseline bandit agent that ignores all feedback: at every step it picks uniformly among the currently
  available arms, never updating any estimates. It’s a useful lower bound—if another agent doesn’t beat Random, it’s failing to learn anything
  meaningful.

"""