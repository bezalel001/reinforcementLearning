"""Detailed, line-by-line commented reconstruction of N3_contextual.ipynb."""

# %% [markdown] cell 0
# <div class="alert alert-block alert-warning">
# Submit this notebook via Moodle!
# </div>

# %% [code] cell 1
import pytest  # needed to rerun the dedicated contextual bandit tests
import numpy as np  # provides RNG/state handling for experiments
import matplotlib.pyplot as plt  # used to plot silhouette scores and reward curves

from bandits import UCBAgent, ContextualClusteredBandit  # base UCB agent and contextual wrapper under test
from environments.base import State, Action  # state/action dataclasses referenced in helper code
from environments.recsys import RecsysEnvironment  # contextual news environment for experiments

from utils import experiment_factory, run_multi_seeds  # helpers to instantiate/run multi-seed experiments
from utils import plot_reward_band  # helper visualization function

from typing import Any, Dict, List  # typing hints used in the experiment code

# %% [code] cell 2 – import the tested UCB implementations from notebook 1
from ipynb.fs.full.N1_basic import ucb_select, ucb_update  # reuse the validated N1 logic for UCB

UCBAgent.select = ucb_select  # attach imported selection rule to the UCBAgent class
UCBAgent.update = ucb_update  # attach the update rule as well

# %% [code] cell 5 – notebook implementation of ContextualClusteredBandit.select

def select(self, s: State) -> int:  # override for ContextualClusteredBandit.select
    """Assign the user to a cluster and delegate to the respective base agent."""  # summary of approach
    if s.context is None:  # contextual bandit needs user features to route to a cluster
        raise ValueError("User must be provided to select().")
    cluster = self._fast_predict(s.context)  # map the context vector to the nearest k-means center
    self._last_cluster = cluster  # remember the cluster so update() can fall back if needed
    return int(self.agents[cluster].select(s))  # forward the selection call to the cluster-specific agent


# %% [code] cell 6 – notebook implementation of ContextualClusteredBandit.update

def update(self, arm_id: int, reward: float, s: State) -> None:  # override for ContextualClusteredBandit.update
    """Route the reward signal to the same cluster-specific agent used during select()."""
    if s.context is None:  # contextual data required to recompute routing
        raise ValueError("User must be provided to select().")
    cluster = self._fast_predict(s.context)  # identify which sub-agent handled this user
    self._last_cluster = cluster  # update cached cluster assignment (for consistency across calls)
    self.agents[cluster].update(arm_id, reward, s)  # delegate update to the matching non-contextual agent


ContextualClusteredBandit.select = select  # monkey patch notebook implementation onto the class
ContextualClusteredBandit.update = update
print("Clustered Contextual Bandits")  # header for the console
result = pytest.main(["--disable-warnings", "-q", "tests/test_contextual.py"])  # run contextual-specific tests
print("All tests passed!" if result == 0 else "Some tests failed.")  # summarize outcome

# %% [code] cell 9 – determine number of clusters via silhouette score
from sklearn.cluster import KMeans  # clustering algorithm used to partition contexts
from sklearn.metrics import silhouette_score  # evaluation metric for the cluster quality

N_USERS, RNG = 10_000, np.random.RandomState(19241)  # number of sampled contexts and RNG seed for reproducibility
env = RecsysEnvironment(RNG)  # instantiate the recsys environment once for sampling
X = env.sample_contexts(samples=N_USERS)  # dataset used to fit KMeans
X_test = env.sample_contexts(samples=N_USERS)  # unused but kept for parity with the notebook (could be for validation)
K_RANGE = list(range(2, 11))  # evaluate cluster counts from 2 through 10
cluster_scores: List[float] = []  # container for silhouette scores so we can plot them later
for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RNG.randint(1e6))  # fit KMeans with multiple random restarts
    labels = kmeans.fit_predict(X)  # obtain cluster assignments for each sampled context
    score = silhouette_score(X, labels)  # compute how well separated/compact the clusters are
    cluster_scores.append(score)  # store the metric for visualization/argmax
plt.plot(K_RANGE, cluster_scores)  # visualize silhouette score vs. cluster count
plt.xlabel("#clusters")  # label x-axis for clarity
plt.ylabel("Silhouette Score")  # label y-axis
plt.title("Silhouette per cluster count")  # provide descriptive title
plt.show()  # display the figure inline when running interactively
N_CLUSTERS = int(K_RANGE[int(np.argmax(cluster_scores))])  # select the k with the highest silhouette score
assert type(N_CLUSTERS) is int  # ensure compatibility with downstream code/autograder expectations

# %% [code] cell 13 – compare contextual vs non-contextual UCB
SEEDS = np.arange(200)  # large number of seeds to smooth the comparison
N_STEPS = int(30 * 60 * 0.33)  # simulate roughly 20 minutes of user traffic (~600 steps)
groups: Dict[str, Any] = {}  # dictionary storing experiment factories keyed by legend label
c = 1.5  # confidence parameter shared between contextual and non-contextual UCB
groups[f"UCB, c={c}"] = experiment_factory(  # plain UCB baseline without context clustering
    lambda rng: UCBAgent(c=c),
    lambda rng: RecsysEnvironment(rng=rng)
)
groups[f"contextual-UCB, c={c}"] = experiment_factory(  # contextualized variant using the wrapper
    lambda rng: UCBAgent(c=c),
    lambda rng: RecsysEnvironment(rng=rng),
    contextual=True,  # flag instructs experiment_factory to wrap with ContextualClusteredBandit
    n_clusters=N_CLUSTERS  # supply the number of states determined via silhouette
)
group_stats: Dict[str, Dict[str, Any]] = {}  # hold the aggregated reward traces per experiment
for name, builder in groups.items():
    group_stats[name] = run_multi_seeds(builder, SEEDS, n_steps=N_STEPS)  # roll out each experiment over all seeds
plot_reward_band(group_stats, users_per_minute=30)  # visualize mean/min-max reward; x-axis corresponds to minutes

CONTEXTUAL_IS_BETTER = True  # final answer for the conceptual question (observed improvement with contexts)
assert type(CONTEXTUAL_IS_BETTER) is bool  # guard so the grader receives a boolean value

"""
The N3 plot compares two curves:

  - UCB, c=1.5: a single non-contextual UCB agent that treats every user the same.
  - contextual-UCB, c=1.5: the same UCB base logic wrapped in ContextualClusteredBandit, which first clusters
    users via k-means and runs one UCB instance per cluster.

  Both curves show average reward across ~20 minutes (∼600 steps) with 200 seeds, shaded by min–max
  envelopes. Early on they behave similarly because the contextual wrapper still needs to gather data for
  each cluster. After a short warm-up, the contextual line climbs above the baseline and stays there: each
  cluster-specific UCB specializes on the arms that work best for that user segment, so once the clustering
  stabilizes it yields consistently higher rewards. The gap between the two lines—small at first, then
  widening—illustrates the benefit of exploiting user context when the environment truly has heterogeneous
  preferences.

"""
