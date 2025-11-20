import numpy as np
import pytest

sklearn_cluster = pytest.importorskip("sklearn.cluster")  # skip these tests if scikit-learn isn't installed

from environments.base import State, Action
from bandits import ContextualClusteredBandit, BanditAgent

# -----------------------
# Test helpers & fixtures
# -----------------------

class DummyAgent(BanditAgent):
    """Minimal agent that records select/update calls and returns a fixed arm id."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id            # helpful to identify which cluster's agent got called
        self.select_calls = []              # list[State]
        self.update_calls = []              # list[tuple[arm_id, reward, State]]
        self.fixed_arm = 1000 + agent_id    # distinct arm per cluster

    def select(self, s: State) -> int:
        self.select_calls.append(s)
        return self.fixed_arm

    def update(self, arm_id: int, reward: float, s: State) -> None:
        self.update_calls.append((arm_id, reward, s))


class MockEnv:
    """
    Minimal environment with the sampling API expected by ContextualClusteredBandit.
    Produces two clearly separated blobs centered near (0,0) and (10,0) so KMeans is stable.
    """
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    def sample_contexts(self, samples: int = 1000) -> np.ndarray:
        n0 = samples // 2
        n1 = samples - n0
        blob0 = self.rng.normal(loc=[0.0, 0.0], scale=0.3, size=(n0, 2))
        blob1 = self.rng.normal(loc=[10.0, 0.0], scale=0.3, size=(n1, 2))
        return np.vstack([blob0, blob1])


@pytest.fixture
def rng():
    return np.random.RandomState(123)


@pytest.fixture
def env(rng):
    return MockEnv(rng)


def make_factory(created):
    """
    Returns a base_agent_factory that creates DummyAgents with incremental IDs (cluster index order).
    Also collects them into `created` dict for external inspection if needed.
    """
    def factory(_rng):
        agent_id = len(created)  # 0, 1, 2, ...
        ag = DummyAgent(agent_id=agent_id)
        created[agent_id] = ag
        return ag
    return factory


# -------------
# The tests
# -------------

def test_contextual_clustered_bandit_select_routes_to_correct_cluster(env, rng):
    """
    select(s) should compute the nearest cluster for s.context and delegate to that cluster's agent.
    We verify by:
      - calling select on two contexts near different centers,
      - checking the returned arm matches the fixed arm of the predicted cluster's DummyAgent,
      - ensuring only the expected agent's select() recorded a call for each context.
    """
    created = {}
    factory = make_factory(created)

    bandit = ContextualClusteredBandit(
        env=env,
        random_state=rng,
        base_agent_factory=factory,
        n_clusters=2,
        n_sample_contexts=600,
    )

    # Two clearly separated query contexts
    s_left  = State(actions=[Action(1), Action(2)], context=np.array([0.1, -0.2]), steps=0)
    s_right = State(actions=[Action(3), Action(4)], context=np.array([9.8,  0.2]), steps=0)

    # Predict clusters using the bandit's own fast path (ensures labels are consistent with this instance)
    left_cluster  = bandit._fast_predict(s_left.context)
    right_cluster = bandit._fast_predict(s_right.context)

    # Sanity: in this setup they should differ; if not, raise a helpful failure
    assert left_cluster != right_cluster, (
        f"Expected distinct clusters for left/right contexts, got {left_cluster} and {right_cluster}."
    )

    # Calls to select should route to the predicted cluster's agent and return its fixed arm id
    arm_left = bandit.select(s_left)
    assert arm_left == created[left_cluster].fixed_arm, (
        f"Left context should route to agent {left_cluster} returning arm {created[left_cluster].fixed_arm}, "
        f"got {arm_left}"
    )

    arm_right = bandit.select(s_right)
    assert arm_right == created[right_cluster].fixed_arm, (
        f"Right context should route to agent {right_cluster} returning arm {created[right_cluster].fixed_arm}, "
        f"got {arm_right}"
    )

    # Verify only the expected agents saw select() for each context
    for cid, ag in created.items():
        expected_left_calls  = 1 if cid == left_cluster else 0
        expected_right_calls = 1 if cid == right_cluster else 0
        # total calls so far for each agent:
        expected_total = expected_left_calls + expected_right_calls
        assert len(ag.select_calls) == expected_total, (
            f"Agent {cid} select_calls mismatch. Expected {expected_total}, got {len(ag.select_calls)}"
        )
        # if it did receive the left/right call, the stored state should match
        if expected_left_calls:
            assert ag.select_calls[0] is created[left_cluster].select_calls[0], (
                "Recorded left select call does not match the expected state object."
            )


def test_contextual_clustered_bandit_update_routes_to_same_cluster(env, rng):
    """
    update(arm_id, reward, s) should route to the agent for the cluster inferred from s.context.
    We:
      - select for a context to establish the path,
      - call update with that context,
      - confirm only the correct agent recorded the update (with the same arm/reward/state).
    """
    created = {}
    factory = make_factory(created)

    bandit = ContextualClusteredBandit(
        env=env,
        random_state=rng,
        base_agent_factory=factory,
        n_clusters=2,
        n_sample_contexts=600,
    )

    s = State(actions=[Action(10), Action(20)], context=np.array([0.05, 0.1]), steps=0)
    cluster = bandit._fast_predict(s.context)

    # selection establishes which agent should be used (and gives us a valid arm_id)
    arm = bandit.select(s)
    reward = 1.5

    # Perform update; must land on the same cluster's agent
    bandit.update(arm, reward, s)

    # Check routing correctness
    for cid, ag in created.items():
        if cid == cluster:
            assert len(ag.update_calls) == 1, f"Agent {cid} should have exactly one update call."
            u_arm, u_reward, u_state = ag.update_calls[0]
            assert u_arm == arm, f"Expected arm {arm} in update, got {u_arm}"
            assert u_reward == reward, f"Expected reward {reward}, got {u_reward}"
            assert u_state is s, "Expected the same State instance passed to update"
        else:
            assert len(ag.update_calls) == 0, f"Agent {cid} should not receive updates for this context."
