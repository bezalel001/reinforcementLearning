import numpy as np
import pytest

from bandits import UCBAgent
from environments.base import State, Action


# ----------------------------
# Helpers / Fixtures
# ----------------------------

@pytest.fixture
def agent():
    """Fresh UCB agent for each test."""
    return UCBAgent(c=2.0, initial_value=0.0)


def _ids(state):
    """Utility: return list of action IDs for clearer assertion messages."""
    return [a.id for a in state.actions]


# ----------------------------
# Tests
# ----------------------------


#---- select
def test_ucb_lazy_init_and_forced_first_round(agent):
    """
    On first encounters, the agent must lazily initialize new arms (Q,N)
    and *force* each available arm to be selected once (N[a] becomes 1 for all).
    """
    s = State(actions=[Action(1), Action(2), Action(3)], context=np.array(0), steps=0)

    # Before selection, internal maps are empty
    assert agent.Q == {} and agent.N == {}, f"Expected empty Q/N, got Q={agent.Q}, N={agent.N}"

    # First pass: each select() should return an unseen arm in available order
    for expected in [1, 2, 3]:
        chosen = agent.select(s)
        assert chosen == expected, (
            f"Expected first-round forced selection to pick unseen arm {expected}, "
            f"got {chosen}; available={_ids(s)} unseen-order is respected."
        )
        # Update once to mark this arm as seen
        agent.update(chosen, float(0.0), s)

    # After forcing each arm once, all should be initialized and seen
    assert set(agent.Q.keys()) == {1, 2, 3}, f"Q keys mismatch after init: {set(agent.Q.keys())}"
    assert set(agent.N.keys()) == {1, 2, 3}, f"N keys mismatch after init: {set(agent.N.keys())}"
    assert all(agent.N[a] == 1 for a in [1, 2, 3]), f"Each arm should have N=1; got N={agent.N}"
    assert agent.t == 3, f"Global timestep t should be 3 after three updates; got t={agent.t}"



def test_ucb_selection_matches_ucb_index_after_first_round(agent):
    """
    After every available arm has N[a] >= 1, select() should pick argmax of:
    Q[a] + c * sqrt(ln(t) / N[a]) among currently available arms.
    """
    s = State(actions=[Action(1), Action(2)], context=np.array(0), steps=0)

    # Force both arms to be seen once with defined Q/N/t
    for _ in range(2):
        a = agent.select(s)
        agent.update(a, float(0.0), s)

    # Manually set a controlled internal state for a deterministic check
    # t must be > 1, N must be >=1 for both arms (already satisfied above).
    agent.Q[1], agent.N[1] = 0.50, 10
    agent.Q[2], agent.N[2] = 0.40, 1
    agent.t = 11  # matches N1+N2

    expected = 2
    chosen = agent.select(s)
    assert chosen == expected, (
        "UCB selection mismatch.\n"
        f"chosen={chosen}, expected={expected}\n"
        f"State: Q={agent.Q}, N={agent.N}, t={agent.t}, c={agent.c}"
    )


def test_ucb_initializes_and_selects_newly_appearing_arm(agent):
    """
    When a new arm appears in the available set, the agent must lazily initialize it
    and select it (forced try) before using UCB indices among already-seen arms.
    """
    # First step: arms 1 and 2
    s1 = State(actions=[Action(1), Action(2)], context=np.array(0), steps=0)

    # Force both to be selected once
    for _ in range(2):
        a = agent.select(s1)
        agent.update(a, float(0.0), s1)

    assert set(agent.Q.keys()) == {1, 2}, f"Expected Q keys {{1,2}} after s1, got {set(agent.Q.keys())}"
    assert all(agent.N[a] == 1 for a in [1, 2]), f"Both arms should be seen once, got N={agent.N}"

    # Second step: arm 3 appears, arm 1 remains
    s2 = State(actions=[Action(1), Action(3)], context=np.array(0), steps=1)

    # Since arm 3 has N[3]==0 (unseen), select() MUST pick 3 first
    chosen = agent.select(s2)
    assert chosen == 3, (
        f"Expected forced selection of new unseen arm 3, got {chosen}; "
        f"available now={_ids(s2)}, N={agent.N}"
    )

    # After update, arm 3 should be initialized with N3==1
    agent.update(chosen, float(1.0), s2)
    assert 3 in agent.Q and 3 in agent.N and agent.N[3] == 1, (
        f"New arm 3 should be initialized and updated once; Q={agent.Q}, N={agent.N}"
    )


def test_ucb_ignores_unavailable_arms(agent):
    """
    Selection must only consider currently available actions, even if other arms
    exist in the agent's internal memory from previous steps.
    """
    # Initialize arms 1 and 2
    s1 = State(actions=[Action(1), Action(2)], context=np.array(0), steps=0)
    for _ in range(2):
        a = agent.select(s1)
        agent.update(a, float(0.0), s1)

    # Now only arm 2 is available
    s2 = State(actions=[Action(2)], context=np.array(0), steps=1)

    for i in range(5):
        chosen = agent.select(s2)
        assert chosen == 2, (
            f"Iteration {i}: Only arm 2 is available, but select() returned {chosen}. "
            f"Available={_ids(s2)}, Q={agent.Q}, N={agent.N}"
        )

#---- update
def test_ucb_update_applies_sample_average(agent):
    """
    update() must increment t, increment N[arm_id], and update Q with the sample average rule.
    """
    s = State(actions=[Action(7)], context=np.array(0), steps=0)

    # Force 7 to be selected at least once, then update with two rewards
    _ = agent.select(s)
    agent.update(7, float(1.0), s)   # Q7 -> 1.0
    assert agent.t == 1 and agent.N[7] == 1 and agent.Q[7] == pytest.approx(1.0), (
        f"After first update: t={agent.t}, N7={agent.N[7]}, Q7={agent.Q[7]}"
    )

    agent.update(7, float(3.0), s)   # Q7 -> average of [1,3] = 2.0
    assert agent.t == 2 and agent.N[7] == 2 and agent.Q[7] == pytest.approx(2.0), (
        f"After second update: expected t=2, N7=2, Q7approx2.0; got t={agent.t}, N7={agent.N[7]}, Q7={agent.Q[7]}"
    )

def test_ucb_throws_valueerror_on_wrong_update(agent):
    """
    The agent should thow an exception if any arm was not seen before.
    """
    s = State(actions=[Action(1)], context=np.array(1), steps=0)

    # no state seen
    with pytest.raises(ValueError) as ei:
        _ = agent.update(0, 1.0, s)

    # unseen action in state
    _ = agent.select(s)
    s = State(actions=[Action(2)], context=np.array(1), steps=0)
    with pytest.raises(ValueError) as ei:
        _ = agent.update(1, 1.0, s)

    # action not availble in state
    _ = agent.select(s)
    s = State(actions=[Action(2)], context=np.array(1), steps=0)
    with pytest.raises(ValueError) as ei:
        _ = agent.update(1, 1.0, s)


#---- default interface tests
def test_select_raises_typeerror_when_s_is_not_state(agent):
    """
    select(s): passing a non-State object must raise TypeError with a clear message.
    """
    not_a_state = {"actions": [1, 2, 3]}  # dict instead of State

    with pytest.raises(TypeError) as ei:
        agent.select(not_a_state)  # type: ignore[arg-type]
    assert "s must be a State" in str(ei.value), (
        f"Expected TypeError mentioning 's must be a State', got: {ei.value!r}"
    )


def test_update_raises_typeerror_when_arm_id_not_int(agent):
    """
    update(arm_id, reward, s): arm_id must be int; otherwise raise TypeError.
    """
    s = State(actions=[Action(1)], context=np.array(0), steps=0)

    with pytest.raises(TypeError) as ei:
        agent.update("1", 0.0, s)  # type: ignore[arg-type]
    # Match the exact message used by the implementation
    assert "arm id must be int" in str(ei.value), (
        f"Expected TypeError with 'arm id must be int', got: {ei.value!r}"
    )


def test_update_raises_typeerror_when_reward_not_float(agent):
    """
    update(arm_id, reward, s): reward must be float; otherwise raise TypeError.
    """
    s = State(actions=[Action(1)], context=np.array(0), steps=0)

    with pytest.raises(TypeError) as ei:
        agent.update(1, 1, s)  # int instead of float
    # Note: message in the method is "reward id must be float"
    assert "reward id must be float" in str(ei.value), (
        f"Expected TypeError with 'reward id must be float', got: {ei.value!r}"
    )


def test_update_raises_typeerror_when_s_not_state(agent):
    """
    update(arm_id, reward, s): s must be a State; otherwise raise TypeError.
    """
    with pytest.raises(TypeError) as ei:
        agent.update(1, 0.0, object())  # not a State
    # Note: message in the method is "s id must be State"
    assert "s id must be State" in str(ei.value), (
        f"Expected TypeError with 's id must be State', got: {ei.value!r}"
    )


def test_update_raises_valueerror_for_unseen_arm(agent):
    """
    update(arm_id, reward, s): if arm_id hasn't been initialized via select(), raise ValueError.
    """
    s = State(actions=[Action(1), Action(2)], context=np.array(0), steps=0)

    with pytest.raises(ValueError) as ei:
        agent.update(999, 0.0, s)  # 999 not initialized in Q
    assert "Unseen arm 999" in str(ei.value) and "call `select` before `update`" in str(ei.value), (
        f"Expected ValueError guiding user to call select() first, got: {ei.value!r}"
    )

