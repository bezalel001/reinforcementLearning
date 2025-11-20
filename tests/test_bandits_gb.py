from collections import Counter
import numpy as np
import pytest

from bandits import GradientBandit
from environments.base import State, Action


# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.RandomState(0)


@pytest.fixture
def agent(rng):
    """Fresh GradientBandit with baseline enabled by default."""
    return GradientBandit(rng=rng, alpha=0.1)

def _ids(state):
    """Utility: return list of action IDs for clearer assertion messages."""
    return [a.id for a in state.actions]


# ----------------------------
# Tests
# ----------------------------

#---- select

def test_new_action_is_lazily_initialized_on_select(agent):
    """
    When a new action appears, select() should lazily initialize H[new]=0.0
    and include it in the sampling distribution.
    """
    # Initial state: arms 1 and 2
    s1 = State(actions=[Action(1), Action(2)], context=np.array(0), steps=0)
    _ = agent.select(s1)  # lazy init for 1 and 2
    assert set(agent.H.keys()) == {1, 2}, f"Expected H keys {{1,2}}, got {set(agent.H.keys())}"

    # New state: arm 3 appears, arm 1 remains
    s2 = State(actions=[Action(1), Action(3)], context=np.array(0), steps=1)
    choice = agent.select(s2)

    # H should now include the newly-appeared arm 3 with preference 0.0
    assert set(agent.H.keys()) == {1, 2, 3}, f"Expected H keys {{1,2,3}}, got {set(agent.H.keys())}"
    assert agent.H[3] == pytest.approx(0.0), f"New arm 3 should initialize to H=0.0, got {agent.H[3]}"
    assert choice in {1, 3}, (
        f"Selection must be from currently available arms {{1,3}}, got {choice}"
    )

def test_select_lazy_init_and_uniform_when_equal_prefs(agent):
    """
    select() should lazily initialize preferences H[a]=0.0 for newly available arms.
    With equal preferences, the softmax distribution should be (approximately) uniform.
    """
    s = State(actions=[Action(1), Action(2), Action(3)], context=np.array(0), steps=0)

    # Precondition: H is empty
    assert agent.H == {}, f"Expected empty H before any select(), got H={agent.H!r}"

    # Call select() to trigger lazy init
    choice = agent.select(s)
    assert choice in {1, 2, 3}, f"select() returned {choice}, not in available {_ids(s)}"

    # All available arms should now be initialized in H
    assert set(agent.H.keys()) == {1, 2, 3}, f"H keys mismatch: {set(agent.H.keys())}"

    # With equal preferences, softmax should be close to uniform (not asserting exact values)
    probs = agent._softmax(_ids(s))
    expected = 1.0 / len(s.actions)
    for a in _ids(s):
        assert pytest.approx(probs[a], rel=0.15) == expected, (
            f"Expected ~uniform prob {expected:.3f} for arm {a}, got {probs[a]:.3f}"
        )


def test_select_respects_preference_ranking(agent, rng):
    """
    With distinct preferences, the arm with the higher preference should be sampled more often.
    We check rank order of frequencies rather than exact probabilities.
    """
    # Manually seed preferences to create a strong ordering: 5 >> 6 >> 7
    agent.H = {5: 5.0, 6: 0.0, 7: -5.0}
    s = State(actions=[Action(5), Action(6), Action(7)], context=np.array(0), steps=0)

    history = [agent.select(s) for _ in range(2000)]
    counts = Counter(history)

    assert counts[5] > counts[6] > counts[7], (
        f"Expected sampling order 5 > 6 > 7 due to preferences H={agent.H}; "
        f"observed counts={counts}"
    )


#---- update
def test_gb_throws_error_on_wrong_update(rng, agent):
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


def test_update_rule_with_baseline_true(agent):
    """
    update(): preferences must be updated with H[a] += alpha*(r - b)*(1{a=i} - pi(a)),
    where b is avg_reward. Also verify t and avg_reward increments.
    """
    s = State(actions=[Action(1), Action(2)], context=np.array(0), steps=0)
    # Ensure lazy init (H[1], H[2] = 0)
    _ = agent.select(s)

    # Snapshot before update
    H_before = dict(agent.H)
    b_before = agent.avg_reward
    t_before = agent.t
    probs = agent._softmax(_ids(s))

    # Apply update with a known reward on arm 1
    r = 0.5
    agent.update(arm_id=1, reward=float(r), s=s)

    # Expected updates
    for a in _ids(s):
        indicator = 1.0 if a == 1 else 0.0
        expected = H_before[a] + agent.alpha * (r - b_before) * (indicator - probs[a])
        assert agent.H[a] == pytest.approx(expected), (
            f"H[{a}] mismatch: expected {expected}, got {agent.H[a]}. "
            f"Before: {H_before}, probs={probs}, b={b_before}"
        )

    # t increments by 1; avg_reward becomes running mean (here first update)
    assert agent.t == t_before + 1, f"Expected t={t_before+1}, got {agent.t}"
    expected_avg = b_before + (r - b_before) / max(1, t_before + 1)
    assert agent.avg_reward == pytest.approx(expected_avg), (
        f"avg_reward mismatch: expected {expected_avg}, got {agent.avg_reward}"
    )



def test_update_ignores_unavailable_arms(agent):
    """
    update() should only update preferences for actions present in the current state's actions.
    Preferences for arms not in s.actions must remain unchanged.
    """
    # Seed three arms in H
    agent.H = {1: 0.2, 2: -0.1, 3: 0.7}
    # Current state only exposes arms 1 and 2
    s = State(actions=[Action(1), Action(2)], context=np.array(0), steps=0)

    H3_before = agent.H[3]
    # Ensure the chosen arm is known to the agent and present in state
    agent.H[1] = 0.0

    # Also ensure avg_reward/t have a defined state
    t_before = agent.t
    agent.update(arm_id=1, reward=float(1.0), s=s)

    # Arm 3 should be untouched
    assert agent.H[3] == pytest.approx(H3_before), (
        f"Preferences for unavailable arm 3 must not change; was {H3_before}, now {agent.H[3]}"
    )
    assert agent.t == t_before + 1, f"t should increment by 1; expected {t_before+1}, got {agent.t}"



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
