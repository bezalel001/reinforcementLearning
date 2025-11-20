from collections import Counter
import numpy as np
import pytest
from scipy.stats import chisquare

from bandits import EpsilonGreedyAgent
from environments.base import State, Action


@pytest.fixture
def rng():
    """Provide a deterministic RNG so tests are reproducible."""
    return np.random.RandomState(0)

@pytest.fixture
def agent(rng):
    """Fresh GradientBandit with baseline enabled by default."""
    return EpsilonGreedyAgent(rng=rng, epsilon=0.0)


#---- select
def test_egreedy_initializes_new_arms_on_select(rng, agent):
    """
    The agent should lazily initialize Q and N for any newly available arms on first select().
    """
    agent = EpsilonGreedyAgent(rng=rng, epsilon=0.0, initial_value=1.23)
    s = State(actions=[Action(1), Action(2), Action(3)], context=np.array(1), steps=0)

    # No internal state before first select
    assert agent.Q == {}, f"Expected empty Q before initialization, got {agent.Q!r}"
    assert agent.N == {}, f"Expected empty N before initialization, got {agent.N!r}"

    # triggers lazy init
    _ = agent.select(s)

    # Verify arms were added
    assert set(agent.Q.keys()) == {1, 2, 3}, f"Q keys mismatch: {set(agent.Q.keys())}"
    assert set(agent.N.keys()) == {1, 2, 3}, f"N keys mismatch: {set(agent.N.keys())}"

    # Verify initial values and counts
    for a_id, q in agent.Q.items():
        assert q == pytest.approx(1.23), f"Arm {a_id} Q expected 1.23, got {q}"
    for a_id, n in agent.N.items():
        assert n == 0, f"Arm {a_id} N expected 0, got {n}"


def test_egreedy_exploit_picks_argmax_among_available(rng):
    """
    With epsilon=0 (pure exploitation), select() should return the available arm with the largest Q.
    If the best arm disappears, the next-best among the *current* available arms is chosen.
    """
    agent = EpsilonGreedyAgent(rng=rng, epsilon=0.0)  # pure exploit
    s = State(actions=[Action(10), Action(11), Action(12)], context=np.array(1), steps=0)

    # Initialize
    agent.select(s)

    # Set custom values
    agent.Q[10] = 0.1
    agent.Q[11] = 0.9  # best
    agent.Q[12] = 0.5

    chosen = agent.select(s)
    assert chosen == 11, f"Expected best arm 11, got {chosen} with Q={agent.Q}"

    # If the best arm becomes unavailable, pick the best among remaining
    s2 = State(actions=[Action(10), Action(12)], context=np.array(1), steps=1)
    chosen2 = agent.select(s2)
    assert chosen2 == 12, (
        f"Expected best among available {{10,12}} to be 12 (Q={agent.Q[12]}), "
        f"got {chosen2} (Q={agent.Q.get(chosen2)})"
    )


def test_egreedy_explore_respects_available_set_when_epsilon_1(rng):
    """
    With epsilon=1 (pure exploration), sampled actions must be within the current available set.
    """
    agent = EpsilonGreedyAgent(rng=rng, epsilon=1.0)  # always explore

    # Warm-up with a different set to ensure dynamic availability handling
    s0 = State(actions=[Action(1), Action(11), Action(111)], context=np.array(1), steps=0)
    _ = agent.select(s0)

    # New availability set
    s = State(actions=[Action(1), Action(222), Action(333)], context=np.array(1), steps=0)
    allowed = {1, 222, 333}

    for i in range(1000):
        a = agent.select(s)
        assert a in allowed, (
            f"Sample #{i}: selected arm {a} not in allowed set {allowed}. "
            f"Available now: {[act.id for act in s.actions]}"
        )


def test_egreedy_explore_uniform_distribution_when_epsilon_1(rng):
    """
    With epsilon=1, the selection distribution should be approximately uniform across available arms.
    We perform a chi-square goodness-of-fit test at p=0.05 and expect to FAIL to reject uniformity.
    """
    agent = EpsilonGreedyAgent(rng=rng, epsilon=1.0)  # always explore
    s = State(actions=[Action(1), Action(222), Action(333)], context=np.array(1), steps=0)

    # Draw a large number of samples
    n = 2000  # slightly larger sample for better power/robustness
    history = [agent.select(s) for _ in range(n)]
    counts = Counter(history)

    # Expected uniform counts
    num_arms = len(s.actions)
    expected = [n / num_arms] * num_arms
    observed = [counts[a.id] for a in s.actions]

    chi2, p_value = chisquare(f_obs=observed, f_exp=expected)

    assert p_value > 0.05, (
        "Chi-square test indicates non-uniform sampling at alpha=0.05.\n"
        f"Observed counts (by arm order {[a.id for a in s.actions]}): {observed}\n"
        f"Expected uniform: {expected}\n"
        f"chi2={chi2:.3f}, p={p_value:.4f}. "
        "If this fails sporadically, increase sample size `n` or relax alpha slightly."
    )


def test_egreedy_dynamic_action_initialization_ignores_unavailable(rng):
    """
    The agent must initialize newly appearing arms but only choose from currently available actions.
    """
    agent = EpsilonGreedyAgent(rng=rng, epsilon=0.0, initial_value=0.0)

    s1 = State(actions=[Action(1), Action(2)], context=np.array(1), steps=0)
    agent.select(s1)  # initialize 1 and 2
    assert set(agent.Q.keys()) == {1, 2}, f"Expected Q keys {{1,2}}, got {set(agent.Q.keys())}"

    # Next step, arm 3 appears, 2 disappears
    s2 = State(actions=[Action(3), Action(1)], context=np.array(1), steps=0)
    _ = agent.select(s2)

    assert set(agent.Q.keys()) == {1, 2, 3}, f"Expected Q keys {{1,2,3}}, got {set(agent.Q.keys())}"

    for i in range(10):
        a = agent.select(s2)
        assert a in {1, 3}, (
            f"Iteration {i}: selected arm {a} but only arms {{1,3}} are available; "
            f"available={ [act.id for act in s2.actions] }"
        )


#---- update
def test_egreedy_throws_error_on_wrong_update(rng, agent):
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

        
def test_egreedy_update_sample_average_when_alpha_none(rng):
    """
    With alpha=None, Q updates should follow the sample-average rule over repeated updates to the same arm.
    """
    agent = EpsilonGreedyAgent(rng=rng, epsilon=0.0, alpha=None, initial_value=0.0)
    s = State(actions=[Action(0), Action(1), Action(2)], context=np.array(1), steps=0)

    # Initialize internal state
    agent.select(s)

    rewards = [1.0, 0.0, 2.0, 2.0]
    running_avg = 0.0
    for t, r in enumerate(rewards, start=1):
        agent.update(1, r, s)
        running_avg += (r - running_avg) / t
        assert agent.Q[1] == pytest.approx(running_avg), (
            f"After {t} updates with rewards={rewards[:t]}, expected Q[1]approx{running_avg}, "
            f"got {agent.Q[1]}"
        )
        assert agent.N[1] == t, (
            f"After {t} updates with expected N[1]={t}, "
            f"got {agent.N[1]}"
        )


def test_egreedy_update_constant_step_size_alpha(rng):
    """
    With a constant step-size alpha, Q should follow the exponential recency-weighted update rule.
    """
    agent = EpsilonGreedyAgent(rng=rng, epsilon=0.0, alpha=0.5, initial_value=0.0)
    s = State(actions=[Action(5), Action(4), Action(7)], context=np.array(1), steps=0)

    agent.select(s)

    agent.update(7, 1.0, s)  # Q <- 0 + 0.5*(1 - 0) = 0.5
    assert agent.Q[7] == pytest.approx(0.5), f"After first update, expected Q[7]=0.5, got {agent.Q[7]}"

    agent.update(7, 2.0, s)  # Q <- 0.5 + 0.5*(2 - 0.5) = 1.25
    assert agent.Q[7] == pytest.approx(1.25), f"After second update, expected Q[7]=1.25, got {agent.Q[7]}"


def test_egreedy_exploit_breaks_ties_by_first_available_order(rng):
    """
    Python's max over a list returns the first occurrence on ties; verify deterministic tie-breaking
    based on the order of the currently available IDs.
    """
    agent = EpsilonGreedyAgent(rng=rng, epsilon=0.0, initial_value=0.0)
    s = State(actions=[Action(5), Action(6), Action(7)], context=np.array(1), steps=0)
    agent.select(s)

    # Create a tie between 5 and 6 as the highest-valued arms
    agent.Q[5] = 1.0
    agent.Q[6] = 1.0
    agent.Q[7] = 0.5

    chosen = agent.select(s)
    assert chosen == 5, (
        f"With tie Q[5]=Q[6]=1.0 and available order [5,6,7], expected 5, got {chosen}"
    )


#---- default interface tests

def test_select_raises_typeerror_when_s_is_not_state(rng, agent):
    """
    select(s): passing a non-State object must raise TypeError with a clear message.
    """
    not_a_state = {"actions": [1, 2, 3]}  # dict instead of State

    with pytest.raises(TypeError) as ei:
        agent.select(not_a_state)  # type: ignore[arg-type]
    assert "s must be a State" in str(ei.value), (
        f"Expected TypeError mentioning 's must be a State', got: {ei.value!r}"
    )


def test_update_raises_typeerror_when_arm_id_not_int(rng, agent):
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


def test_update_raises_typeerror_when_reward_not_float(rng, agent):
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


def test_update_raises_typeerror_when_s_not_state(rng, agent):
    """
    update(arm_id, reward, s): s must be a State; otherwise raise TypeError.
    """
    with pytest.raises(TypeError) as ei:
        agent.update(1, 0.0, object())  # not a State
    # Note: message in the method is "s id must be State"
    assert "s id must be State" in str(ei.value), (
        f"Expected TypeError with 's id must be State', got: {ei.value!r}"
    )


def test_update_raises_valueerror_for_unseen_arm(rng, agent):
    """
    update(arm_id, reward, s): if arm_id hasn't been initialized via select(), raise ValueError.
    """
    s = State(actions=[Action(1), Action(2)], context=np.array(0), steps=0)

    with pytest.raises(ValueError) as ei:
        agent.update(999, 0.0, s)  # 999 not initialized in Q
    assert "Unseen arm 999" in str(ei.value) and "call `select` before `update`" in str(ei.value), (
        f"Expected ValueError guiding user to call select() first, got: {ei.value!r}"
    )
