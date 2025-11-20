import pytest
import numpy as np
from environments.normal import MABTestEnvironment, MABTestState, MABTestAction

# __init__
def test_init_valid():
    rng = np.random.RandomState(42)
    env = MABTestEnvironment(rng, n_actions=5)
    assert env.n_actions == 5
    assert len(env.q_star) == 5


def test_init_invalid_rng():
    with pytest.raises(TypeError):
        MABTestEnvironment(rng="not_rng")

    with pytest.raises(TypeError):
        MABTestEnvironment(rng=None)

    with pytest.raises(TypeError):
        MABTestEnvironment(rng=0)

    with pytest.raises(TypeError):
        MABTestEnvironment(rng=-1)

    with pytest.raises(TypeError):
        MABTestEnvironment(rng=False)


def test_init_invalid_n_actions():
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError):
        MABTestEnvironment(rng, n_actions=0)

    with pytest.raises(ValueError):
        MABTestEnvironment(rng, n_actions=-1)

    with pytest.raises(TypeError):
        MABTestEnvironment(rng, n_actions=None)

    with pytest.raises(TypeError):
        MABTestEnvironment(rng, n_actions=False)

    with pytest.raises(TypeError):
        MABTestEnvironment(rng, n_actions=True)

    with pytest.raises(TypeError):
        MABTestEnvironment(rng, n_actions="asd")

# state
def test_state_structure():
    rng = np.random.RandomState(1)
    env = MABTestEnvironment(rng, n_actions=3)
    state = env.state()
    assert isinstance(state, MABTestState)
    assert len(state.actions) == 3
    assert all(isinstance(a, MABTestAction) for a in state.actions)

# step
def test_step_valid():
    rng = np.random.RandomState(123)
    env = MABTestEnvironment(rng, n_actions=3)
    reward = env.step(1)
    assert isinstance(reward, float)


def test_step_invalid_type():
    rng = np.random.RandomState(0)
    env = MABTestEnvironment(rng)
    with pytest.raises(TypeError):
        env.step("bad")

    with pytest.raises(TypeError):
        env.step(None)

    with pytest.raises(TypeError):
        env.step(False)

    with pytest.raises(TypeError):
        env.step(True)


def test_step_out_of_bounds():
    rng = np.random.RandomState(0)
    env = MABTestEnvironment(rng, n_actions=3)
    with pytest.raises(ValueError):
        env.step(5)

    with pytest.raises(ValueError):
        env.step(-1)
