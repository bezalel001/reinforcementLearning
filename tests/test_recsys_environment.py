import pytest
import numpy as np
from environments.recsys import RecsysEnvironment


def valid_kwargs():
    """Baseline valid parameters for constructing the environment."""
    return dict(
        rng=np.random.RandomState(0),
        users_per_minute=120.0,
        n_user_clusters=13,
        n_topics=25,
        embedding_dim=6,
        user_cluster_std=0.3,
        topic_std=0.2,
        subtopic_drift=0.1,
        weight_personal=5.0,
        weight_topic=1.0,
        weight_publisher=0.3,
        weight_article=0.0,
        weight_freshness=5.0,
        decay_hours=36.0,
        global_bias=0.0,
        score_noise=0.01,
        normalize_vectors=True,
        n_publishers=40,
        catalog_size=10,
        new_articles_per_hour=0.5,
    )


def ctor_raises(**overrides):
    """Helper that builds the environment and returns (raised, exception_type)."""
    kwargs = valid_kwargs()
    kwargs.update(overrides)
    try:
        RecsysEnvironment(**kwargs)
        return False, None
    except Exception as e:
        return True, type(e)


# ---------------------------
# rng (must be RandomState)
# ---------------------------

@pytest.mark.parametrize(
    "val,expect_exception",
    [
        (None, TypeError),
        (True, TypeError),
        (False, TypeError),
        (0, TypeError),
        (1.0, TypeError),
        ("rng", TypeError),
        (np.random.RandomState(123), None),  # valid
    ],
)
def test_rng_type_strict(val, expect_exception):
    raised, exc = ctor_raises(rng=val)
    if expect_exception is None:
        assert not raised
    else:
        assert raised and issubclass(exc, expect_exception)


# ----------------------------------------
# users_per_minute (positive float)
# ----------------------------------------

@pytest.mark.parametrize(
    "val,expect",
    [
        (None, TypeError),
        (True, TypeError),
        (False, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (0.0, ValueError),
        (-0.5, ValueError),
        ("120", TypeError),
        (120.0, None),
        (200, None),
    ],
)
def test_users_per_minute_strict(val, expect):
    raised, exc = ctor_raises(users_per_minute=val)
    if expect is None:
        assert not raised
    else:
        assert raised and issubclass(exc, expect)


# ---------------------------------------------------------
# n_user_clusters / n_topics / embedding_dim (positive int)
# ---------------------------------------------------------

@pytest.mark.parametrize("field", ["n_user_clusters", "n_topics", "embedding_dim"])
@pytest.mark.parametrize(
    "val,expect",
    [
        (None, TypeError),
        (True, TypeError),
        (False, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (6.0, TypeError),
        ("x", TypeError),
        (3, None),
    ],
)
def test_counts_positive_int_strict(field, val, expect):
    raised, exc = ctor_raises(**{field: val})
    if expect is None:
        assert not raised
    else:
        assert raised and issubclass(exc, expect)


# --------------------------------
# catalog_size (positive int)
# --------------------------------

@pytest.mark.parametrize(
    "val,expect",
    [
        (None, TypeError),
        (True, TypeError),
        (False, TypeError),
        (0, ValueError),
        (-5, ValueError),
        (5.5, TypeError),
        ("x", TypeError),
        (10, None),
    ],
)
def test_catalog_size_strict(val, expect):
    raised, exc = ctor_raises(catalog_size=val)
    if expect is None:
        assert not raised
    else:
        assert raised and issubclass(exc, expect)


# --------------------------------------------------------
# n_publishers (optional int)
# --------------------------------------------------------

@pytest.mark.parametrize(
    "val,expect",
    [
        (None, None),
        (True, TypeError),
        (False, TypeError),
        (-1, ValueError),
        (0, ValueError),
        ("40", TypeError),
        (5.5, TypeError),
        (10, None),
    ],
)
def test_n_publishers_strict(val, expect):
    raised, exc = ctor_raises(n_publishers=val)
    if expect is None:
        assert not raised
    else:
        assert raised and issubclass(exc, expect)


# ----------------------------------------------------------------
# new_articles_per_hour (numeric >= 0)
# ----------------------------------------------------------------

@pytest.mark.parametrize(
    "val,expect",
    [
        (None, TypeError),
        (True, TypeError),
        (False, TypeError),
        ("x", TypeError),
        (-1, ValueError),
        (0.0, None),
        (2.5, None),
    ],
)
def test_new_articles_per_hour_strict(val, expect):
    raised, exc = ctor_raises(new_articles_per_hour=val)
    if expect is None:
        assert not raised
    else:
        assert raised and issubclass(exc, expect)


# ------------------------------------------------------------------------
# Floating params (stds/weights/bias/noise/decay): must be numeric and non-boolean
# ------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "user_cluster_std",
    "topic_std",
    "subtopic_drift",
    "weight_personal",
    "weight_topic",
    "weight_publisher",
    "weight_article",
    "weight_freshness",
    "decay_hours",
    "global_bias",
    "score_noise",
])
@pytest.mark.parametrize(
    "val,expect",
    [
        (None, TypeError),
        (True, TypeError),
        (False, TypeError),
        ("x", TypeError),
        (-1.0, None),
        (0.0, None),
        (1.5, None),
    ],
)
def test_float_like_params_strict(field, val, expect):
    raised, exc = ctor_raises(**{field: val})
    if expect is None:
        assert not raised
    else:
        assert raised and issubclass(exc, expect)


# -------------------------------------------------------
# normalize_vectors (must be bool)
# -------------------------------------------------------

@pytest.mark.parametrize(
    "val,expect",
    [
        (None, TypeError),
        (False, None),
        (True, None),
        (0, TypeError),
        (1, TypeError),
        (1.0, TypeError),
        ("yes", TypeError),
    ],
)
def test_normalize_vectors_strict(val, expect):
    raised, exc = ctor_raises(normalize_vectors=val)
    if expect is None:
        assert not raised
    else:
        assert raised and issubclass(exc, expect)
