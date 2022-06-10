import numpy as np

from benderopt.rng import RNG

HIGH = 1e10


def test_seed():
    RNG.seed(42)
    assert RNG.integers(HIGH) == 7739560485


def test_separate_rng_seed():
    np.random.seed(0)
    custom_rng1 = RNG.integers(HIGH)
    numpy_rng1 = np.random.randint(HIGH)

    np.random.seed(0)
    custom_rng2 = RNG.integers(HIGH)
    numpy_rng2 = np.random.randint(HIGH)

    assert custom_rng1 != custom_rng2
    assert numpy_rng1 == numpy_rng2
