import pytest

from benderopt.rng import RNG

@pytest.fixture(autouse=True)
def seed_rng():
    RNG.seed()
