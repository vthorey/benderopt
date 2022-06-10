import numpy as np


class ResetableRNG:
    def __init__(self, seed=None):
        self.seed(seed)

    def __getattr__(self, name: str):
        return getattr(self._rng, name)

    def seed(self, seed=0):
        self._rng = np.random.default_rng(seed)


RNG = ResetableRNG()

__all__ = ["RNG"]
