import os

import numpy as np


def get_test_optimization_problem():
    from benderopt.base import OptimizationProblem

    return OptimizationProblem.from_json(
        "{}/tests/test_data.json".format(os.path.dirname(os.path.abspath(__file__)))
    )


def logb(samples, base):
    return np.log(samples) / np.log(base)
