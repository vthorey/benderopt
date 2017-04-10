import numpy as np
from benderopt.minimizer import minimize


def f(x):
    y = np.sin(x)
    return (y - 1) ** 2


def test_random_uniform():

    optimization_problem = [
        {
            "name": "x",
            "category": "uniform",
            "search_space": {
                "low": 0,
                "high": np.pi,
            }
        }
    ]

    best_sample = minimize(f,
                           optimization_problem,
                           optimizer_type="random",
                           number_of_evaluation=100)

    assert np.abs(best_sample["x"] - (np.pi / 2)) < 0.1
