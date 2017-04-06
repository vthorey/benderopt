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


def test_random_normal():

    optimization_problem = [
        {
            "name": "x",
            "category": "normal",
            "search_space": {
                "mu": np.pi / 2,
                "sigma": 0.3,
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


def test_random_categorical():

    optimization_problem = [
        {
            "name": "x",
            "category": "categorical",
            "search_space": {
                "values": [0,
                           np.pi / 5,
                           np.pi / 4,
                           np.pi / 3,
                           np.pi / 2,
                           np.pi]
            }
        }
    ]

    best_sample = minimize(f,
                           optimization_problem,
                           optimizer_type="random",
                           number_of_evaluation=100)

    assert np.abs(best_sample["x"] - (np.pi / 2)) < 1e-5


def test_random_uniform_step():

    optimization_problem = [
        {
            "name": "x",
            "category": "uniform",
            "search_space": {
                "low": 0,
                "high": np.pi,
                "step": 0.01,
            }
        }
    ]

    best_sample = minimize(f,
                           optimization_problem,
                           optimizer_type="random",
                           number_of_evaluation=100)

    assert np.abs(best_sample["x"] - (np.pi / 2)) < 0.1
