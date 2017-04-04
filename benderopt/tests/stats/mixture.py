from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
import numpy as np

np.random.seed(0)

def test_mixture_generator():
    weights = [0.2, 0.8]
    parameters = [
        {
            "name": "param1",
            "category": "uniform",
            "search_space": {
                "low": 0,
                "high": 1,
            }
        },
        {
            "name": "param2",
            "category": "uniform",
            "search_space": {
                "low": -1,
                "high": 0,
            }
        }
    ]
    size = 1000
    samples = list(sample_generators["mixture"](weights=weights,
                                                parameters=parameters,
                                                size=size))
    assert np.abs(np.mean(samples) + 0.3) < 1


def test_mixture_with_category_generator():
    weights = [0.2, 0.8]
    parameters = [
        {
            "name": "param1",
            "category": "uniform",
            "search_space": {
                "low": -1,
                "high": 0,
            }
        },
        {
            "name": "param2",
            "category": "categorical",
            "search_space": {
                "values": [5, 10],
                "probabilities": [0.5, 0.5]
            }
        }
    ]
    size = 1000
    samples = list(sample_generators["mixture"](weights=weights,
                                                parameters=parameters,
                                                size=size))
    assert np.abs(np.mean(samples) - 5.9) < 1
