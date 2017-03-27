from benderopt.stats import sample_generators
import numpy as np


def test_categorical_draw():
    values = ["a", "b", "c"]
    probabilities = [0.1, 0.3, 0.6]
    size = 1000
    samples = list(sample_generators["categorical"](values=values,
                                                    probabilities=probabilities,
                                                    size=size))
    assert (samples.count("a") - probabilities[0] * size) < size / 10
    assert (samples.count("b") - probabilities[1] * size) < size / 10
    assert (samples.count("c") - probabilities[2] * size) < size / 10


def test_mixture_draw():
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
    assert np.mean(samples) < 0
