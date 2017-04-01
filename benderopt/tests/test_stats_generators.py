from benderopt.stats import sample_generators
import numpy as np
from scipy import stats


def test_categorical_generator():
    values = ["a", "b", "c"]
    probabilities = [0.1, 0.3, 0.6]
    size = 1000
    samples = list(sample_generators["categorical"](values=values,
                                                    probabilities=probabilities,
                                                    size=size))
    assert (samples.count("a") - probabilities[0] * size) < size / 10
    assert (samples.count("b") - probabilities[1] * size) < size / 10
    assert (samples.count("c") - probabilities[2] * size) < size / 10


def test_uniform_generator():
    data = {
        'low': 0,
        'high': 1,
        'log': False,
        'step': None,
        'size': 1000,
    }
    samples = sample_generators["uniform"](**data)
    assert np.abs(np.mean(samples) - 0.5) < 0.1


def test_log_uniform_generator():
    data = {
        'low': 0,
        'high': 1,
        'log': True,
        'step': None,
        'size': 1000,
    }
    samples = sample_generators["uniform"](**data)
    assert np.abs(np.mean(samples) - np.exp(0.5)) < 0.1


def test_uniform_step_generator():
    data = {
        'low': 0,
        'high': 10,
        'log': False,
        'step': 2,
        'size': 1000,
    }
    samples = sample_generators["uniform"](**data)
    assert np.sum(samples % 2) == 0


def test_log_uniform_step_generator():
    data = {
        'low': 0,
        'high': 10,
        'log': True,
        'step': 2,
        'size': 1000,
    }
    samples = sample_generators["uniform"](**data)
    assert np.sum(samples % 2) == 0


def test_normal_generator():
    mu = 5
    sigma = 3
    data = {
        'mu': mu,
        'sigma': sigma,
        'low': 0,
        'high': 15,
        'log': False,
        'step': None,
        'size': 1000,
    }
    alpha = (data["low"] - mu) / sigma
    beta = (data["high"] - mu) / sigma

    Z = stats.norm.cdf(beta, loc=mu, scale=sigma)

    mean = mu + sigma * (
        stats.norm.pdf(alpha, loc=mu, scale=sigma) - stats.norm.pdf(beta, loc=mu, scale=sigma)) / Z

    samples = sample_generators["normal"](**data)
    assert sum(samples > data["low"]) == data["size"]
    assert sum(samples < data["high"]) == data["size"]
    assert np.abs(np.mean(samples) - mean) < 0.1
    assert np.abs(np.std(samples) - 0.5) < 0.1


# def test_log_normal_generator():
#     data = {
#         'mu': 5,
#         'sigma': 0.5,
#         'low': 0,
#         'high': 10,
#         'log': True,
#         'step': None,
#         'size': 1000,
#     }
#     samples = sample_generators["normal"](**data)
#     assert np.abs(np.mean(samples) - np.exp(5 + (0.5 ** 2) / 2)) < 0.1
#     assert np.abs(np.std(samples) - np.exp(0.5)) < 0.1


# def test_normal_step_generator():
#     data = {
#         'mu': 5,
#         'sigma': 0.5,
#         'low': 0,
#         'high': 10,
#         'log': False,
#         'step': 2,
#         'size': 1000,
#     }
#     samples = sample_generators["normal"](**data)
#     assert np.sum(samples % 2) == 0


# def test_log_normal_step_generator():
#     data = {
#         'mu': 5,
#         'sigma': 0.5,
#         'low': 0,
#         'high': 10,
#         'log': True,
#         'step': 2,
#         'size': 1000,
#     }
#     samples = sample_generators["normal"](**data)
#     assert np.sum(samples % 2) == 0


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
