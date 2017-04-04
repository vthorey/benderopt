from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
import numpy as np

np.random.seed(0)


def test_uniform_generator():
    data = {
        'low': 0,
        'high': 1,
        'step': None,
    }
    size = 10000,
    epsilon = 1e-2
    samples = sample_generators["uniform"](size=size, **data)
    assert np.abs(np.mean(samples) - 0.5) < epsilon
    assert np.sum(samples < data["low"]) == 0
    assert np.sum(samples >= data["high"]) == 0


def test_uniform_step_generator():
    data = {
        'low': 0,
        'high': 10,
        'step': 2,
    }
    size = 10000,
    samples = sample_generators["uniform"](size=size, **data)
    assert np.sum(samples % 2) == 0
    assert np.sum(samples < data["low"]) == 0
    assert np.sum(samples >= data["high"]) == 0


def test_uniform_pdf():
    data = {
        'low': 0,
        'high': 2.7,
        'step': None,
    }
    samples = np.arange(-3, 3, 0.001)
    densities = probability_density_function["uniform"](samples=samples, **data)
    assert np.sum(densities[samples < data["low"]]) == 0
    assert np.sum(densities[samples >= data["high"]]) == 0
    assert np.sum((densities[1:] - densities[:-1]) != 0) == 2
    assert densities[densities != 0][0] == 1 / (data["high"])
    assert (np.sum(densities) / len(samples) - 1) <= 1e-3


def test_uniform_step_pdf():
    data = {
        'low': 0,
        'high': 10,
        'step': 2,
    }
    samples = np.arange(-20, 15, 2)
    densities = probability_density_function["uniform"](samples=samples, **data)
    assert np.sum(densities[samples < data["low"]]) == 0
    assert np.sum(densities[samples >= data["high"]]) == 0
    assert np.sum((densities[1:] - densities[:-1]) != 0) == 2
    assert np.sum(densities) == 5 * 2 / 10
