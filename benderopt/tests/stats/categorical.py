from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
import numpy as np

np.random.seed(0)


def test_categorical_generator():
    values = ["a", "b", "c"]
    probabilities = [0.1, 0.3, 0.6]
    size = 10000
    epsilon = 1e-2
    samples = list(sample_generators["categorical"](values=values,
                                                    probabilities=probabilities,
                                                    size=size))
    assert (samples.count("a") / size - probabilities[0]) < epsilon
    assert (samples.count("b") / size - probabilities[1]) < epsilon
    assert (samples.count("c") / size - probabilities[2]) < epsilon


def test_categorical_pdf():
    values = ["a", "b", "c"]
    probabilities = [0.1, 0.3, 0.6]
    size = 100
    samples = list(sample_generators["categorical"](values=values,
                                                    probabilities=probabilities,
                                                    size=size))
    densities = probability_density_function["categorical"](values=values,
                                                            probabilities=probabilities,
                                                            samples=samples)
    assert np.sum(densities == probabilities[0]) == samples.count(values[0])
    assert np.sum(densities == probabilities[1]) == samples.count(values[1])
    assert np.sum(densities == probabilities[2]) == samples.count(values[2])
