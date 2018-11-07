from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
import numpy as np

np.random.seed(0)


def test_uniform_generator():

    low = 0
    high = 1
    step = None

    size = 10000
    epsilon = 1e-2
    samples = sample_generators["uniform"](size=size,
                                           low=low,
                                           high=high,
                                           step=step)
    assert np.abs(np.mean(samples) - (low + high) / 2) < epsilon
    assert np.abs(np.median(samples) - (low + high) / 2) < epsilon
    assert np.sum(samples < low) == 0
    assert np.sum(samples >= high) == 0


def test_uniform_step_generator():

    low = 0
    high = 10
    step = 2

    size = 10000
    samples = sample_generators["uniform"](size=size,
                                           low=low,
                                           high=high,
                                           step=step)
    assert np.sum(samples % 2) == 0
    assert np.sum(samples < low) == 0
    assert np.sum(samples >= high) == 0


def test_uniform_pdf():

    low = 0
    high = 2
    step = None
    size = 10000
    bins = 1000
    epsilon = 1e-1
    samples = sample_generators["uniform"](size=size, low=low, high=high, step=step)
    hist, bin_edges = np.histogram(samples, bins=bins, density=True)
    densities = probability_density_function["uniform"](
        samples=(bin_edges[1:] + bin_edges[:-1]) * 0.5,
        low=low, high=high, step=step)
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon


def test_uniform_step_pdf():

    low = 0
    high = 10
    step = 2
    size = 10000
    epsilon = 1e-1

    samples = sample_generators["uniform"](size=size, low=low, high=high, step=step)

    hist, bin_edges = np.histogram(samples,
                                   bins=np.arange(low - step / 10, high, step),
                                   density=True)
    densities = probability_density_function["uniform"](
        samples=bin_edges[0:-1] + step / 10,
        low=low, high=high, step=step)
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon
