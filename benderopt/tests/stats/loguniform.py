from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function

import numpy as np

np.random.seed(0)


def logb(samples, base):
    return np.log(samples) / np.log(base)


def test_loguniform_generator():

    low = 10 ** -7.23
    high = 10 ** -6.569
    step = None
    base = 10

    size = 100000
    epsilon = 1e-1

    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)
    # Median
    theorical_median = base ** (0.5 * (logb(high, base) + logb(low, base)))
    assert np.abs(np.median(samples) - theorical_median) / theorical_median < epsilon
    # mean (expected value)
    theorical_mean = (high - low) / ((logb(high, base) - logb(low, base)) * np.log(base))
    assert np.abs(np.mean(samples) - theorical_mean) / theorical_mean < epsilon
    assert np.sum(samples < low) == 0
    assert np.sum(samples >= high) == 0


def test_loguniform_step_generator():

    low = 10 ** 2
    high = 10 ** 4
    step = 2
    base = 10

    size = 10000,
    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)
    assert np.sum(samples % step) == 0
    assert np.sum(samples < low) == 0
    assert np.sum(samples >= high) == 0


def test_loguniform_pdf():

    low = 10 ** 1.125
    high = 10 ** 4.365
    step = None
    base = 10

    size = 10000
    bins = 100
    epsilon = 1e-1
    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)
    hist, bin_edges = np.histogram(samples, bins=bins, normed=True)
    densities = probability_density_function["loguniform"](
        samples=(bin_edges[1:] + bin_edges[:-1]) * 0.5,
        low=low, high=high, step=step, base=base)
    assert np.sum(densities[samples < low]) == 0
    assert np.sum(densities[samples >= high]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon


def test_loguniform_step_pdf():

    low = 10 ** 0
    high = 10 ** 3
    step = low
    base = 10
    size = 10000
    epsilon = 1e-1

    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)

    hist, bin_edges = np.histogram(samples,
                                   bins=np.arange(low - step / 10, high, step),
                                   normed=True)
    densities = probability_density_function["loguniform"](
        samples=bin_edges[0:-1] + 1,
        low=low, high=high, step=step, base=base)
    assert np.sum(densities[samples < low]) == 0
    assert np.sum(densities[samples >= high]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon
