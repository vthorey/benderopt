from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
import numpy as np

np.random.seed(0)


def test_loguniform_generator():

    low = 0
    high = 1.569
    step = None
    base = 10

    size = 100000
    epsilon = (base ** high - base ** low) / size * 1e2
    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)
    # Median
    assert np.abs(
        np.median(samples) -
        (base ** (0.5 * (high + low)))) < epsilon
    # mean (expected value)
    assert np.abs(
        np.mean(samples) -
        ((base ** high - base ** low) /
            ((high - low) * np.log(base)))) < epsilon
    assert np.sum(samples < base ** low) == 0
    assert np.sum(samples >= base ** high) == 0

    low = -5
    high = 1.56
    step = None
    base = 10

    size = 100000
    epsilon = (base ** high - base ** low) / size * 1e2
    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)
    # Median
    assert np.abs(
        np.median(samples) -
        (base ** (0.5 * (high + low)))) < epsilon
    # mean (expected value)
    assert np.abs(
        np.mean(samples) -
        ((base ** high - base ** low) /
            ((high - low) * np.log(base)))) < epsilon
    assert np.sum(samples < base ** low) == 0
    assert np.sum(samples >= base ** high) == 0


def test_loguniform_step_generator():

    low = 2
    high = 4
    step = 2
    base = 10

    size = 10000,
    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)
    assert np.sum(samples % step) == 0
    assert np.sum(samples < base ** low) == 0
    assert np.sum(samples >= base ** high) == 0


def test_loguniform_pdf():

    low = 1.125
    high = 4.365
    step = None
    base = 10

    size = 1000000
    bins = 100000
    epsilon = 1e-2
    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)
    hist, bin_edges = np.histogram(samples, bins=bins, normed=True)
    densities = probability_density_function["loguniform"](
        samples=(bin_edges[1:] + bin_edges[:-1]) * 0.5,
        low=low, high=high, step=step, base=base)
    assert np.sum(densities[samples < (base ** low)]) == 0
    assert np.sum(densities[samples >= (base ** high)]) == 0
    assert np.sum((hist - densities).mean()) <= epsilon

    low = -5.125
    high = -0.365
    step = None
    base = 10

    size = 1000000
    bins = 100000
    epsilon = 1e-2
    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)
    hist, bin_edges = np.histogram(samples, bins=bins, normed=True)
    densities = probability_density_function["loguniform"](
        samples=(bin_edges[1:] + bin_edges[:-1]) * 0.5,
        low=low, high=high, step=step, base=base)
    assert np.sum(densities[samples < (base ** low)]) == 0
    assert np.sum(densities[samples >= (base ** high)]) == 0
    assert np.sum((hist - densities).mean()) <= epsilon


def test_loguniform_step_pdf():

    low = -2
    high = 5
    step = 0.01
    base = 10
    size = 10000
    epsilon = 1e-2

    samples = sample_generators["loguniform"](size=size, low=low, high=high, step=step, base=base)

    hist, bin_edges = np.histogram(samples, bins=np.arange(base ** low - 1, base ** high, step),
                                   normed=True)
    densities = probability_density_function["loguniform"](
        samples=bin_edges[0:-1] + 1,
        low=low, high=high, step=step, base=base)
    assert np.sum(densities[samples < (base ** low)]) == 0
    assert np.sum(densities[samples >= (base ** high)]) == 0
    assert np.abs((hist - densities).mean()) <= epsilon
