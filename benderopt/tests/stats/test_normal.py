from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
import numpy as np
from scipy import stats

np.random.seed(0)


def test_normal_generator():
    """Test to reassure."""
    mu = 5
    sigma = 5
    low = 0
    high = 15
    step = None
    size = 100000
    epsilon = 1e-1

    samples = sample_generators["normal"](size=size,
                                          mu=mu,
                                          sigma=sigma,
                                          low=low,
                                          high=high,
                                          step=step)
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    # Median
    theorical_median = stats.truncnorm.median(a=a, b=b, loc=mu, scale=sigma)
    assert np.abs(np.median(samples) - theorical_median) / theorical_median < epsilon
    # mean (expected value)
    theorical_mean = stats.truncnorm.mean(a=a, b=b, loc=mu, scale=sigma)
    assert np.abs(np.mean(samples) - theorical_mean) / theorical_mean < epsilon
    assert np.sum(samples < low) == 0
    assert np.sum(samples >= high) == 0


def test_normal_generator_step():
    """Test to reassure."""
    mu = 50
    sigma = 3
    low = 0
    high = 100
    step = 2
    size = 100000

    samples = sample_generators["normal"](size=size,
                                          mu=mu,
                                          sigma=sigma,
                                          low=low,
                                          high=high,
                                          step=step)
    assert np.sum(samples % 2) == 0
    assert np.sum(samples < low) == 0
    assert np.sum(samples >= high) == 0


def test_normal_pdf():
    mu = 50
    sigma = 20
    low = 0
    high = 100
    step = None
    size = 100000
    bins = 10000
    epsilon = 1e-1
    samples = sample_generators["normal"](size=size,
                                          low=low,
                                          high=high,
                                          step=step,
                                          mu=mu,
                                          sigma=sigma)
    hist, bin_edges = np.histogram(samples, bins=bins, density=True)
    densities = probability_density_function["normal"](
        samples=(bin_edges[1:] + bin_edges[:-1]) * 0.5,
        low=low, high=high, step=step, mu=mu, sigma=sigma)
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon


def test_normal_pdf_step():
    mu = 50
    sigma = 20
    low = 0
    high = 100
    step = 2
    size = 100000
    epsilon = 1e-1

    samples = sample_generators["normal"](size=size,
                                          low=low,
                                          high=high,
                                          step=step,
                                          mu=mu,
                                          sigma=sigma)

    hist, bin_edges = np.histogram(samples,
                                   bins=np.arange(low - step / 10, high, step),
                                   density=True)
    densities = probability_density_function["normal"](
        samples=bin_edges[0:-1] + step / 10,
        low=low, high=high, step=step, mu=mu, sigma=sigma)
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon
