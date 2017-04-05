from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
from benderopt.utils import logb
import numpy as np
from scipy import stats

np.random.seed(0)


def test_lognormal_generator():
    """Test to reassure."""
    mu = 1e-5
    sigma = 1e2
    low = 1e-7
    high = 1e1
    base = 10
    step = None
    size = 1000000
    epsilon = 1e-1

    samples = logb(sample_generators["lognormal"](size=size,
                                                  mu=mu,
                                                  sigma=sigma,
                                                  low=low,
                                                  high=high,
                                                  base=base,
                                                  step=step), base)
    a = (logb(low, base) - logb(mu, base)) / logb(sigma, base)
    b = (logb(high, base) - logb(mu, base)) / logb(sigma, base)
    # Median
    theorical_median = stats.truncnorm.median(a=a, b=b, loc=logb(mu, base), scale=logb(sigma, base))
    assert np.abs(np.median(samples) - theorical_median) / theorical_median < epsilon
    # mean (expected value)
    theorical_mean = stats.truncnorm.mean(a=a, b=b, loc=logb(mu, base), scale=logb(sigma, base))
    assert np.abs(np.mean(samples) - theorical_mean) / theorical_mean < epsilon
    assert np.sum(samples < logb(low, base)) == 0
    assert np.sum(samples >= logb(high, base)) == 0


def test_lognormal_generator_step():
    """Test to reassure."""
    mu = 50
    sigma = 3
    low = 0
    high = 100
    base = 10

    step = 2
    size = 100000

    samples = sample_generators["lognormal"](size=size,
                                             mu=mu,
                                             sigma=sigma,
                                             low=low,
                                             high=high,
                                             base=base,
                                             step=step)
    assert np.sum(samples % 2) == 0
    assert np.sum(samples < low) == 0
    assert np.sum(samples >= high) == 0


def test_lognormal_pdf():
    mu = 50
    sigma = 20
    low = 0
    high = 100
    base = 10

    step = None
    size = 100000
    bins = 10000
    epsilon = 1e-1
    samples = sample_generators["lognormal"](size=size,
                                             low=low,
                                             high=high,
                                             base=base,
                                             step=step,
                                             mu=mu,
                                             sigma=sigma)
    hist, bin_edges = np.histogram(samples, bins=bins, normed=True)
    densities = probability_density_function["lognormal"](
        samples=(bin_edges[1:] + bin_edges[:-1]) * 0.5,
        low=low, high=high,
        base=base, step=step, mu=mu, sigma=sigma)
    assert np.sum(densities[samples < low]) == 0
    assert np.sum(densities[samples >= high]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon


def test_lognormal_pdf_step():
    mu = 50
    sigma = 20
    low = 0
    high = 100
    base = 10

    step = 2
    size = 100000
    epsilon = 1e-1

    samples = sample_generators["lognormal"](size=size,
                                             low=low,
                                             high=high,
                                             base=base,
                                             step=step,
                                             mu=mu,
                                             sigma=sigma)

    hist, bin_edges = np.histogram(samples,
                                   bins=np.arange(low - step / 10, high, step),
                                   normed=True)
    densities = probability_density_function["lognormal"](
        samples=bin_edges[0:-1] + step / 10,
        low=low,
        high=high,
        base=base,
        step=step,
        mu=mu, sigma=sigma)
    assert np.sum(densities[samples < low]) == 0
    assert np.sum(densities[samples >= high]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon
