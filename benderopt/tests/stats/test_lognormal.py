from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
from benderopt.utils import logb
import numpy as np
from scipy import stats

np.random.seed(0)


def test_lognormal_generator():
    """Test to reassure."""
    base = 10
    mu = 1e-5
    sigma = 1e2
    low = 1e-7
    high = 1e1
    step = None
    size = 1000000
    epsilon = 1e-1

    mu_log = logb(mu, base)
    sigma_log = logb(sigma, base)
    low_log = logb(low, base)
    high_log = logb(high, base)
    samples = logb(sample_generators["lognormal"](size=size,
                                                  mu_log=mu_log,
                                                  sigma_log=sigma_log,
                                                  low_log=low_log,
                                                  low=low,
                                                  high_log=high_log,
                                                  base=base,
                                                  step=step), base)
    a = (low_log - mu_log) / sigma_log
    b = (high_log - mu_log) / sigma_log
    # Median
    theorical_median = stats.truncnorm.median(a=a, b=b, loc=mu_log, scale=sigma_log)
    assert np.abs(np.median(samples) - theorical_median) / theorical_median < epsilon
    # mean (expected value)
    theorical_mean = stats.truncnorm.mean(a=a, b=b, loc=mu_log, scale=sigma_log)
    assert np.abs(np.mean(samples) - theorical_mean) / theorical_mean < epsilon
    assert np.sum(samples < low_log) == 0
    assert np.sum(samples >= high_log) == 0


def test_lognormal_generator_step():
    """Test to reassure."""
    mu = 50
    sigma = 3
    low = 2
    high = 100
    base = 10

    step = 2
    size = 100000

    mu_log = logb(mu, base)
    sigma_log = logb(sigma, base)
    low_log = logb(low, base)
    high_log = logb(high, base)
    samples = sample_generators["lognormal"](size=size,
                                             mu_log=mu_log,
                                             sigma_log=sigma_log,
                                             low_log=low_log,
                                             low=low,
                                             high_log=high_log,
                                             base=base,
                                             step=step)

    assert np.sum(samples % 2) == 0
    assert np.sum(samples < low) == 0
    assert np.sum(samples >= high) == 0


def test_lognormal_pdf():
    mu = 10 ** 3.34
    sigma = 10 ** 2
    low = 10 ** 1.125
    high = 10 ** 4.365
    step = None
    base = 10

    size = 100000
    bins = 100
    epsilon = 1e-1

    low_log = logb(low, base)
    high_log = logb(high, base)
    mu_log = logb(mu, base)
    sigma_log = logb(sigma, base)

    samples = sample_generators["lognormal"](size=size,
                                             low=low,
                                             high=high,
                                             mu=mu,
                                             sigma=sigma,
                                             mu_log=mu_log,
                                             sigma_log=sigma_log,
                                             low_log=low_log,
                                             high_log=high_log,
                                             step=step,
                                             base=base)
    hist, bin_edges = np.histogram(samples, bins=bins, density=True)
    densities = probability_density_function["lognormal"](
        samples=(bin_edges[1:] + bin_edges[:-1]) * 0.5, low_log=low_log, high_log=high_log,
        low=low, high=high, step=step, base=base, mu=mu, sigma=sigma, mu_log=mu_log,
        sigma_log=sigma_log)
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon


def test_lognormal_pdf_step():
    mu = 50
    sigma = 20
    low = 2
    high = 100
    base = 10

    step = 2
    size = 100000
    epsilon = 1e-1

    mu_log = logb(mu, base)
    sigma_log = logb(sigma, base)
    low_log = logb(low, base)
    high_log = logb(high, base)

    samples = sample_generators["lognormal"](size=size,
                                             low=low,
                                             high=high,
                                             base=base,
                                             step=step,
                                             mu=mu,
                                             mu_log=mu_log,
                                             sigma_log=sigma_log,
                                             low_log=low_log,
                                             high_log=high_log,
                                             sigma=sigma)

    hist, bin_edges = np.histogram(samples,
                                   bins=np.arange(low - step / 10, high, step),
                                   density=True)
    densities = probability_density_function["lognormal"](
        samples=bin_edges[0:-1] + step / 10,
        low=low,
        high=high,
        base=base,
        step=step,
        mu_log=mu_log,
        sigma_log=sigma_log,
        low_log=low_log,
        high_log=high_log,
        mu=mu,
        sigma=sigma)
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert np.sum(densities[(bin_edges[1:] + bin_edges[:-1]) * 0.5 < low]) == 0
    assert ((hist - densities) / densities).mean() <= epsilon
