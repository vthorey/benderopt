import numpy as np
from scipy import stats
from benderopt.utils import logb


def generate_samples_lognormal(mu,
                               sigma,
                               low,
                               high,
                               step,
                               base,
                               size=1,
                               ):
    """Generate sample for (truncated)(discrete)log10normal density."""

    # Draw a samples which fit between low and high (if they are given)
    a = (logb(low, base) - logb(mu, base)) / logb(sigma, base)
    b = (logb(high, base) - logb(mu, base)) / logb(sigma, base)
    samples = base ** stats.truncnorm.rvs(a=a,
                                          b=b,
                                          size=size,
                                          loc=logb(mu, base),
                                          scale=logb(sigma, base))

    if step:
        samples = step * np.floor(samples / step)

    return samples


def lognormal_cdf(samples,
                  mu,
                  sigma,
                  low,
                  high,
                  base,
                  ):
    """Evaluate (truncated)normal cumulated density function for each samples.

    https://onlinecourses.science.psu.edu/stat414/node/157

    From scipy:
    If X normal, log(X) = Y follow a lognormal dist if s=sigma and scale = exp(mu)
    So we infer for a base b : s = sigma * np.log(b) and scale = base ** mu
    are similar
    mu = 9.2156
    sigma = 8.457
    base = 145.2
    a = (stats.norm.rvs(size=1000000, loc=mu, scale=sigma))
    b = np.log(stats.lognorm.rvs(size=1000000, s=sigma * np.log(base), scale=base ** mu)) / np.log(base)

    plt.subplot(2, 1, 1)
    plt.hist(a, bins=5000)
    plt.subplot(2, 1, 2)
    plt.hist(b, bins=5000)
    plt.show()

    """
    parametrization = {
        's': logb(sigma, base) * np.log(base),
        'scale': base ** (logb(mu, base)),
    }
    cdf_low = stats.lognorm.cdf(low, **parametrization)
    cdf_high = stats.lognorm.cdf(high, **parametrization)
    values = (stats.lognorm.cdf(samples, **parametrization) - cdf_low) / (cdf_high - cdf_low)
    values[(samples < low)] = 0
    values[(samples >= high)] = 1
    return values


def lognormal_pdf(samples,
                  mu,
                  sigma,
                  low,
                  high,
                  base,
                  step
                  ):
    """Evaluate (truncated)(discrete)normal probability density function for each sample."""
    values = None
    if step is None:
        parametrization = {
            's': logb(sigma, base) * np.log(base),
            'scale': base ** (logb(mu, base)),
        }
        cdf_low = lognormal_cdf(
            np.array([low]), mu=mu, sigma=sigma, low=low, high=high, base=base)[0]
        cdf_high = lognormal_cdf(
            np.array([high]), mu=mu, sigma=sigma, low=low, high=high, base=base)[0]
        values = stats.lognorm.pdf(samples, **parametrization) / (cdf_high - cdf_low)

    else:
        values = (lognormal_cdf(samples + step, mu=mu, sigma=sigma, low=low, high=high, base=base) -
                  lognormal_cdf(samples, mu=mu, sigma=sigma, low=low, high=high, base=base))

    values[(samples < low) + (samples >= high)] = 0
    return values
