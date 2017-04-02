import numpy as np
from scipy import stats


def generate_samples_lognormal(mu,
                               sigma,
                               low,
                               high,
                               step=None,
                               size=1,
                               max_retry=50
                               ):
    """Generate sample for (truncated)(discrete)normal density."""

    # Draw a samples which fit between low and high (if they are given)
    a, b = (low - mu) / sigma, (high - mu) / sigma
    samples = np.exp(stats.truncnorm.rvs(a=a, b=b, size=size, loc=mu, scale=sigma))

    if step:
        samples = step * np.round(samples / step)

    return samples


def lognormal_cdf(samples,
                  mu,
                  sigma,
                  low,
                  high,
                  ):
    """Evaluate (truncated)normal cumulated density function for each samples.

    http://mathworld.wolfram.com/GibratsDistribution.html

    From scipy:
    If X normal, log(X) = Y follow a lognormal dist if s=sigma and scale = exp(mu)
    np.exp(stats.norm.rvs(size=1000000, loc=mu, scale=sigma))
    is similar to
    stats.lognorm.rvs(size=1000000, s=sigma, scale=np.exp(mu))
    """
    parametrization = {
        's': sigma,
        'scale': np.exp(mu),
    }
    cdf_low = stats.lognorm.cdf(low, **parametrization)
    cdf_high = stats.lognorm.cdf(high, **parametrization)
    values = (stats.lognorm.cdf(samples, **parametrization) - cdf_low) / (cdf_high - cdf_low)
    values[(samples < low)] = 0
    values[(samples > high)] = 1

    return values


def lognormal_pdf(samples,
                  mu,
                  sigma,
                  low,
                  high,
                  step=None
                  ):
    """Evaluate (truncated)(discrete)normal probability density function for each sample."""
    values = None
    if step is None:
        parametrization = {
            's': sigma,
            'scale': np.exp(mu),
        }
        cdf_low = stats.lognorm.cdf(low, **parametrization)
        cdf_high = stats.lognorm.cdf(high, **parametrization)
        values = stats.lognorm.pdf(samples, **parametrization) / (cdf_high - cdf_low)
        values[(samples < low) + (samples > high)] = 0

    else:
        values = (lognormal_cdf(samples + step / 2, mu=mu, sigma=sigma, low=low, high=high) -
                  lognormal_cdf(samples - step / 2, mu=mu, sigma=sigma, low=low, high=high))
    return values
