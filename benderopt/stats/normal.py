import numpy as np
from scipy import stats


def generate_samples_normal(mu,
                            sigma,
                            low,
                            high,
                            log=False,
                            step=None,
                            size=1,
                            max_retry=50):
    """Generate sample for (log)(truncated)(discrete)normal density."""

    # Draw a samples which fit between low and high (if they are given)
    a, b = (low - mu) / sigma, (high - mu) / sigma
    samples = stats.truncnorm.rvs(a=a, b=b, size=size, loc=mu, scale=sigma)

    if log:
        samples = np.exp(samples)

    if step:
        samples = step * np.round(samples / step)

    return samples


def normal_cdf(samples,
               mu,
               sigma,
               low,
               high,
               log=False):
    """Evaluate (log)(truncated)normal cumulated density function for each samples."""
    distribution = stats.norm if not log else stats.lognorm
    values = distribution.cdf(np.clip(samples, a_min=None, a_max=high), loc=mu, scale=sigma)

    values -= distribution.cdf(low, loc=mu, scale=sigma)
    values = np.clip(values, a_min=0, a_max=None)

    values /= (distribution.cdf(high, loc=mu, scale=sigma) -
               distribution.cdf(low, loc=mu, scale=sigma))

    return values


def normal_pdf(samples,
               mu,
               sigma,
               low,
               high,
               log=False,
               step=None):
    """Evaluate (log)(truncated)(discrete)normal probability density function for each sample."""
    values = None
    if step is None:
        distribution = stats.norm if not log else stats.lognorm

        values = distribution.pdf(samples, loc=mu, scale=sigma)

        # rescale if needed
        values /= (distribution.cdf(high, loc=mu, scale=sigma) -
                   distribution.cdf(low, loc=mu, scale=sigma))
        values[(samples < low) + (samples > high)] = 0
    else:
        values = (normal_cdf(samples + step / 2, mu=mu, sigma=sigma, low=low, high=high, log=log) -
                  normal_cdf(samples - step / 2, mu=mu, sigma=sigma, low=low, high=high, log=log))
    return values
