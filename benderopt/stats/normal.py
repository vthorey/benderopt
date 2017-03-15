import numpy as np
from numpy import random
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
    samples = np.ones(size) * np.nan
    nans_locations = np.where(np.isnan(samples))[0]
    for _ in range(max_retry):
        samples[nans_locations] = random.normal(loc=mu, scale=sigma, size=len(nans_locations))
        samples[(samples < low) + (samples > high)] = np.nan
        nans_locations = np.where(np.isnan(samples))[0]
        if len(nans_locations) == 0:
            break
    else:
        raise ValueError("No sample could be drawn in given bounds with max_retry {}".format(
            max_retry))

    if log:
        samples = np.exp(samples)

    if step:
        samples = step * round(samples / step)

    return samples


def normal_cdf(samples,
               mu,
               sigma,
               low,
               high,
               log=False):
    """Evaluate (log)(truncated)normal cumulated density function for each samples."""
    distribution = stats.norm if not log else stats.lognorm

    values = distribution.cdf(np.clip(samples, a_min=None, a_max=max), loc=mu, scale=sigma)

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
