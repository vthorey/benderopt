import numpy as np
from numpy import random
from scipy import stats


def generate_samples_uniform(min, max, log=False, step=None, size=1):
    """Generate sample for (log)(discrete)uniform density."""
    samples = random.uniform(low=min,
                             high=max,
                             size=size)
    if log:
        samples = np.exp(samples)
    if step:
        samples = step * round(samples / step)
    return samples


def generate_samples_normal(mu,
                            sigma,
                            min=None,
                            max=None,
                            log=False,
                            step=None,
                            size=1,
                            max_retry=50):
    """Generate sample for (log)(truncated)(discrete)normal density."""

    # Draw a samples which fit between min and max (if they are given)
    samples = np.ones(size) * np.nan
    nans_locations = np.where(np.isnan(samples))[0]
    min = min if min is not None else -np.inf
    max = max if max is not None else np.inf
    for _ in range(max_retry):
        samples[nans_locations] = random.normal(loc=mu, scale=sigma, size=len(nans_locations))
        samples[(samples < min) * (samples > max)] = np.nan
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


def generate_samples_categorical(values, weights, size=1):
    """Generate sample for categorical data with probability weights."""
    return random.choice(values, p=weights, size=size)


def generate_samples_gaussian_mixture(mus,
                                      sigmas,
                                      weights=None,
                                      min=None,
                                      max=None,
                                      log=False,
                                      step=None,
                                      size=1,
                                      max_retry=50):
    """ Generate a random sample according to a (log)(truncated)(discrete)mixture of gaussians."""

    number_of_gaussian = len(mus)
    selected_gaussian = random.choice(range(number_of_gaussian), p=weights, size=size)
    return np.concatenate([
        generate_samples_normal(mu=mu,
                                sigma=sigma,
                                min=min,
                                max=max,
                                step=step,
                                log=log,
                                size=np.sum(selected_gaussian == i),
                                max_retry=max_retry)
        for i, (mu, sigma) in enumerate(zip(mus, sigmas))
    ])


sample_generators = {
    "uniform": generate_samples_uniform,
    "normal": generate_samples_normal,
    "categorical": generate_samples_categorical,
    "gaussian_mixture": generate_samples_gaussian_mixture,
}


def categorical_logpdf(samples, values, weights):
    """Evaluate categorical log probability density function for each samples."""
    converter = dict(zip(values, weights))
    return np.log([converter[value] for value in values])


def normal_cdf(samples,
               mu,
               sigma,
               min=None,
               max=None,
               log=False):
    """Evaluate (log)(truncated)normal cumulated density function for each samples."""
    distribution = stats.norm if not log else stats.lognorm

    values = distribution.cdf(np.clip(samples, a_min=None, a_max=max), loc=mu, scale=sigma)

    if min:
        values -= distribution.cdf(min, loc=mu, scale=sigma)
        values = np.clip(values, a_min=0, a_max=None)

    values /= (distribution.cdf(max if max is not None else np.inf, loc=mu, scale=sigma) -
               distribution.cdf(min if min is not None else -np.inf, loc=mu, scale=sigma))

    return values


def normal_pdf(samples,
               mu,
               sigma,
               min=None,
               max=None,
               log=False,
               step=None):
    """Evaluate (log)(truncated)(discrete)normal probability density function for each sample."""
    values = None
    if step is None:
        distribution = stats.norm if not log else stats.lognorm

        values = distribution.pdf(samples, loc=mu, scale=sigma)

        # rescale if needed
        values /= (distribution.cdf(max if max is not None else np.inf, loc=mu, scale=sigma) -
                   distribution.cdf(min if min is not None else -np.inf, loc=mu, scale=sigma))
        values[samples < (min if min is not None else -np.inf)] = 0
        values[samples > (max if max is not None else np.inf)] = 0
    else:
        values = (normal_cdf(samples + step / 2, mu=mu, sigma=sigma, min=min, max=max, log=log) -
                  normal_cdf(samples - step / 2, mu=mu, sigma=sigma, min=min, max=max, log=log))
    return values


def gaussian_mixture_logpdf(samples,
                            mus,
                            sigmas,
                            weights=None,
                            min=None,
                            max=None,
                            log=False,
                            step=None):
    """Evaluate log (log)(truncated)(discrete) gaussian gaussian_mixture probability density
    function for each sample.
    """
    if weights is None:
        weights = np.ones(len(mus)) / len(mus)

    # Compute pdf as weighted sum of pdfs for each gaussian
    pdf = np.sum([normal_pdf(samples,
                             mu=mu,
                             sigma=sigma,
                             min=min,
                             max=max,
                             log=log,
                             step=step) * weight
                  for mu, sigma, weight in zip(mus, sigmas, weights)], axis=0)

    # return lof_pdf taking care of numerical stability
    return np.log(np.clip(pdf, a_min=1e-30, a_max=None))
