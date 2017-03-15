import numpy as np
from numpy import random
from scipy import stats


def generate_sample_uniform(min, max, log=False, step=None):
    sample = random.uniform(low=min,
                            high=max)
    if log:
        sample = np.exp(sample)
    if step:
        sample = step * round(sample / step)
    return sample


def generate_sample_normal(mu, sigma, min=None, max=None, log=False, step=None, max_retry=50):
    # Draw a sample which fit between min and max (if they are given)
    for _ in range(max_retry):
        sample = random.normal(loc=mu, scale=sigma)
        if min and sample < min:
            continue
        if max and sample > max:
            continue
        break
    else:
        raise ValueError("No sample could be drawn in given bounds with max_retry {}".format(
            max_retry))

    if log:
        sample = np.exp(sample)
    if step:
        sample = step * round(sample / step)
    return sample


def generate_sample_categorical(values, weights):
    return random.choice(values, p=weights)


def categorical_logpdf(sample, values, weights):
    return np.log(weights[values.find(sample)])


def generate_sample_gaussian_mixture(mus,
                                     sigmas,
                                     weights=None,
                                     min=None,
                                     max=None,
                                     log=False,
                                     step=None,
                                     max_retry=50):
    """ Generate a random sample according to a mixture of gaussians."""

    number_of_gaussian = len(mus)
    selected_gaussian = random.choice(range(number_of_gaussian), p=weights)

    return generate_sample_normal(mu=mus[selected_gaussian],
                                  sigma=sigmas[selected_gaussian],
                                  min=min,
                                  max=max,
                                  step=step,
                                  log=log,
                                  max_retry=max_retry)


def normal_pdf(sample,
               mu,
               sigma,
               min=None,
               max=None,
               log=False):
    distribution = stats.norm if not log else stats.lognorm

    value = distribution.pdf(sample, loc=mu, scale=sigma)
    # rescale if needed
    value /= (distribution.cdf(max if max else np.inf, loc=mu, scale=sigma) -
              distribution.cdf(min if min else -np.inf, loc=mu, scale=sigma))
    return value


def gaussian_mixture_logpdf(sample,
                            mus,
                            sigmas,
                            weights=None,
                            min=None,
                            max=None,
                            log=False):
    if weights is None:
        weights = np.ones(len(mus)) / len(mus)
    return np.log(np.sum([normal_pdf(mu=mu, sigma=sigma, min=min, max=max, log=log) * weights
                          for mu, sigma, weight in zip(mus, sigmas, weights)]))


sample_generators = {
    "uniform": generate_sample_uniform,
    "normal": generate_sample_normal,
    "categorical": generate_sample_categorical,
    "gaussian_mixture": generate_sample_gaussian_mixture,
}
