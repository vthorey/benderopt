import numpy as np
from numpy import random
from .normal import generate_samples_normal, normal_pdf


def generate_samples_gaussian_mixture(mus,
                                      sigmas,
                                      low,
                                      high,
                                      weights=None,
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
                                low=low,
                                high=high,
                                step=step,
                                log=log,
                                size=np.sum(selected_gaussian == i),
                                max_retry=max_retry)
        for i, (mu, sigma) in enumerate(zip(mus, sigmas))
    ])


def gaussian_mixture_pdf(samples,
                         mus,
                         sigmas,
                         low,
                         high,
                         weights=None,
                         log=False,
                         step=None):
    """Evaluate log (log)(truncated)(discrete) gaussian gaussian_mixture probability density
    function for each sample.
    """
    if weights is None:
        weights = np.ones(len(mus)) / len(mus)

    # Compute pdf as weighted sum of pdfs for each gaussian
    return np.sum([normal_pdf(samples,
                              mu=mu,
                              sigma=sigma,
                              low=low,
                              high=high,
                              log=log,
                              step=step) * weight
                   for mu, sigma, weight in zip(mus, sigmas, weights)], axis=0)
