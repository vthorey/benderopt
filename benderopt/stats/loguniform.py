import numpy as np

from benderopt.rng import RNG
from benderopt.utils import logb


def generate_samples_loguniform(low, low_log, high_log, step, base, size=1, **kwargs):
    """Generate sample for (discrete)uniform density."""
    samples = base ** (RNG.uniform(low=low_log, high=high_log, size=size))
    if step:
        samples = step * np.floor((samples - low) / step) + low
    return samples


def loguniform_cdf(samples, low, low_log, high, high_log, base):
    """Evaluate (truncated)(discrete)normal cumulated probability density function for each sample.

    Integral of below pdf between base ** low and sample
    """
    values = (logb(samples, base) - low_log) / (high_log - low_log)
    values[(samples < low) + (samples >= high)] = 0
    return values


def loguniform_pdf(samples, low, low_log, high, high_log, base, step):
    """Evaluate (truncated)(discrete)normal probability density function for each sample.

    https://onlinecourses.science.psu.edu/stat414/node/157
    Definition. Let X be a continuous random variable with generic probability density function
     (x) defined over the support v(low) <= x < v(high). And, let Y = u(X) be an invertible function of X
     with inverse function X = v(Y). Then, using the change-of-variable technique,*
     the probability density function of Y is:

    fY(y) = fX(v(y)) × |v′(y)|

    defined over the support low <= y < high.

    here y = base ** x
         x = ln(y) / ln(base)
    fX(x) = 1 / (v(high) - v(low))

    fY(y) = (1 / (v(high) - v(low))) * (1 / (y * np.log(base))
    for low <= y < high

    Visual proof:

    import matplotlib.pyplot as plt
    low = 1
    high = 3
    base = 10
    size = 1000000
    samples = generate_samples_loguniform(low=low, high=high, step=None, base=base, size=size)
    t = np.arange(5, 2000, 0.1)
    pdf = loguniform_pdf(t, low=low, high=high, base=base, step=None)
    plt.hist(samples, bins=5000, normed=True)
    plt.plot(t, pdf, color='r')
    plt.show()
    """
    if step is None:
        values = 1 / ((high_log - low_log) * samples * np.log(base))
    else:
        values = loguniform_cdf(
            samples + step, low=low, low_log=low_log, high=high, high_log=high_log, base=base
        ) - loguniform_cdf(
            samples, low=low, low_log=low_log, high=high, high_log=high_log, base=base
        )
    values[(samples < low) + (samples >= high)] = 0
    return values
