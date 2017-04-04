import numpy as np
from numpy import random


def generate_samples_loguniform(low, high, step, base, size=1):
    """Generate sample for (discrete)uniform density."""
    samples = base ** (random.uniform(low=low, high=high, size=size))
    if step:
        samples = step * np.floor(samples / step)
    return samples


def loguniform_cdf(samples, low, high, base):
    """Evaluate (truncated)(discrete)normal cumulated probability density function for each sample.

    Integral of below pdf between base ** low and sample
    """
    samples = ((np.log(samples) / np.log(base)) - low) / (high - low)
    samples[(samples < (base ** low)) + (samples >= (base ** high))] = 0
    return samples


def loguniform_pdf(samples, low, high, base, step):
    """Evaluate (truncated)(discrete)normal probability density function for each sample.

    https://onlinecourses.science.psu.edu/stat414/node/157
    Definition. Let X be a continuous random variable with generic probability density function
     (x) defined over the support low <= x < high. And, let Y = u(X) be an invertible function of X
     with inverse function X = v(Y). Then, using the change-of-variable technique,*
     the probability density function of Y is:

    fY(y) = fX(v(y)) × |v′(y)|

    defined over the support u(low) <= y < u(high).

    here y = base ** x
         x = ln(y) / ln(base)
    fX(x) = 1 / (high - low)

    fY(y) = (1 / (high - low)) * (1 / (y * np.log(base))
    for base ** low <= y < base ** high

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
        values = 1 / ((high - low) * samples * np.log(base))
        values[(samples < (base ** low)) + (samples >= (base ** high))] = 0
    else:
        values = (loguniform_cdf(samples + step, low=low, high=high, base=base) -
                  loguniform_cdf(samples, low=low, high=high, base=base))
    return values
