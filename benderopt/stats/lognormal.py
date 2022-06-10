import numpy as np
from scipy import stats


def generate_samples_lognormal(
    mu_log, sigma_log, low_log, low, high_log, step, base, size=1, **kwargs
):
    """Generate sample for (truncated)(discrete)log10normal density."""

    # Draw a samples which fit between low and high (if they are given)
    a = (low_log - mu_log) / sigma_log
    b = (high_log - mu_log) / sigma_log
    samples = base ** stats.truncnorm.rvs(a=a, b=b, size=size, loc=mu_log, scale=sigma_log)

    if step and low != -np.inf:
        samples = step * np.floor((samples - low) / step) + low
    elif step and low == -np.inf:
        samples = step * np.floor(samples / step)
    return samples


def lognormal_cdf(samples, mu_log, sigma_log, low, high, base):
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
    parametrization = {"s": sigma_log * np.log(base), "scale": base**mu_log}
    cdf_low = stats.lognorm.cdf(low, **parametrization)
    cdf_high = stats.lognorm.cdf(high, **parametrization)
    values = (stats.lognorm.cdf(samples, **parametrization) - cdf_low) / (cdf_high - cdf_low)
    values[(samples < low)] = 0
    values[(samples >= high)] = 1
    return values


def lognormal_pdf(samples, mu, mu_log, sigma, sigma_log, low, low_log, high, high_log, base, step):
    """Evaluate (truncated)(discrete)normal probability density function for each sample."""
    values = None
    if step is None:
        parametrization = {"s": sigma_log * np.log(base), "scale": base**mu_log}
        cdf_low = stats.lognorm.cdf(low, **parametrization)
        cdf_high = stats.lognorm.cdf(high, **parametrization)
        values = stats.lognorm.pdf(samples, **parametrization) / (cdf_high - cdf_low)

    else:
        values = lognormal_cdf(
            samples + step, mu_log=mu_log, sigma_log=sigma_log, low=low, high=high, base=base
        ) - lognormal_cdf(
            samples, mu_log=mu_log, sigma_log=sigma_log, low=low, high=high, base=base
        )

    values[(samples < low) + (samples >= high)] = 0
    return values
