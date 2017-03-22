import numpy as np
from numpy import random


def generate_samples_uniform(low, high, log=False, step=None, size=1):
    """Generate sample for (log)(discrete)uniform density."""
    samples = random.uniform(low=low,
                             high=high,
                             size=size)
    if log:
        samples = np.exp(samples)
    if step:
        samples = step * np.round(samples / step)
    return samples


def uniform_pdf(samples,
                low,
                high,
                log=False,
                step=None):
    """Evaluate (log)(truncated)(discrete)normal probability density function for each sample."""
    return 1 / (high - low)
