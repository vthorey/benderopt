import numpy as np
from numpy import random


def generate_samples_loguniform(low, high, step=None, base=10, size=1):
    """Generate sample for (discrete)uniform density."""
    samples = base ** (random.uniform(low=low, high=high, size=size))
    if step:
        samples = step * np.round(samples / step)
    return samples


def loguniform_pdf(samples, low, high, base=10, step=None):
    """Evaluate (truncated)(discrete)normal probability density function for each sample."""
    values = np.ones(len(samples)) * 1 / (base ** high - base ** low)
    values[samples < base ** low + samples > base ** high] = 0
    return values
