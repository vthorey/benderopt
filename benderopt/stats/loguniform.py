import numpy as np
from numpy import random


def generate_samples_loguniform(low, high, step, base, size=1):
    """Generate sample for (discrete)uniform density."""
    samples = base ** (random.uniform(low=low, high=high, size=size))
    if step:
        samples = step * np.round(samples / step)
    return samples


def loguniform_pdf(samples, low, high, base, step):
    """Evaluate (truncated)(discrete)normal probability density function for each sample."""
    values = np.ones(len(samples)) * 1 / (base ** high - base ** low)
    values[samples < base ** low + samples > base ** high] = 0
    return values
