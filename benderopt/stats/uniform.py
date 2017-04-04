import numpy as np
from numpy import random


def generate_samples_uniform(low, high, step, size=1):
    """Generate sample for (discrete)uniform density."""
    samples = random.uniform(low=low,
                             high=high,
                             size=size)
    if step:
        samples = step * np.floor(samples / step)
    return samples


def uniform_pdf(samples, low, high, step):
    """Evaluate (truncated)(discrete)normal probability density function for each sample."""
    values = np.ones(len(samples)) * 1 / (high - low)
    values[(samples < low) + (samples >= high)] = 0
    if step:
        values *= step
    return values
