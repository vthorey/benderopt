import numpy as np

from benderopt.rng import RNG


def generate_samples_uniform(low, high, step, size=1):
    """Generate sample for (discrete)uniform density."""
    samples = RNG.uniform(low=low, high=high, size=size)
    if step:
        samples = step * np.floor((samples - low) / step) + low
    return samples


def uniform_pdf(samples, low, high, step):
    """Evaluate (truncated)(discrete)normal probability density function for each sample."""
    values = np.ones(len(samples)) * 1 / (high - low)

    if step:
        values *= step

    values[(samples < low) + (samples >= high)] = 0
    return values
