import numpy as np
from numpy import random


def generate_samples_loguniform(low, high, step=None, size=1):
    """Generate sample for (discrete)uniform density."""
    samples = np.exp(random.uniform(low=low,
                                    high=high,
                                    size=size))
    if step:
        samples = step * np.round(samples / step)
    return samples


def loguniform_pdf(samples,
                   low,
                   high,
                   step=None):
    """Evaluate (truncated)(discrete)normal probability density function for each sample."""
    values = np.ones(len(samples)) * 1 / (np.exp(high) - np.exp(low))
    values[samples < np.exp(low) + samples > np.exp(high)] = 0
    return values
