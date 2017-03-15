from numpy import random
import numpy as np


def generate_samples_categorical(values, weights, size=1):
    """Generate sample for categorical data with probability weights."""
    return random.choice(values, p=weights, size=size)


def categorical_logpdf(samples, values, weights):
    """Evaluate categorical log probability density function for each samples."""
    converter = dict(zip(values, weights))
    return np.log([converter[value] for value in values])
