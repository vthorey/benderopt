import numpy as np

from benderopt.rng import RNG


def generate_samples_categorical(values, probabilities, size=1):
    """Generate sample for categorical data with probability probabilities."""
    return RNG.choice(values, p=probabilities, size=size)


def categorical_pdf(samples, values, probabilities):
    """Evaluate categorical log probability density function for each samples."""
    converter = dict(zip(values, probabilities))
    return np.array([converter[sample] for sample in samples])
