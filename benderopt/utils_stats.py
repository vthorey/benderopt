import numpy as np
from numpy import random


def generate_sample_uniform(scope):
    sample = random.uniform(low=scope["min"],
                            high=scope["max"])
    if scope.get("log", False):
        sample = np.exp(sample)
    if scope.get("step", None):
        sample = scope["step"] * round(sample / scope["step"])
    return sample


def generate_sample_normal(scope):
    sample = random.normal(loc=scope["mu"], scale=scope["sigma"])
    if scope.get("log", False):
        sample = np.exp(sample)
    if scope.get("step", None):
        sample = scope["step"] * round(sample / scope["step"])
    return sample


def generate_sample_categorical(scope):
    return random.choice(scope['values'], p=scope['weight'])


sample_generators = {
    "uniform": generate_sample_uniform,
    "normal": generate_sample_normal,
    "categorical": generate_sample_categorical,
}
