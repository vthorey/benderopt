from numpy import random
import numpy as np


def generate_samples_mixture(parameters, weights, size):
    selected_parameter = random.choice(range(len(parameters)),
                                       p=weights,
                                       size=size)
    return np.concatenate([
        parameter.draw(size=np.sum(selected_parameter == i))
        for i, parameter in enumerate(parameters)
    ])


def mixture_pdf(samples, parameters, weights):
    value = np.sum([parameter.pdf(samples) * weight
                    for parameter, weight in zip(parameters, weights)], axis=0)
    return value
