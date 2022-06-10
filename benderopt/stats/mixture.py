import numpy as np

from benderopt.rng import RNG

from .categorical import categorical_pdf, generate_samples_categorical
from .lognormal import generate_samples_lognormal, lognormal_pdf
from .loguniform import generate_samples_loguniform, loguniform_pdf
from .normal import generate_samples_normal, normal_pdf
from .uniform import generate_samples_uniform, uniform_pdf

sample_generators_base = {
    "uniform": generate_samples_uniform,
    "normal": generate_samples_normal,
    "loguniform": generate_samples_loguniform,
    "lognormal": generate_samples_lognormal,
    "categorical": generate_samples_categorical,
}

probability_density_function_base = {
    "categorical": categorical_pdf,
    "normal": normal_pdf,
    "uniform": uniform_pdf,
    "lognormal": lognormal_pdf,
    "loguniform": loguniform_pdf,
}


def generate_samples_mixture(parameters, weights, size):
    selected_parameter = RNG.choice(range(len(parameters)), p=weights, size=size)
    return np.concatenate(
        [
            sample_generators_base[parameter["category"]](
                size=np.sum(selected_parameter == i), **parameter["search_space"]
            )
            for i, parameter in enumerate(parameters)
        ]
    )


def mixture_pdf(samples, parameters, weights):
    value = np.sum(
        [
            probability_density_function_base[parameter["category"]](
                samples=samples, **parameter["search_space"]
            )
            * weight
            for parameter, weight in zip(parameters, weights)
        ],
        axis=0,
    )
    return value
