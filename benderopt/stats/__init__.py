from .normal import generate_samples_normal, normal_pdf
from .uniform import generate_samples_uniform
from .categorical import generate_samples_categorical, categorical_pdf
from .gaussian_mixture import generate_samples_gaussian_mixture, gaussian_mixture_pdf

sample_generators = {
    "uniform": generate_samples_uniform,
    "normal": generate_samples_normal,
    "categorical": generate_samples_categorical,
    "gaussian_mixture": generate_samples_gaussian_mixture,
}

probability_density_function = {
    "gaussian_mixture": gaussian_mixture_pdf,
    "categorical": categorical_pdf,
    "normal": normal_pdf,
}


__all__ = [
    "sample_generators",
    "probability_density_function"
]
