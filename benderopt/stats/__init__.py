from .normal import generate_samples_normal
from .uniform import generate_samples_uniform
from .categorical import generate_samples_categorical, categorical_logpdf
from .gaussian_mixture import generate_samples_gaussian_mixture, gaussian_mixture_logpdf

sample_generators = {
    "uniform": generate_samples_uniform,
    "normal": generate_samples_normal,
    "categorical": generate_samples_categorical,
    "gaussian_mixture": generate_samples_gaussian_mixture,
}


__all__ = [
    "sample_generators",
    "gaussian_mixture_logpdf",
    "categorical_logpdf",
]
