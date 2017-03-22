from .normal import generate_samples_normal, normal_pdf
from .uniform import generate_samples_uniform, uniform_pdf
from .categorical import generate_samples_categorical, categorical_pdf
from .mixture import generate_samples_mixture, mixture_pdf

sample_generators = {
    "uniform": generate_samples_uniform,
    "normal": generate_samples_normal,
    "categorical": generate_samples_categorical,
    "mixture": generate_samples_mixture,
}

probability_density_function = {
    "categorical": categorical_pdf,
    "normal": normal_pdf,
    "uniform": uniform_pdf,
    "mixture": mixture_pdf,
}


__all__ = [
    "sample_generators",
    "probability_density_function"
]
