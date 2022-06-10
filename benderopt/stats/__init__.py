from .categorical import categorical_pdf, generate_samples_categorical
from .lognormal import generate_samples_lognormal, lognormal_pdf
from .loguniform import generate_samples_loguniform, loguniform_pdf
from .mixture import generate_samples_mixture, mixture_pdf
from .normal import generate_samples_normal, normal_pdf
from .uniform import generate_samples_uniform, uniform_pdf

sample_generators = {
    "uniform": generate_samples_uniform,
    "loguniform": generate_samples_loguniform,
    "normal": generate_samples_normal,
    "lognormal": generate_samples_lognormal,
    "categorical": generate_samples_categorical,
    "mixture": generate_samples_mixture,
}

probability_density_function = {
    "categorical": categorical_pdf,
    "normal": normal_pdf,
    "lognormal": lognormal_pdf,
    "uniform": uniform_pdf,
    "loguniform": loguniform_pdf,
    "mixture": mixture_pdf,
}


__all__ = ["sample_generators", "probability_density_function"]
