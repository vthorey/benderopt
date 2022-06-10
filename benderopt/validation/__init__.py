from .categorical import validate_categorical, validate_categorical_value
from .lognormal import validate_lognormal, validate_lognormal_value
from .loguniform import validate_loguniform, validate_loguniform_value
from .mixture import validate_mixture
from .normal import validate_normal, validate_normal_value
from .uniform import validate_uniform, validate_uniform_value

validate_search_space = {
    "categorical": validate_categorical,
    "normal": validate_normal,
    "uniform": validate_uniform,
    "lognormal": validate_lognormal,
    "loguniform": validate_loguniform,
    "mixture": validate_mixture,
}

is_parameter_value_valid = {
    "categorical": validate_categorical_value,
    "normal": validate_normal_value,
    "uniform": validate_uniform_value,
    "lognormal": validate_lognormal_value,
    "loguniform": validate_loguniform_value,
    # "mixture": validate_mixture_value,
}

__all__ = ["is_parameter_value_valid", "validate_search_space"]
