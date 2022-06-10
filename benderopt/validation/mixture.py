import numpy as np

from ..rng import RNG
from .categorical import validate_categorical, validate_categorical_value
from .lognormal import validate_lognormal, validate_lognormal_value
from .loguniform import validate_loguniform, validate_loguniform_value
from .normal import validate_normal, validate_normal_value
from .uniform import validate_uniform, validate_uniform_value
from .utils import ValidationError, mandatory_key

validate_search_space = {
    "categorical": validate_categorical,
    "normal": validate_normal,
    "uniform": validate_uniform,
    "lognormal": validate_lognormal,
    "loguniform": validate_loguniform,
}

is_parameter_value_valid = {
    "categorical": validate_categorical_value,
    "normal": validate_normal_value,
    "uniform": validate_uniform_value,
    "lognormal": validate_lognormal_value,
    "loguniform": validate_loguniform_value,
}


def validate_mixture(search_space):
    # error = "Expected a type dict with mandatory keys : [low, high] and optional key [log]"
    if type(search_space) != dict:
        raise ValidationError(message_key="search_space_type")

    search_space = search_space.copy()

    if "parameters" not in search_space.keys():
        raise ValidationError(message_key="parameters_mandatory")

    if type(search_space["parameters"]) != list:
        raise ValidationError(message="parameters_type")

    for i, parameter in enumerate(search_space["parameters"]):
        if "category" not in parameter.keys():
            raise ValidationError(message=mandatory_key("category") + " in parameter {}".format(i))
        if parameter["category"] not in validate_search_space.keys():
            raise ValidationError(message="'category' not recognized for parameter {}".format(i))

        if "search_space" not in parameter.keys():
            raise ValidationError(
                message=mandatory_key("search_space") + " in parameter {}".format(i)
            )

        if type(parameter["search_space"]) != dict:
            raise ValidationError(
                message=ValidationError.error_messages["search_space_type"]
                + " in parameter {}".format(i)
            )

        try:
            search_space["parameters"][i]["search_space"] = validate_search_space[
                parameter["category"]
            ](parameter["search_space"])
        except ValidationError as e:
            raise ValidationError(message="{} for parameter {}".format(e.args[0], i))

    if "weights" in search_space.keys():
        if type(search_space["weights"]) != list:
            raise ValidationError(message_key="weights_type")

        if len(search_space["weights"]) != len(search_space["parameters"]):
            raise ValidationError(message_key="weights_size")

    # Lazy test for summing to 1 (avoiding numerical rounding)
    if "weights" in search_space.keys():
        try:
            RNG.choice(range(len(search_space["weights"])), p=search_space["weights"])
        except ValueError:
            raise ValidationError(message_key="weights_sum")

    search_space.setdefault(
        "weights", list(np.ones(len(search_space["parameters"])) / len(search_space["parameters"]))
    )

    return search_space


def validate_mixture_value(value, parameters, **kwargs):
    test = True
    return test
