import numpy as np
from .categorical import validate_categorical, validate_categorical_value
from .uniform import validate_uniform, validate_uniform_value
from .loguniform import validate_loguniform, validate_loguniform_value
from .normal import validate_normal, validate_normal_value
from .lognormal import validate_lognormal, validate_lognormal_value

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
        raise ValueError("Search space must be a dict.")

    search_space = search_space.copy()

    if "parameters" not in search_space.keys():
        raise ValueError

    if type(search_space["parameters"]) != list:
        raise ValueError

    for i, parameter in enumerate(search_space["parameters"]):
        if ("category" not in parameter.keys()) or (parameter["category"] not in ("normal",
                                                                                  "uniform",
                                                                                  "categorical")):
            raise ValueError

        if "search_space" not in parameter.keys() or type(parameter["search_space"]) != dict:
            raise ValueError

        search_space["parameters"][i]["search_space"] = validate_search_space[
            parameter["category"]](parameter["search_space"])

    if "weights" not in search_space.keys():
        number_of_values = len(search_space["parameters"])
        search_space["probabilities"] = list(np.ones(number_of_values) / number_of_values)

    return search_space


def validate_mixture_value(value, parameters, **kwargs):

    return test
