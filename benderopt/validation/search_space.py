import numpy as np


def validate_categorical(search_space):
    # error = "Expected a dict with mandatory key 'values' (list) and optional key 'probabilities' (list)"
    search_space = search_space.copy()

    if type(search_space) != dict:
        raise ValueError
    if "values" not in search_space.keys() or type(search_space['values']) != list:
        raise ValueError
    if "probabilities" in search_space.keys() and (
            type(search_space['probabilities']) != list or
            len(search_space['probabilities']) != len(search_space['values'])):
        raise ValueError

    # Test that proba sum to 1 but we are lazy and we try directly
    if "probabilities" in search_space.keys():
        np.random.choice(range(len(search_space["probabilities"])),
                         p=search_space["probabilities"])

    if "probabilities" not in search_space.keys():
        number_of_values = len(search_space["values"])
        search_space["probabilities"] = list(np.ones(number_of_values) / number_of_values)

    return search_space


def validate_uniform(search_space):
    # error = "Expected a type dict with mandatory keys : [low, high] and optional key [log]"
    search_space = search_space.copy()

    if type(search_space) != dict:
        raise ValueError

    if "low" not in search_space.keys():
        raise ValueError

    if "high" not in search_space.keys():
        raise ValueError

    if "low" in search_space.keys():
        if type(search_space["low"]) not in (int, float):
            raise ValueError

    if "high" in search_space.keys():
        if type(search_space["high"]) not in (int, float):
            raise ValueError

    if "high" in search_space.keys() and "low" in search_space.keys():
        if search_space["high"] >= search_space["low"]:
            raise ValueError("low <= high")

    if "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError
        if search_space["step"] >= search_space["high"]:
            raise ValueError("Step must be strictly lower than high bound.")

    search_space.setdefault("step", None)

    return search_space


def validate_loguniform(search_space):
    # error = "Expected a type dict with mandatory keys : [low, high] and optional key [log]"
    search_space = search_space.copy()

    if type(search_space) != dict:
        raise ValueError

    if "low" not in search_space.keys():
        raise ValueError

    if "high" not in search_space.keys():
        raise ValueError

    if "low" in search_space.keys():
        if type(search_space["low"]) not in (int, float):
            raise ValueError
        if search_space["low"] <= 0:
            raise ValueError("Low bound must be strictly positive")

    if "high" in search_space.keys():
        if type(search_space["high"]) not in (int, float):
            raise ValueError

    if "high" in search_space.keys() and "low" in search_space.keys():
        if search_space["high"] >= search_space["low"]:
            raise ValueError("low <= high")
        if search_space["high"] <= 0:
            raise ValueError("High bound must be strictly positive")

    if "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError
        if ("base" in search_space.keys() and
                search_space["step"] >= (search_space["base"] ** search_space["high"])):
            raise ValueError("Step must be strictly lower than base ** high.")

    if search_space.get("base") and type(search_space.get("base")) not in (float, int,):
        raise ValueError

    search_space.setdefault("step", None)
    search_space.setdefault("base", 10)

    return search_space


def validate_normal(search_space):
    # error = "Expected a type dict with mandatory keys : [mu, sigma] and optional key log  or step"
    search_space = search_space.copy()

    if type(search_space) != dict:
        raise ValueError

    if "mu" not in search_space.keys() or type(search_space["mu"]) not in (int, float):
        print(search_space)
        raise ValueError

    if "sigma" not in search_space.keys() or type(search_space["sigma"]) not in (int, float):
        raise ValueError

    if "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError

    if "low" in search_space.keys():
        if type(search_space["low"]) not in (int, float):
            raise ValueError

    if "high" in search_space.keys():
        if type(search_space["high"]) not in (int, float):
            raise ValueError

    if "high" in search_space.keys() and "low" in search_space.keys():
        if search_space["high"] >= search_space["low"]:
            raise ValueError("low <= high")

    search_space.setdefault("low", -np.inf)
    search_space.setdefault("high", -np.inf)

    search_space.setdefault("step", None)

    return search_space


def validate_lognormal(search_space):
    # error = "Expected a type dict with mandatory keys : [mu, sigma] and optional key log  or step"
    search_space = search_space.copy()

    if type(search_space) != dict:
        raise ValueError

    if "mu" not in search_space.keys() or type(search_space["mu"]) not in (int, float):
        print(search_space)
        raise ValueError

    if "sigma" not in search_space.keys() or type(search_space["sigma"]) not in (int, float):
        raise ValueError

    if "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError
        if (search_space["step"] >= search_space["high"]):
            raise ValueError("Step must be strictly lower than high.")

    if "low" in search_space.keys():
        if type(search_space["low"]) not in (int, float):
            raise ValueError
        if search_space["low"] <= 0:
            raise ValueError("Low bound must be strictly positive")

    if "high" in search_space.keys():
        if type(search_space["high"]) not in (int, float):
            raise ValueError
        if search_space["high"] <= 0:
            raise ValueError("High bound must be strictly positive")

    if "high" in search_space.keys() and "low" in search_space.keys():
        if search_space["high"] >= search_space["low"]:
            raise ValueError("low <= high")

    if search_space.get("base") and type(search_space.get("base")) not in (float, int,):
        raise ValueError

    search_space.setdefault("low", -np.inf)
    search_space.setdefault("high", -np.inf)
    search_space.setdefault("step", None)
    search_space.setdefault("base", 10)

    return search_space


def validate_mixture(search_space):
    # error = "Expected a type dict with mandatory keys : [low, high] and optional key [log]"
    search_space = search_space.copy()

    if type(search_space) != dict:
        raise ValueError

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

        search_space["parameters"][i]["search_space"] = validate_search_space[parameter["category"]](
            parameter["search_space"])

    if "weights" not in search_space.keys():
        number_of_values = len(search_space["parameters"])
        search_space["probabilities"] = list(np.ones(number_of_values) / number_of_values)

    return search_space


validate_search_space = {
    "categorical": validate_categorical,
    "normal": validate_normal,
    "uniform": validate_uniform,
    "lognormal": validate_lognormal,
    "loguniform": validate_loguniform,
    "mixture": validate_mixture,
}
