import numpy as np


def validate_normal(search_space):
    # error = "Expected a type dict with mandatory keys : [mu, sigma] and optional key log  or step"
    if type(search_space) != dict:
        raise ValueError("Search space must be a dict.")

    search_space = search_space.copy()

    if "mu" not in search_space.keys() or type(search_space["mu"]) not in (int, float):
        print(search_space)
        raise ValueError("key 'mu' (loc of normal distribution) is mandatory")

    if "sigma" not in search_space.keys() or type(search_space["sigma"]) not in (int, float):
        raise ValueError("key 'sigma' (scale of normal distribution) is mandatory")

    if "low" in search_space.keys():
        if type(search_space["low"]) not in (int, float):
            raise ValueError("'low' bound must be a float or int")

    if "high" in search_space.keys():
        if type(search_space["high"]) not in (int, float):
            raise ValueError("'high' bound must be a float or int")

    if "high" in search_space.keys() and "low" in search_space.keys():
        if search_space["high"] <= search_space["low"]:
            raise ValueError("'low' must be < 'high'")

    search_space.setdefault("low", -np.inf)
    search_space.setdefault("high", np.inf)

    if "step" in search_space.keys():
        if search_space["step"] and type(search_space["step"]) not in (int, float):
            raise ValueError("'step' must be a float or int.")
        if search_space["step"] and search_space["step"] >= search_space["high"]:
            raise ValueError("Step must be strictly lower than high bound.")

    search_space.setdefault("step", None)

    return search_space


def validate_normal_value(value, low, high, **kwargs):
    test = True
    if value < low:
        test = False
    elif value > high:
        test = False
    return test
