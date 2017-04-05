import numpy as np


def validate_lognormal(search_space):
    # error = "Expected a type dict with mandatory keys : [mu, sigma] and optional key log  or step"
    if type(search_space) != dict:
        raise ValueError("Search space must be a dict.")

    search_space = search_space.copy()

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
        if search_space["high"] <= search_space["low"]:
            raise ValueError("low <= high")

    if search_space.get("base") and type(search_space.get("base")) not in (float, int,):
        raise ValueError

    search_space.setdefault("low", -np.inf)
    search_space.setdefault("high", -np.inf)
    search_space.setdefault("step", None)
    search_space.setdefault("base", 10)

    return search_space


def validate_lognormal_value(value, low, high, base, **kwargs):
    test = True
    low = base ** low
    high = base ** high
    if value < low:
        test = False
    elif value > high:
        test = False
    return test
