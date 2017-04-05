def validate_loguniform(search_space):
    # error = "Expected a type dict with mandatory keys : [low, high] and optional key [log]"
    if type(search_space) != dict:
        raise ValueError("Search space must be a dict.")

    search_space = search_space.copy()

    if "low" not in search_space.keys():
        raise ValueError("key 'low' is mandatory")

    if "high" not in search_space.keys():
        raise ValueError("key 'high' is mandatory")

    if type(search_space["low"]) not in (int, float):
        raise ValueError("'low' bound must be a float or int")

    if type(search_space["high"]) not in (int, float):
        raise ValueError("'high' bound must be a float or int")

    if search_space["low"] <= 0:
        raise ValueError("Low bound must be strictly positive")

    if search_space["high"] <= search_space["low"]:
        raise ValueError("'low' must be < 'high'")

    if "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError("'step' must be a float or int.")
        if (search_space["step"] >= (search_space["high"])):
            raise ValueError("'Step' must be < high.")

    if search_space.get("base") and type(search_space.get("base")) not in (float, int,):
        raise ValueError("'base' must be a float or int.")

    search_space.setdefault("step", None)
    search_space.setdefault("base", 10)

    return search_space


def validate_loguniform_value(value, low, high, base, **kwargs):
    test = True
    if value < low:
        test = False
    elif value >= high:
        test = False
    return test
