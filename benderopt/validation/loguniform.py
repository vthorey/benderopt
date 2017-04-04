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


def validate_loguniform_value(value, low, high, base, **kwargs):
    test = True
    low = base ** low
    high = base ** high
    if value < low:
        test = False
    elif value > high:
        test = False
    return test
