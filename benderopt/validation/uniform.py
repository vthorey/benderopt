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
        if search_space["high"] <= search_space["low"]:
            raise ValueError("low <= high")

    if "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError
        if search_space["step"] >= search_space["high"]:
            raise ValueError("Step must be strictly lower than high bound.")

    search_space.setdefault("step", None)

    return search_space


def validate_uniform_value(value, low, high, **kwargs):
    test = True
    if value < low:
        test = False
    elif value > high:
        test = False
    return test
