from .utils import error_messages


def validate_uniform(search_space):
    # error = "Expected a type dict with mandatory keys : [low, high] and optional key [log]"
    if type(search_space) != dict:
        raise ValueError("Search space must be a dict.")

    search_space = search_space.copy()

    if "low" not in search_space.keys():
        raise ValueError(error_messages["low_mandatory"])

    if "high" not in search_space.keys():
        raise ValueError(error_messages["high_mandatory"])

    if type(search_space["low"]) not in (int, float):
        raise ValueError("'low' bound must be a float or int")

    if type(search_space["high"]) not in (int, float):
        raise ValueError("'high' bound must be a float or int")

    if search_space["high"] <= search_space["low"]:
        raise ValueError("'low' must be < 'high'")

    if "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError(error_messages["step_type"])
        if search_space["step"] >= search_space["high"]:
            raise ValueError("Step must be strictly lower than high bound.")

    search_space.setdefault("step", None)

    return search_space


def validate_uniform_value(value, low, high, **kwargs):
    test = True
    if value < low:
        test = False
    elif value >= high:
        test = False
    return test
