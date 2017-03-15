def validate_categorical(search_space):
    # error = "Expected a dict with mandatory key 'values' (list) and optional key 'weight' (list)"
    if type(search_space) != dict:
        raise ValueError
    elif "values" not in search_space.keys() or type(search_space['values']) != list:
        raise ValueError
    elif "weight" in search_space.keys() and (type(search_space['weight']) != list or
                                              len(search_space['weight']) != len(search_space['values']) or
                                              sum(search_space['weight']) != 1):
        raise ValueError
    return True


def validate_normal(search_space):
    # error = "Expected a type dict with mandatory keys : [mu, sigma] and optional key log  or step"

    if type(search_space) != dict:
        raise ValueError

    elif "mu" not in search_space.keys() or type(search_space["mu"]) not in (int, float):
        raise ValueError

    elif "sigma" not in search_space.keys() or type(search_space["sigma"]) not in (int, float):
        raise ValueError

    elif "log" in search_space.keys():
        if type(search_space["log"]) not in (bool,):
            raise ValueError

    elif "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError
    return True


def validate_uniform(search_space):
    # error = "Expected a type dict with mandatory keys : [min, max] and optional key [log]"

    if type(search_space) != dict:
        raise ValueError

    elif "min" not in search_space.keys() or type(search_space["min"]) not in (int, float):
        raise ValueError

    elif "max" not in search_space.keys() or type(search_space["max"]) not in (int, float):
        raise ValueError

    elif "log" in search_space.keys():
        if type(search_space["log"]) not in (bool,):
            raise ValueError

    elif "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValueError
    return True


is_search_space_valid = {
    "categorical": validate_categorical,
    "normal": validate_normal,
    "uniform": validate_uniform,
}
