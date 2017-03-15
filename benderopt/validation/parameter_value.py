"""Module to validate a observations list"""


def validate_categorical(value, search_space):
    test = True
    if value not in search_space["values"]:
        test = False
    return test


def validate_normal(value, search_space):
    test = True
    return test


def validate_uniform(value, search_space):
    test = True
    if value < search_space["low"]:
        test = False
    elif value > search_space["high"]:
        test = False

    return test


is_parameter_value_valid = {
    "categorical": validate_categorical,
    "normal": validate_normal,
    "uniform": validate_uniform,
}
