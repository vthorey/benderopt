"""Module to validate a observations list"""
import numpy as np


def validate_categorical(value, search_space):
    test = True
    if value not in search_space["values"]:
        test = False
    return test


def validate_normal(value, search_space):
    test = True
    if not search_space.get("log", False):
        if value < search_space.get("low", -np.inf):
            test = False
        elif value > search_space.get("high", np.inf):
            test = False
    else:
        if value < search_space.get("low", -np.inf):
            test = False
        elif value > search_space.get("high", np.inf):
            test = False
    return test


def validate_uniform(value, search_space):
    test = True
    if value < search_space.get("low", -np.inf):
        test = False
    elif value > search_space.get("high", np.inf):
        test = False

    return test


def validate_gaussian_mixture(value, search_space):
    test = True
    if value < search_space.get("low", -np.inf):
        test = False
    elif value > search_space.get("high", np.inf):
        test = False
    return test


is_parameter_value_valid = {
    "categorical": validate_categorical,
    "normal": validate_normal,
    "uniform": validate_uniform,
    "gaussian_mixture": validate_gaussian_mixture,
}
