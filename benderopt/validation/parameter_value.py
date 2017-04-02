"""Module to validate a observations list"""
import numpy as np


def validate_categorical(value, values, **kwargs):
    test = True
    if value not in values:
        test = False
    return test


def validate_normal(value, low, high, **kwargs):
    test = True
    if value < low:
        test = False
    elif value > high:
        test = False
    return test


def validate_lognormal(value, low, high, **kwargs):
    test = True
    low = np.exp(low)
    high = np.exp(high)
    if value < low:
        test = False
    elif value > high:
        test = False
    return test


def validate_uniform(value, low, high, **kwargs):
    test = True
    if value < low:
        test = False
    elif value > high:
        test = False
    return test


def validate_loguniform(value, low, high, **kwargs):
    test = True
    low = np.exp(low)
    high = np.exp(high)
    if value < low:
        test = False
    elif value > high:
        test = False
    return test


def validate_mixture(value, parameters, **kwargs):
    test = False
    for parameter in parameters:
        if parameter.check_value(value):
            test = True
            break

    return test


is_parameter_value_valid = {
    "categorical": validate_categorical,
    "normal": validate_normal,
    "uniform": validate_uniform,
    "lognormal": validate_lognormal,
    "loguniform": validate_loguniform,
    "mixture": validate_mixture,
}
