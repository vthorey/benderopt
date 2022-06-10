import numpy as np

from benderopt.validation.utils import ValidationError


def validate_normal(search_space):
    # error = "Expected a type dict with mandatory keys : [mu, sigma] and optional key log  or step"
    if type(search_space) != dict:
        raise ValidationError(message_key="search_space_type")

    search_space = search_space.copy()

    if "mu" not in search_space.keys():
        raise ValidationError(message_key="mu_mandatory")

    if type(search_space["mu"]) not in (int, float):
        raise ValidationError(message_key="mu_type")

    if "sigma" not in search_space.keys():
        raise ValidationError(message_key="sigma_mandatory")

    if type(search_space["sigma"]) not in (int, float):
        raise ValidationError(message_key="sigma_type")

    if "low" in search_space.keys():
        if type(search_space["low"]) not in (int, float):
            raise ValidationError(message_key="low_type")

    if "high" in search_space.keys():
        if type(search_space["high"]) not in (int, float):
            raise ValidationError(message_key="high_type")

    if "high" in search_space.keys() and "low" in search_space.keys():
        if search_space["high"] <= search_space["low"]:
            raise ValidationError(message_key="high_inferior_low")

    search_space.setdefault("low", -np.inf)
    search_space.setdefault("high", np.inf)

    if "step" in search_space.keys():
        if search_space["step"] and type(search_space["step"]) not in (int, float):
            raise ValidationError(message_key="step_type")
        if search_space["step"] and search_space["step"] >= max(
            [np.abs(search_space["high"]), np.abs(search_space["low"])]
        ):
            raise ValidationError(message_key="high_inferior_step")

    search_space.setdefault("step", None)

    return search_space


def validate_normal_value(value, low, high, **kwargs):
    test = True
    if value < low:
        test = False
    elif value > high:
        test = False
    return test
