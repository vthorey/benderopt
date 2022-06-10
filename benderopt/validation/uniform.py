import numpy as np

from .utils import ValidationError


def validate_uniform(search_space):
    # error = "Expected a type dict with mandatory keys : [low, high] and optional key [log]"
    if type(search_space) != dict:
        raise ValidationError(message_key="search_space_type")

    search_space = search_space.copy()

    if "low" not in search_space.keys():
        raise ValidationError(message_key="low_mandatory")

    if "high" not in search_space.keys():
        raise ValidationError(message_key="high_mandatory")

    if type(search_space["low"]) not in (int, float):
        raise ValidationError(message_key="low_type")

    if type(search_space["high"]) not in (int, float):
        raise ValidationError(message_key="high_type")

    if search_space["high"] <= search_space["low"]:
        raise ValidationError(message_key="high_inferior_low")

    if "step" in search_space.keys():
        if type(search_space["step"]) not in (int, float):
            raise ValidationError(message_key="step_type")
        if search_space["step"] >= max([np.abs(search_space["high"]), np.abs(search_space["low"])]):
            raise ValidationError(message_key="high_inferior_step")

    search_space.setdefault("step", None)

    return search_space


def validate_uniform_value(value, low, high, **kwargs):
    test = True
    if value < low:
        test = False
    elif value >= high:
        test = False
    return test
