import numpy as np

from benderopt.utils import logb

from .utils import ValidationError


def validate_loguniform(search_space):
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

    if search_space["low"] <= 0:
        raise ValidationError(message_key="low_inferior_0")

    if search_space["high"] <= search_space["low"]:
        raise ValidationError(message_key="high_inferior_low")

    if "step" in search_space.keys():
        if search_space["step"] and type(search_space["step"]) not in (int, float):
            raise ValidationError(message_key="step_type")
        if search_space["step"] and search_space["step"] >= max(
            [np.abs(search_space["high"]), np.abs(search_space["low"])]
        ):
            raise ValidationError(message_key="high_inferior_step")

    if search_space.get("base") and type(search_space.get("base")) not in (float, int):
        raise ValidationError(message_key="base_type")

    search_space.setdefault("step", None)
    search_space.setdefault("base", 10)

    with np.errstate(divide="ignore"):  # Low can be 0
        search_space["low_log"] = logb(search_space["low"], search_space["base"])
    search_space["high_log"] = logb(search_space["high"], search_space["base"])

    return search_space


def validate_loguniform_value(value, low, high, base, **kwargs):
    test = True
    if value < low:
        test = False
    elif value >= high:
        test = False
    return test
