import numpy as np

from benderopt.rng import RNG

from .utils import ValidationError


def validate_categorical(search_space):
    # error = "Expected a dict with mandatory key 'values' (list) and optional key 'probabilities' (list)"
    if type(search_space) != dict:
        raise ValidationError(message_key="search_space_type")

    search_space = search_space.copy()

    if "values" not in search_space.keys():
        raise ValidationError(message_key="values_mandatory")

    if type(search_space["values"]) != list:
        raise ValidationError(message_key="values_type")

    if "probabilities" in search_space.keys():
        if type(search_space["probabilities"]) != list:
            raise ValidationError(message_key="probabilities_type")

        if len(search_space["probabilities"]) != len(search_space["values"]):
            raise ValidationError(message_key="probabilities_size")

    # Lazy test for summing to 1 (avoiding numerical rounding)
    if "probabilities" in search_space.keys():
        try:
            RNG.choice(range(len(search_space["probabilities"])), p=search_space["probabilities"])
        except ValueError:
            raise ValidationError(message_key="probabilities_sum")

    search_space.setdefault(
        "probabilities", list(np.ones(len(search_space["values"])) / len(search_space["values"]))
    )

    return search_space


def validate_categorical_value(value, values, **kwargs):
    test = True
    if value not in values:
        test = False
    return test
