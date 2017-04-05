import numpy as np


def validate_categorical(search_space):
    # error = "Expected a dict with mandatory key 'values' (list) and optional key 'probabilities' (list)"

    if type(search_space) != dict:
        raise ValueError("Search space must be a dict.")

    search_space = search_space.copy()

    if "values" not in search_space.keys() or type(search_space['values']) != list:
        raise ValueError("search_space['values'] must be a list")

    if "probabilities" in search_space.keys() and (
            type(search_space['probabilities']) != list or
            len(search_space['probabilities']) != len(search_space['values'])):
        raise ValueError("search_space['probabilities'] must be a list summing to 1.")

    # Test that proba sum to 1 but we are lazy and we try directly
    if "probabilities" in search_space.keys():
        try:
            np.random.choice(range(len(search_space["probabilities"])),
                             p=search_space["probabilities"])
        except Exception:
            raise ValueError("search_space['probabilities'] must be a list summing to 1.")

    if "probabilities" not in search_space.keys():
        number_of_values = len(search_space["values"])
        search_space["probabilities"] = list(np.ones(number_of_values) / number_of_values)

    return search_space


def validate_categorical_value(value, values, **kwargs):
    test = True
    if value not in values:
        test = False
    return test
