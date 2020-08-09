def mandatory_key(key):
    return "key '{}' is mandatory".format(key)


def float_int_key(key):
    return "'{}' must be a float or int".format(key)


def list_key(key):
    return "'{}' must be a list".format(key)


def list_size_key(key_1, key_2):
    return "'{}' must have same size as {}".format(key_1, key_2)


def sum_to_one_key(key):
    return "'{}' sum must equal 1".format(key)


def must_be_strictly_superior(key1, key2):
    return "'{}' must be strictly superior than {}".format(key1, key2)


class ValidationError(Exception):

    error_messages = {
        "search_space_type": "Search space must be a dict.",
        "low_mandatory": mandatory_key("low"),
        "high_mandatory": mandatory_key("high"),
        "low_type": float_int_key("low"),
        "high_type": float_int_key("high"),
        "low_inferior_0": must_be_strictly_superior("low", 0),
        "high_inferior_low": must_be_strictly_superior("high", "'low'"),
        "mu_mandatory": mandatory_key("mu"),
        "sigma_mandatory": mandatory_key("sigma"),
        "mu_type": float_int_key("mu"),
        "sigma_type": float_int_key("sigma"),
        "mu_inferior_0": must_be_strictly_superior("mu", 0),
        "sigma_inferior_1": must_be_strictly_superior("sigma", "1"),
        "step_type": float_int_key("step"),
        "high_inferior_step": must_be_strictly_superior("high", "'step'"),
        "base_type": float_int_key("base"),
        "category_mandatory": mandatory_key("category"),
        "search_space_mandatory": mandatory_key("search_space"),
        "parameters_mandatory": mandatory_key("parameters"),
        "parameters_type": float_int_key("parameters"),
        "weights_type": list_key("weights"),
        "weights_sum": sum_to_one_key("weights"),
        "weights_size": list_size_key("weights", "parameters"),
        "values_mandatory": mandatory_key("values"),
        "values_type": list_key("values"),
        "probabilities_type": list_key("probabilities"),
        "probabilities_type": list_key("values"),
        "probabilities_sum": sum_to_one_key("values"),
        "probabilities_size": list_size_key("probabilities", "values"),
        "high_low_multiple_of_step": "'low' and 'high' must be a multiple of step",
    }

    def __init__(self, message_key=None, message=None):

        # Call the base class constructor with the parameters it needs
        if message_key:
            super(ValidationError, self).__init__(self.error_messages[message_key])
        if message:
            super(ValidationError, self).__init__(message)
