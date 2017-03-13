"""Module to validate a trials list"""


def validate_categorical(value, scope):
    test = True
    if value not in scope:
        test = False
    return test


def validate_normal(value, scope):
    test = True
    return test


def validate_uniform(value, scope):
    test = True
    if value < scope["min"]:
        test = False
    elif value > scope["max"]:
        test = False

    return test


validators = {
    "categorical": validate_categorical,
    "normal": validate_normal,
    "uniform": validate_uniform,
}


def validate_data(data):
    validated_trials = []
    parameters_keys = set([parameter['name'] for parameter in data["parameters"]])
    for trial in data["trials"]:
        if set(trial['parameters'].keys()) != parameters_keys:
            continue
        for parameter in data["parameters"]:
            if not validators[parameter['category']](trial['parameters'][parameter['name']],
                                                     parameter['scope']):
                break
        else:
            validated_trials.append(trial)
    return {
        'parameters': data["parameters"],
        'trials': validated_trials,
    }
