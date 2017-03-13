import json
import os


def get_test_data():
    return json.load(open("{}/tests/test_data.json".format(
        os.path.dirname(os.path.abspath(__file__))), "r"))


def get_number_of_possibility(parameters):
    number_of_possibility = None
    for parameter in parameters:
        if parameter['category'] == "categorical":
            if number_of_possibility is not None:
                number_of_possibility *= len(parameter['scope'])
            else:
                number_of_possibility = len(parameter['scope'])
        elif parameter['category'] == "uniform":
            if "step" in parameter["scope"]:
                scope = parameter["scope"]
                if number_of_possibility is not None:
                    number_of_possibility *= int((scope["max"] - scope["min"]) / scope["step"])
                else:
                    number_of_possibility = int((scope["max"] - scope["min"]) / scope["step"])
            else:
                number_of_possibility = None
                break
        elif parameter['category'] == "normal":
            if "step" in parameter["scope"]:
                scope = parameter["scope"]
                if number_of_possibility is not None:
                    number_of_possibility *= int(scope["sigma"] * 4 / scope["step"])
                else:
                    number_of_possibility = int(scope["sigma"] * 4 / scope["step"])
            else:
                number_of_possibility = None
                break
    return number_of_possibility
