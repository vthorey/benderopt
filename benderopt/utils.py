import os
from benderopt.base import OptimizationProblem


def get_test_optimization_problem():
    return OptimizationProblem.from_json("{}/tests/test_data.json".format(
        os.path.dirname(os.path.abspath(__file__))))


def get_number_of_possibility(parameters):
    number_of_possibility = None
    for parameter in parameters:
        if parameter['category'] == "categorical":
            if number_of_possibility is not None:
                number_of_possibility *= len(parameter['search_space'])
            else:
                number_of_possibility = len(parameter['search_space'])
        elif parameter['category'] == "uniform":
            if "step" in parameter["search_space"]:
                search_space = parameter["search_space"]
                if number_of_possibility is not None:
                    number_of_possibility *= int((search_space["max"] - search_space["min"]) / search_space["step"])
                else:
                    number_of_possibility = int((search_space["max"] - search_space["min"]) / search_space["step"])
            else:
                number_of_possibility = None
                break
        elif parameter['category'] == "normal":
            if "step" in parameter["search_space"]:
                search_space = parameter["search_space"]
                if number_of_possibility is not None:
                    number_of_possibility *= int(search_space["sigma"] * 4 / search_space["step"])
                else:
                    number_of_possibility = int(search_space["sigma"] * 4 / search_space["step"])
            else:
                number_of_possibility = None
                break
    return number_of_possibility
