import os
from benderopt.base import OptimizationProblem


def get_test_optimization_problem():
    return OptimizationProblem.from_json("{}/tests/test_data.json".format(
        os.path.dirname(os.path.abspath(__file__))))
