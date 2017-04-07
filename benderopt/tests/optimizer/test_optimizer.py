import numpy as np
from benderopt.minimizer import minimize
from benderopt.tests.optimizer.helpers import function_to_optimize, optimization_problem, best_sample


def test_random_uniform():

    best_sample_random = minimize(function_to_optimize,
                                  optimization_problem,
                                  optimizer_type="parzen_estimator",
                                  number_of_evaluation=150)

    assert best_sample_random == best_sample
