import numpy as np
from benderopt.minimizer import minimize
from benderopt.tests.optimizer.helpers import function_to_optimize, optimization_problem, best_sample



def test_random_uniform():
    pass
    np.random.seed(0)
    import time
    t1 = time.time()
    best_sample_random = minimize(function_to_optimize,
                                  optimization_problem,
                                  optimizer_type="random",
                                  number_of_evaluation=25)
    print(function_to_optimize(**best_sample_random))
    print(time.time() - t1)

    t1 = time.time()
    np.random.seed(0)
    best_sample_random = minimize(function_to_optimize,
                                  optimization_problem,
                                  optimizer_type="parzen_estimator",
                                  number_of_evaluation=25)
    print(function_to_optimize(**best_sample_random))
    print(time.time() - t1)

    # assert best_sample_random == best_sample
