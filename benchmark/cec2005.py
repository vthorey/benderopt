from benderopt.base import OptimizationProblem, Observation
from benderopt.optimizer import optimizers
from optproblems import cec2005
import numpy as np


def benchmark_cec2005(f,
                      optimization_problem,
                      num_var,
                      solutions,
                      stop=1,
                      optimizer_type="parzen_estimator",
                      number_of_evaluation=100,
                      verbose=True):
    if type(optimization_problem) == list:
        optimization_problem = OptimizationProblem.from_list(optimization_problem)

    for i in range(number_of_evaluation):
        if verbose and i % 20 == 0:
            print(int(i / number_of_evaluation * 100))
        optimizer = optimizers[optimizer_type](optimization_problem)
        sample = optimizer.suggest()
        print(sample)
        if np.linalg.norm(solutions - np.array([sample[str(i)] for i in range(num_var)]),
                          axis=1).min() < stop:
            return i, sample
        loss = f(*[sample[str(i)] for i in range(num_var)])
        observation = Observation.from_dict({"loss": loss, "sample": sample})
        optimization_problem.add_observation(observation)
    return i, optimization_problem.best_sample


if __name__ == "__main__":
    num_var = 2
    p = cec2005.F12(num_var)

    def f(*args, p=p):
        return p.objective_function(args)

    optimization_problem = [
        {
            "name": "{}".format(i),
            "category": "uniform",
            "search_space": {
                "low": -100,
                "high": 100,
            }
        } for i in range(num_var)
    ]

    steps, best_sample = benchmark_cec2005(f,
                                           optimization_problem,
                                           num_var,
                                           np.array([tuple(x.phenome)
                                                     for x in p.get_optimal_solutions()]),
                                           optimizer_type="parzen_estimator",
                                           number_of_evaluation=1000)
