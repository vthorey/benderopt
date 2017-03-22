from benderopt.optimizer import optimizers
from benderopt.base import OptimizationProblem, Observation
import numpy as np


def minimize(f,
             optimization_problem,
             optimizer_type="parzen_estimator",
             number_of_evaluation=100):
    if type(optimization_problem) == list:
        optimization_problem = OptimizationProblem.from_list(optimization_problem)

    for _ in range(number_of_evaluation):
        optimizer = optimizers[optimizer_type](optimization_problem)
        sample = optimizer.suggest()
        loss = f(**sample)
        observation = Observation.from_dict({"loss": loss, "sample": sample})
        optimization_problem.add_observation(observation)
    return optimization_problem.best_sample


if __name__ == "__main__":
    def f(x):
        y = np.sin(x)
        return (y - 1) ** 2

    optimization_problem = [
        {
            "name": "x",
            "category": "normal",
            "search_space": {
                "mu": np.pi / 2,
                "sigma": 1,
                "low": 0,
                "high": np.pi,
            }
        }
    ]

    best_sample = minimize(f, optimization_problem, number_of_evaluation=1000)
