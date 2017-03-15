from benderopt.optimizer import optimizers
from benderopt.base import OptimizationProblem, Observation
import numpy as np
import time


def minimize(f,
             optimization_problem,
             optimizer_type="tpe",
             number_of_evaluation=100):
    if type(optimization_problem) == list:
        optimization_problem = OptimizationProblem.from_list(optimization_problem)

    for _ in range(number_of_evaluation):
        t1 = time.time()
        optimizer = optimizers[optimizer_type](optimization_problem)
        sample = optimizer.suggest()
        print(sample)
        loss = f(**sample)
        observation = Observation.from_dict({"loss": loss, "sample": sample})
        optimization_problem.add_observation(observation)
        print(time.time() - t1)
    return optimization_problem.best_sample


if __name__ == "__main__":
    def f(x, y, z):
        y = np.sin(x)
        return (y - 1) ** 2

    optimization_problem = [
        {
            "name": "x",
            "category": "normal",
            "search_space": {
                "mu": np.pi / 2,
                "sigma": 1,
                "min": 0,
                "max": np.pi,
            }
        },
        {
            "name": "y",
            "category": "uniform",
            "search_space": {
                "mu": np.pi / 2,
                "sigma": 1,
                "max": 25,
                "min": 0,
            }
        },
        {
            "name": "z",
            "category": "categorical",
            "search_space": {
                "values": ["lol", "lal"],
                "weights": [0.5, 0.5],
            }
        }
    ]

    best_sample = minimize(f, optimization_problem, number_of_evaluation=1000)
