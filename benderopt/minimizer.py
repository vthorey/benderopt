import logging

import numpy as np

from benderopt.base import Observation, OptimizationProblem
from benderopt.optimizer import optimizers
from benderopt.optimizer.optimizer import BaseOptimizer
from benderopt.rng import RNG


def minimize(
    f,
    optimization_problem_parameters,
    optimizer_type="parzen_estimator",
    number_of_evaluation=100,
    seed=None,
    debug=False,
):
    logger = logging.getLogger("benderopt")

    RNG.seed(seed)

    samples = []
    optimization_problem = OptimizationProblem.from_list(optimization_problem_parameters)

    if isinstance(optimizer_type, str):
        optimizer_type = optimizers[optimizer_type]
    if not issubclass(optimizer_type, BaseOptimizer):
        raise ValueError(
            "optimizer_type should either be a string or a subclass of BaseOptimizer, got {}".format(
                optimizer_type
            )
        )
    optimizer = optimizer_type(optimization_problem)

    for i in range(number_of_evaluation):
        logger.info("Evaluating {0}/{1}...".format(i + 1, number_of_evaluation))
        sample = optimizer.suggest()
        samples.append(sample)
        loss = f(**sample)
        logger.debug("f={0} for optimizer suggestion: {1}.".format(loss, sample))
        observation = Observation.from_dict({"loss": loss, "sample": sample})
        optimization_problem.add_observation(observation)
    if debug is True:
        return samples
    return optimization_problem.best_sample


if __name__ == "__main__":

    def f(x):
        return np.sin(x)

    optimization_problem_parameters = [
        {"name": "x", "category": "uniform", "search_space": {"low": 0, "high": 2 * np.pi}}
    ]

    best_sample = minimize(
        f, optimization_problem_parameters=optimization_problem_parameters, number_of_evaluation=100
    )

    print(best_sample["x"], 3 * np.pi / 2)
