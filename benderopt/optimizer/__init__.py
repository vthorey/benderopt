from .random import RandomOptimizer
from .parzen_estimator import ParzenEstimator
from .random_parzen_estimator import RandomParzenEstimator
from .optimizer import BaseOptimizer
from .model_based_estimator import ModelBasedEstimator

optimizers = {
    "base": BaseOptimizer,
    "random": RandomOptimizer,
    "parzen_estimator": ParzenEstimator,
    "random_parzen_estimator": RandomParzenEstimator,
    "model_based_estimator": ModelBasedEstimator,
}
