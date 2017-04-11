from .random import RandomOptimizer
from .parzen_estimator import ParzenEstimator
from .optimizer import BaseOptimizer
from .model_based_estimator import ModelBasedEstimator

optimizers = {
    "base": BaseOptimizer,
    "random": RandomOptimizer,
    "parzen_estimator": ParzenEstimator,
    "model_based_estimator": ModelBasedEstimator,
}
