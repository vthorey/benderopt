from .random import RandomOptimizer
from .parzen_estimator import ParzenEstimator
from .optimizer import BaseOptimizer

optimizers = {
    "base": BaseOptimizer,
    "random": RandomOptimizer,
    "parzen_estimator": ParzenEstimator,
}
