from .optimizer import BaseOptimizer
from .parzen_estimator import ParzenEstimator
from .random import RandomOptimizer

optimizers = {"base": BaseOptimizer, "random": RandomOptimizer, "parzen_estimator": ParzenEstimator}
