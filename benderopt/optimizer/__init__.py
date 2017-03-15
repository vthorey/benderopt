from .random import RandomOptimizer
from .tpe import TPE

optimizers = {
    "random": RandomOptimizer,
    "tpe": TPE,
}
