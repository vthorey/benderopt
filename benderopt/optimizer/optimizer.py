class BaseOptimizer:
    def __init__(self, optimization_problem):
        self.optimization_problem = optimization_problem

    @property
    def parameters(self):
        return self.optimization_problem.parameters

    @property
    def observations(self):
        return self.optimization_problem.observations

    def suggest(self, size=None):
        results = None
        if size is not None:
            results = self._generate_samples(size)
        else:
            results = self._generate_samples(1)[0]
        return results
