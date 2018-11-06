class BaseOptimizer:

    def __init__(self, optimization_problem, batch):
        self.optimization_problem = optimization_problem
        self.batch = batch

    @property
    def parameters(self):
        return self.optimization_problem.parameters

    @property
    def observations(self):
        return self.optimization_problem.observations

    def suggest(self):
        results = None
        if self.batch:
            results = self._generate_samples(self.batch)
        else:
            results = self._generate_samples(1)[0]
        return results
