class BaseOptimizer:

    def __init__(self, optimization_problem):
        self.optimization_problem = optimization_problem

    @property
    def parameters(self):
        return self.optimization_problem.parameters

    @property
    def observations(self):
        return self.optimization_problem.observations

    def suggest(self, data):
        raise NotImplementedError

    def sample_exist(self, sample):
        return len(self.optimization_problem.find_observations(sample)) == 0
