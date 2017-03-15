from ..base import BaseOptimizer
from ..stats import sample_generators


class RandomOptimizer(BaseOptimizer):

    def __init__(self,
                 optimization_problem,
                 authorize_duplicate=False,
                 batch=None,
                 max_retry=50):
        super(RandomOptimizer, self).__init__(optimization_problem)
        self.authorize_duplicate = authorize_duplicate
        self.batch = batch
        self.max_retry = max_retry

    def _generate_sample(self):
        sample = {
            parameter.name: sample_generators[parameter.category](**parameter.search_space)
            for parameter in self.optimization_problem.parameters
        }
        return sample

    def _generate_unique_sample(self):
        unique_sample = None
        for i in range(self.max_retry):
            sample = self._generate_sample()
            if self.sample_exist(sample):
                unique_sample = sample
                break
        return unique_sample

    def suggest(self):
        results = None
        if self.batch:
            if self.authorize_duplicate:
                results = [self._generate_sample() for _ in range(self.batch)]
            else:
                results = [self._generate_unique_sample() for _ in range(self.batch)]
        else:
            if self.authorize_duplicate:
                results = self._generate_sample()
            else:
                results = self._generate_unique_sample()
        return results
