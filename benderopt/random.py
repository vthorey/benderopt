from .base import BaseOptimizer
from .utils_stats import sample_generators


class RandomOptimizer(BaseOptimizer):

    def __init__(self,
                 data,
                 authorize_duplicate=False,
                 batch=None,
                 max_retry=50):
        super(RandomOptimizer, self).__init__(data)
        self.authorize_duplicate = authorize_duplicate
        self.batch = batch
        self.max_retry = max_retry

    def suggest(self):
        results = None
        if self.batch:
            if self.authorize_duplicate:
                results = [self.generate_sample() for _ in range(self.batch)]
            else:
                results = [self.generate_unique_sample() for _ in range(self.batch)]
        else:
            if self.authorize_duplicate:
                results = self.generate_sample()
            else:
                results = self.generate_unique_sample()
        return results

    def generate_sample(self):
        sample = {
            parameter['name']: sample_generators[parameter['category']](parameter['scope'])
            for parameter in self.data['parameters']
        }
        return sample

    def generate_unique_sample(self):
        unique_sample = None
        for i in range(self.max_retry):
            sample = self.generate_sample()
            if self.is_sample_unique(sample):
                unique_sample = sample
                break
        return unique_sample
