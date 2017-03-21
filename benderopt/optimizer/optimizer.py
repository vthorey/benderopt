import numpy as np


class BaseOptimizer:

    def __init__(self, optimization_problem, authorize_duplicate, batch, max_retry):
        self.optimization_problem = optimization_problem
        if self.optimization_problem.finite and authorize_duplicate is False:
            raise ValueError("Finite problem cannot generate unique solutions.")
        self.authorize_duplicate = authorize_duplicate
        self.batch = batch
        self.max_retry = max_retry

    @property
    def parameters(self):
        return self.optimization_problem.parameters

    @property
    def observations(self):
        return self.optimization_problem.observations

    def _generate_unique_samples(self, size, max_retry=50):
        samples = np.ones(size) * np.nan
        nans_locations = np.where(np.isnan(samples))[0]
        for _ in range(max_retry):
            samples[nans_locations] = self._generate_samples(len(nans_locations))
            samples[self.optimization_problem.samples_unicity(samples)] = np.nan
            nans_locations = np.where(np.isnan(samples))[0]
            if len(nans_locations) == 0:
                break
        else:
            raise ValueError("No sample could be drawn in given bounds with max_retry {}".format(
                max_retry))
        return samples

    def suggest(self):
        results = None
        if self.batch:
            if self.authorize_duplicate:
                results = self._generate_sample(self.batch)
            else:
                results = self._generate_unique_sample(self.batch)
        else:
            if self.authorize_duplicate:
                results = self._generate_samples(1)[0]
            else:
                results = self._generate_unique_samples(1)[0]
        return results
