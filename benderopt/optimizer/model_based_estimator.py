import numpy as np
from ..base import Parameter, OptimizationProblem
from .optimizer import BaseOptimizer
from .parzen_estimator import parzen_estimator_build_posterior_parameter
from benderopt.utils import logb
from .random import RandomOptimizer
from sklearn.ensemble import RandomForestRegressor


class ModelBasedEstimator(BaseOptimizer):
    """ Parzen Estimator

    https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf

    gamma: ratio of best observations to build lowest loss function
    number_of_candidates: number of candidates to draw at each iteration
    subsampling: number of observations to consider at max
    subsampling_type: how to drow observations if number_of_observations > subsampling
    prior_weight: weight of prior when building posterior parameters
    minimum_observations: params will be drawn at random until minimum_observations is reached
    batch: batch size

    """

    def __init__(self,
                 optimization_problem,
                 gamma=0.15,
                 number_of_candidates=100,
                 subsampling=50,
                 subsampling_type="random",
                 prior_weight=0.5,
                 minimum_observations=30,
                 batch=None,
                 random_forest_parameters={
                     "n_estimators": 100,
                     "max_depth": 5,
                     "n_jobs": -1,
                 }
                 ):
        super(ModelBasedEstimator, self).__init__(optimization_problem, batch)

        self.gamma = gamma
        self.number_of_candidates = number_of_candidates
        self.subsampling = subsampling
        self.subsampling_type = subsampling_type
        self.prior_weight = prior_weight
        self.minimum_observations = minimum_observations
        self.batch = batch
        self.random_forest_parameters = random_forest_parameters

    def _generate_samples(self, size):
        assert size < self.number_of_candidates

        # If not enough observations, draw at random
        if self.optimization_problem.number_of_observations < self.minimum_observations:
            return RandomOptimizer(self.optimization_problem, self.batch)._generate_samples(size)

        # Retrieve self.gamma % best observations (lowest loss) observations_l
        # and worst obervations (greatest loss g) observations_g
        observations_l, _ = self.optimization_problem.observations_quantile(
            self.gamma,
            subsampling=min(len(self.observations), self.subsampling),
            subsampling_type=self.subsampling_type)

        # Build a posterior distribution according to best params and draw candidates from this
        candidates = np.array(RandomOptimizer(
            OptimizationProblem(
                [self._build_posterior_parameter(parameter, observations_l)
                 for parameter in self.parameters]), self.batch)._generate_samples(
            self.number_of_candidates))

        # Random forest Regressor trained on all observations.
        clf = RandomForestRegressor(**self.random_forest_parameters)

        clf.fit(*self.optimization_problem.dataset)

        scores = clf.predict([
            [candidate[parameter] for parameter in
             self.optimization_problem.sorted_parameters_name] for candidate in candidates])

        sorted_candidates = candidates[np.argsort(scores)]

        samples = sorted_candidates[:size]

        return samples

    def _build_posterior_parameter(self, parameter, observations):
        observed_values = [observation.sample[parameter.name] for observation in observations]
        return parzen_estimator_build_posterior_parameter[parameter.category](observed_values,
                                                                              parameter,
                                                                              self.prior_weight)
