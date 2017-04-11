import numpy as np
from ..base import Parameter, OptimizationProblem
from .optimizer import BaseOptimizer
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


def build_posterior_categorical(observed_values, parameter, prior_weight):
    """ TODO Compare mean (current implem) vs hyperopt approach."""
    posterior_parameter = None
    prior_probabilities = np.array(parameter.search_space["probabilities"])
    values = parameter.search_space["values"]
    posterior_probabilities = prior_probabilities * prior_weight
    if len(observed_values) != 0:
        observed_probabilities = np.array([observed_values.count(value)
                                           for value in values])
        observed_probabilities = observed_probabilities / np.sum(observed_probabilities)
        posterior_probabilities += observed_probabilities * (1 - prior_weight)

    posterior_probabilities /= sum(posterior_probabilities)

    # Build param
    posterior_parameter = Parameter.from_dict(
        {
            "name": parameter.name,
            "category": "categorical",
            "search_space": {
                "values": values,
                "probabilities": list(posterior_probabilities),
            }
        }
    )
    return posterior_parameter


def find_sigmas_mus(observed_mus, prior_mu, prior_sigma, low, high):
    # Mus
    mus = np.sort(observed_mus + [prior_mu])

    # Sigmas
    # Trick to get for each mu the greater distance from left and right neighbor
    # when low and high are not defined we use inf to get the only available distance
    # (right neighbor for sigmas[0] and left for sigmas[-1])
    tmp = np.concatenate(
        (
            [low if low != -np.inf else np.inf],
            mus,
            [high if high != np.inf else -np.inf],
        )
    )
    sigmas = np.maximum(tmp[1:-1] - tmp[0:-2], tmp[2:] - tmp[1:-1])

    # Use formulas from hyperopt to clip sigmas
    sigma_max_value = prior_sigma
    sigma_min_value = prior_sigma / min(100.0, (1.0 + len(mus)))
    sigmas = np.clip(sigmas, sigma_min_value, sigma_max_value)

    # Fix prior sigma with correct value
    index_prior = np.where(mus == prior_mu)[0]
    sigmas[index_prior] = prior_sigma

    return mus, sigmas, index_prior


def build_posterior_uniform(observed_values, parameter, prior_weight):
    low = parameter.search_space["low"]
    high = parameter.search_space["high"]

    # build prior mu and sigma
    prior_mu = 0.5 * (high + low)
    prior_sigma = (high - low)

    # Build mus and sigmas centered on each observation, taking care of the prior
    mus, sigmas, index_prior = find_sigmas_mus(observed_mus=observed_values,
                                               prior_mu=prior_mu,
                                               prior_sigma=prior_sigma,
                                               low=low,
                                               high=high)

    posterior_parameter = Parameter.from_dict(
        {
            "name": parameter.name,
            "category": "mixture",
            "search_space": {
                "parameters": [
                    {
                        "category": "normal",
                        "search_space": {
                            "mu": mu.tolist(),
                            "sigma": sigma.tolist(),
                            "low": low,
                            "high": high,
                            "step": parameter.search_space.get("step", None)
                        }
                    } for mu, sigma in zip(mus, sigmas)
                ],
                "weights": [(1 - prior_weight) / (len(mus) - 1) if index != index_prior
                            else prior_weight
                            for index in range(len(mus))]
            }
        }
    )

    return posterior_parameter


def build_posterior_loguniform(observed_values, parameter, prior_weight):
    low_log = parameter.search_space["low_log"]
    high_log = parameter.search_space["high_log"]
    base = parameter.search_space["base"]

    # build log prior mu and sigma
    prior_mu_log = 0.5 * (high_log + low_log)
    prior_sigma_log = (high_log - low_log)

    # Build mus and sigmas centered on each observation, taking care of the prior
    mus_log, sigmas_log, index_prior = find_sigmas_mus(observed_mus=logb(observed_values, base),
                                                       prior_mu=prior_mu_log,
                                                       prior_sigma=prior_sigma_log,
                                                       low=low_log,
                                                       high=high_log)

    # Back from log scale
    mus = base ** mus_log
    sigmas = base ** sigmas_log

    posterior_parameter = Parameter.from_dict(
        {
            "name": parameter.name,
            "category": "mixture",
            "search_space": {
                "parameters": [
                    {
                        "category": "normal",
                        "search_space": {
                            "mu": mu.tolist(),
                            "sigma": sigma.tolist(),
                            "low": parameter.search_space["low"],
                            "high": parameter.search_space["high"],
                            "step": parameter.search_space["step"],
                            "base": parameter.search_space["base"],
                        }
                    } for mu, sigma in zip(mus, sigmas)
                ],
                "weights": [(1 - prior_weight) / (len(mus) - 1) if index != index_prior
                            else prior_weight
                            for index in range(len(mus))]
            }
        }
    )

    return posterior_parameter


def build_posterior_normal(observed_values, parameter, prior_weight):
    low = parameter.search_space["low"]
    high = parameter.search_space["high"]

    # build prior mu and sigma
    prior_mu = parameter.search_space["mu"]
    prior_sigma = parameter.search_space["sigma"]

    # Build mus and sigmas centered on each observation, taking care of the prior
    mus, sigmas, index_prior = find_sigmas_mus(observed_mus=observed_values,
                                               prior_mu=prior_mu,
                                               prior_sigma=prior_sigma,
                                               low=low,
                                               high=high)

    posterior_parameter = Parameter.from_dict(
        {
            "name": parameter.name,
            "category": "mixture",
            "search_space": {
                "parameters": [
                    {
                        "category": "normal",
                        "search_space": {
                            "mu": mu.tolist(),
                            "sigma": sigma.tolist(),
                            "low": low,
                            "high": high,
                            "step": parameter.search_space.get("step", None)
                        }
                    } for mu, sigma in zip(mus, sigmas)
                ],
                "weights": [(1 - prior_weight) / (len(mus) - 1) if index != index_prior
                            else prior_weight
                            for index in range(len(mus))]
            }
        }
    )

    return posterior_parameter


def build_posterior_lognormal(observed_values, parameter, prior_weight):
    low_log = parameter.search_space["low_log"]
    high_log = parameter.search_space["high_log"]
    base = parameter.search_space["base"]

    # build log prior mu and sigma
    prior_mu_log = parameter.search_space["mu_log"]
    prior_sigma_log = parameter.search_space["sigma_log"]

    # Build mus and sigmas centered on each observation, taking care of the prior
    mus_log, sigmas_log, index_prior = find_sigmas_mus(observed_mus=logb(observed_values, base),
                                                       prior_mu=prior_mu_log,
                                                       prior_sigma=prior_sigma_log,
                                                       low=low_log,
                                                       high=high_log)

    # Back from log scale
    mus = base ** mus_log
    sigmas = base ** sigmas_log

    posterior_parameter = Parameter.from_dict(
        {
            "name": parameter.name,
            "category": "mixture",
            "search_space": {
                "parameters": [
                    {
                        "category": "normal",
                        "search_space": {
                            "mu": mu.tolist(),
                            "sigma": sigma.tolist(),
                            "low": parameter.search_space["low"],
                            "high": parameter.search_space["high"],
                            "step": parameter.search_space["step"],
                            "base": parameter.search_space["base"],
                        }
                    } for mu, sigma in zip(mus, sigmas)
                ],
                "weights": [(1 - prior_weight) / (len(mus) - 1) if index != index_prior
                            else prior_weight
                            for index in range(len(mus))]
            }
        }
    )

    return posterior_parameter


parzen_estimator_build_posterior_parameter = {
    "categorical": build_posterior_categorical,
    "uniform": build_posterior_uniform,
    "loguniform": build_posterior_loguniform,
    "normal": build_posterior_normal,
    "lognormal": build_posterior_lognormal,
}
