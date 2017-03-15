import numpy as np
from ..base import BaseOptimizer
from ..stats import sample_generators, categorical_logpdf, gaussian_mixture_logpdf


class PosteriorSearchSpaceUsingGaussianMixture:
    def __init__(self, search_space, parameters_value, prior_mu, prior_sigma):
        self.search_space = search_space
        self.parameters_value = parameters_value
        self.mus = np.sort(parameters_value + [prior_mu])

        # Trick to get for each mu the greater distance from left and right neighbor
        # when low and high are not defined we use inf to get the only available distance
        # (right neighbor for sigmas[0] and left for sigmas[-1])
        tmp = np.concatenate(
            ([search_space.get("low", np.inf)], self.mus, [search_space.get("high", -np.inf)],)
        )
        self.sigmas = np.maximum(tmp[1:-1] - tmp[0:-2], tmp[2:] - tmp[1:-1])

        # Use formulas from hyperopt to clip sigmas
        sigma_max_value = prior_sigma
        sigma_min_value = prior_sigma / min(100.0, (1.0 + len(self.mus)))
        self.sigmas = np.clip(self.sigmas, sigma_min_value, sigma_max_value)

        # Fix prior sigma with correct value
        self.sigmas[np.where(self.mus == prior_mu)[0]] = prior_sigma

    def draw(self, size):
        return sample_generators["gaussian_mixture"](mus=self.mus,
                                                     sigmas=self.sigmas,
                                                     weights=None,
                                                     low=self.search_space.get("low"),
                                                     high=self.search_space.get("high"),
                                                     log=self.search_space.get("log"),
                                                     step=self.search_space.get("step"),
                                                     size=size,
                                                     max_retry=50)

    def evaluate(self, samples):
        return gaussian_mixture_logpdf(samples=samples,
                                       mus=self.mus,
                                       sigmas=self.sigmas,
                                       weights=None,
                                       low=self.search_space.get("low"),
                                       high=self.search_space.get("high"),
                                       log=self.search_space.get("log"),
                                       step=self.search_space.get("step"))


class PosteriorSearchSpaceUniform(PosteriorSearchSpaceUsingGaussianMixture):
    def __init__(self, search_space, parameters_value):
        super(PosteriorSearchSpaceUniform, self).__init__(
            search_space=search_space,
            parameters_value=parameters_value,
            prior_mu=0.5 * (search_space["high"] + search_space["low"]),
            prior_sigma=(search_space["high"] - search_space["low"]))


class PosteriorSearchSpaceNormal(PosteriorSearchSpaceUsingGaussianMixture):
    def __init__(self, search_space, parameters_value):
        super(PosteriorSearchSpaceNormal, self).__init__(
            search_space=search_space,
            parameters_value=parameters_value,
            prior_mu=search_space["mu"],
            prior_sigma=search_space["sigma"])


class PosteriorSearchSpaceCategorical:
    def __init__(self, search_space, parameter_values):
        self.values = search_space["values"]
        number_of_values = len(self.values)
        prior_weights = np.array(search_space.get("weights",
                                                  np.ones(number_of_values) / number_of_values))
        self.posterior_weights = prior_weights
        if len(parameter_values) != 0:
            observed_weights = np.array([parameter_values.count(value) for value in self.values])
            observed_weights = observed_weights / np.sum(observed_weights)
            self.posterior_weights = (self.posterior_weights + observed_weights) / 2

    def draw(self, size):
        return sample_generators["categorical"](values=self.values,
                                                weights=self.posterior_weights,
                                                size=size)

    def evaluate(self, samples):
        return categorical_logpdf(samples=samples, values=self.values, weights=self.posterior_weights)


tpe_posterior_search_space = {
    "categorical": PosteriorSearchSpaceCategorical,
    "uniform": PosteriorSearchSpaceUniform,
    "normal": PosteriorSearchSpaceNormal,
}


class TPE(BaseOptimizer):

    def __init__(self,
                 optimization_problem,
                 gamma=0.15,
                 number_of_candidates=100,
                 max_retry=5):
        super(TPE, self).__init__(optimization_problem)
        self.gamma = gamma
        self.number_of_candidates = number_of_candidates

    def _generate_sample(self):
        # Retrieve self.gamma % best observations (lowest loss) observations_l
        # and worst obervations (greatest loss g) observations_g
        observations_l, observations_g = self.optimization_problem.observations_quantile(
            self.gamma)

        # Build a sample going through every parameters
        sample = {}
        for parameter in self.parameters:
            posterior_search_space_l = tpe_posterior_search_space[parameter.category](
                parameter.search_space,
                [observation.sample[parameter.name]
                 for observation in observations_l])
            posterior_search_space_g = tpe_posterior_search_space[parameter.category](
                parameter.search_space,
                [observation.sample[parameter.name]
                 for observation in observations_g])

            # Draw candidates from observations_l
            candidates = posterior_search_space_l.draw(self.number_of_candidates)

            # Evaluate cantidates score according to their proba
            scores = (posterior_search_space_g.evaluate(candidates) -
                      posterior_search_space_l.evaluate(candidates))

            sample[parameter.name] = candidates[np.argmin(scores)]
        return sample

    def suggest(self):
        return self._generate_sample()
