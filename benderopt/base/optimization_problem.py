import json

import numpy as np

from benderopt.base.observation import Observation
from benderopt.base.parameter import Parameter
from benderopt.rng import RNG
from benderopt.validation.utils import ValidationError


class OptimizationProblem:

    """OptimizationProblem


    - Parameters have a category which is "categorical", "uniform", "normal", "loguniform",
    "lognormal" and a search_space to define which values they can take.

    - Observations contains a sample which is a drawing for each parameters and an observed loss


    How to instantiate an OptimizationProblem
    =========================================
     * from list of parameters data

    optimization_problem = OptimizationProblem.from_list([
        {
            "name": "param1",
            "category": "category",
            "search_space": {"values": ["a", "b"]}
        },
        {
            "name": "param1",
            "category": "normal",
            "search_space": {"values": [1, 2]}
        }
    ])

     * from a list of Parameter instance
    parameter1 = Parameter(name="param1", category="categorical", search_space={"values": ["a", "b"]})
    parameter2 = Parameter(name="param2", category="uniform", search_space={"low": 1, "high": 2})
    optimization_problem = OptimizationProblem([parameter1, parameter2])

    How to add a observation
    ==================
     * from a list of observations data

    optimization_problem.add_observations_from_list(
        [
            {
                "loss": 0.5,
                "parameters": {"param1": "a", "param2": 1.5}
            },
            {
                "loss": 0.7,
                "parameters": {"param1": "a", "param2": 1.8}
            },
            {
                "loss": 0.1,
                "parameters": {"param1": "b", "param2": 1.1}
            },
    ],
    raise_exception=True)
    raise_exception = False would just discard wrong observations.

     * from a observation object

    observation1 = Observation(parameters={"param1": "a", "param2": 1.5}, loss=1.5)
    optimization_problem.add_observation(observation1)

    observation2 = Observation(parameters={"param1": "a", "param2": 1.8}, loss=1.8)
    optimization_problem.add_observation(observation2)

    observation3 = Observation(parameters={"param1": "b", "param2": 1.1}, loss=1.1)
    optimization_problem.add_observation(observation3)

    """

    def __init__(self, parameters):
        if type(parameters) != list:
            raise ValidationError(message="'parameters' must be a list of Parameter")
        for parameter in parameters:
            if type(parameter) != Parameter:
                raise ValidationError(message="'parameters' must be a list of Parameter")
        if len(set([x.name for x in parameters])) != len(parameters):
            raise ValidationError(message="Each parameter must have a different name")
        self.parameters = parameters.copy()
        self.observations = []

    @property
    def parameters_name(self):
        """Return a set containing the name for each parameter."""
        return set([parameter.name for parameter in self.parameters])

    @property
    def sorted_parameters(self):
        """Return a set containing the name for each parameter."""
        return sorted(self.parameters, key=lambda x: x.name)

    @property
    def samples(self):
        """Return all the samples."""
        if not hasattr(self, "_samples") or len(self._samples) != len(self.observations):
            self._samples = [observation.sample for observation in self.observations]
        return self._samples

    @property
    def dataset(self):
        """Create a dataset sample/loss

        Return:
          - matrix X with number_of_observations rows and number_of_parameters columns.
          - vector y of corresponding losses

        """
        data = [
            (
                [
                    parameter.numeric_transform(observation.sample[parameter.name])
                    for parameter in self.sorted_parameters
                ],
                observation.loss,
                observation.weight,
            )
            for observation in self.observations
        ]
        X, y, sample_weight = zip(*data)
        data = {"X": np.array(X), "y": np.array(y), "sample_weight": np.array(sample_weight)}
        return data

    @property
    def best_sample(self):
        """Return the sample with the lowest loss."""
        sample = None
        if len(self.observations) > 0:
            sample = self.sorted_observations[0].sample
        return sample

    @property
    def sorted_observations(self):
        """Observations ordered by increasing loss"""
        if not hasattr(self, "_sorted_observations") or len(self._sorted_observations) != len(
            self.observations
        ):
            self._sorted_observations = sorted(self.observations, key=lambda x: x.loss)
        return self._sorted_observations

    @property
    def finite(self):
        """Has the optimization problem infinite solution."""
        if not hasattr(self, "_finite"):
            _finite = True
            for parameter in self.parameters:
                if parameter.category != "categorical":
                    if parameter.search_space.get("step", None) is None:
                        _finite = False
                        break

            self._finite = _finite

        return self._finite

    @property
    def number_of_observations(self):
        return len(self.observations)

    def observations_quantile(self, quantile, subsampling=None, subsampling_type="random"):
        """Return observations int two lists lower than quantil and higher than quantile

        subsampling: max number of observations to consider
        subsampling_type:
          - "random": select subsamples at random
          - "best": select first subsampling best obersvations
        """
        observations_low, observations_high = None, None
        if subsampling is None:
            size = int(self.number_of_observations * quantile)
            observations_low, observations_high = (
                self.sorted_observations[:size],
                self.sorted_observations[size:],
            )
        else:
            if subsampling_type == "random":
                if self.number_of_observations > 0:
                    observations = np.array(self.observations)[
                        RNG.choice(self.number_of_observations, size=subsampling, replace=False)
                    ]
                else:
                    observations = []
                sorted_observations = sorted(observations, key=lambda x: x.loss)
            elif subsampling_type == "best":
                sorted_observations = self.sorted_observations[:subsampling]
            else:
                raise NotImplementedError(
                    "subsampling method {} does not exist!".format(subsampling_type)
                )
            size = int(len(sorted_observations) * quantile)
            observations_low, observations_high = (
                sorted_observations[:size],
                sorted_observations[size:],
            )
        return observations_low, observations_high

    def find_observations(self, sample):
        """Find corresponding observation."""
        observations = [
            observation for observation in self.observations if observation.sample == sample
        ]
        return observations

    def get_best_k_samples(self, k):
        return sorted(self.observations, key=lambda x: x.loss)[:k]

    def add_observation(self, observation, raise_exception=True):
        valid, reason = self._check_observation(observation)
        if valid:
            self.observations.append(observation)
        elif raise_exception:
            raise ValidationError(message=reason)

    def add_observations_from_list(self, observations, raise_exception=False):
        if type(observations) == list:
            for observation_data in observations:
                try:
                    observation = Observation.from_dict(observation_data)
                    self.add_observation(observation, raise_exception=raise_exception)
                except Exception as e:
                    if raise_exception:
                        raise e
        else:
            if raise_exception:
                raise ValidationError(message="Need to give a list of observations")

    def _check_observation(self, observation):
        valid = True
        reason = ""
        if observation.parameters_name != self.parameters_name:
            valid = False
            reason = "observation parameters {} != optimization_problem parameters {}".format(
                observation.parameters_name, self.parameters_name
            )
        else:
            for parameter in self.parameters:
                if not parameter.check_value(observation.sample[parameter.name]):
                    valid = False
                    reason = "Invalid parameter {} with value {}".format(
                        parameter.name, observation.sample[parameter.name]
                    )
                    break
        return valid, reason

    @classmethod
    def from_list(cls, parameters_data):
        parameters = []
        if type(parameters_data) != list:
            raise ValidationError(message="parameters_data must be a list of dict")

        for parameter_data in parameters_data:
            parameters.append(Parameter.from_dict(parameter_data))

        return cls(parameters)

    @classmethod
    def from_json(cls, filename):
        data = json.load(open(filename, "r"))
        optimization_problem = cls.from_list(data["parameters"])
        if data.get("observations") is not None:
            optimization_problem.add_observations_from_list(
                data["observations"], raise_exception=True
            )
        return optimization_problem
