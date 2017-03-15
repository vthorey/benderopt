import json
from . import Observation, Parameter


class OptimizationProblem:

    """ OptimizationProblem aims to


    - Parameters have a category which is "categorical", "uniform", "normal"
     and a search_space to define which values they can take.

    - Observations contains a drawing for each parameters and an observed loss


    How to instantiate an OptimizationProblem
    =========================================
     * from list of parameters data

    optimization_problem = OptimizationProblem.from_list({
        {
            "name": "param1",
            "category": "categorical",
            "search_space": {"values": [1, 2]}
        },
        {
            "name": "param1",
            "category": "categorical",
            "search_space": {"values": [1, 2]}
        }
    })

     * from a list of Parameter instance
    parameter1 = Parameter(name="param1", category="categorical", search_space={"values": ["a", "b"]})
    parameter1 = Parameter(name="param2", category="uniform", search_space={"min": 1, "max": 2})
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
            raise ValueError
        for parameter in parameters:
            if type(parameter) != Parameter:
                raise ValueError
        self.parameters = parameters
        self.observations = []

    @property
    def parameters_name(self):
        return set([parameter.name for parameter in self.parameters])

    @property
    def samples(self):
        if not hasattr(self, "_samples") or len(self._samples) != len(self.observations):
            self._samples = [observation.sample for observation in self.observations]
        return self._samples

    @property
    def best_sample(self):
        sample = None
        if len(self.observations) > 0:
            sample = self.sorted_observations[0]
        return sample

    @property
    def sorted_observations(self):
        if (not hasattr(self, "_sorted_observations") or
                len(self._sorted_observations) != len(self.observations)):
            self._sorted_observations = sorted(self.observations, key=lambda x: x.loss)
        return self._sorted_observations

    def observations_quantile(self, quantile):
        size = int(len(self.trials) * quantile)
        return self.sorted_observations[:size], self.sorted_observations[size:]

    def find_observations(self, sample):
        observations = [
            observation for observation in self.observations if observation.sample == sample]
        return observations

    def get_best_k_sample(self, k):
        return sorted(self.observations, key=lambda x: x.loss)[:k]

    def add_observation(self, observation, raise_exception=True):
        valid, reason = self._check_observation(observation)
        if valid:
            self.observations.append(observation)
        elif raise_exception:
            raise ValueError(reason)

    def add_observations_from_list(self, observations, raise_exception=False):
        if type(observations) == list:
            for observation_data in observations:
                try:
                    observation = Observation.from_dict(observation_data)
                    self.add_observation(observation, raise_exception=raise_exception)
                except Exception as e:
                    if raise_exception:
                        raise ValueError("{} invalid for a observation".format(observation_data))\
                            from e
        else:
            if raise_exception:
                raise ValueError("Need to give a list of observations")

    def _check_observation(self, observation):
        valid = True
        reason = ""
        if observation.parameters_name != self.parameters_name:
            valid = False
            reason = "observation parameters {} != optimization_problem parameters {}".format(
                observation.parameters_name,
                self.parameters_name)
        else:
            for parameter in self.parameters:
                if not parameter.check_value(observation.sample[parameter.name]):
                    valid = False
                    reason = "Invalid parameter {} with value {}".format(
                        parameter.name,
                        observation.sample[parameter.name])
                    break
        return valid, reason

    @classmethod
    def from_list(cls, parameters_data):
        parameters = []
        if type(parameters) != list:
            raise ValueError("parameters must be a list of dict")

        for parameter_data in parameters_data:
            parameters.append(Parameter.from_dict(parameter_data))

        return cls(parameters)

    @classmethod
    def from_json(cls, filename):
        data = json.load(open(filename, "r"))
        optimization_problem = cls.from_list(data["parameters"])
        if data.get("observations") is not None:
            optimization_problem.add_observations_from_list(
                data["observations"], raise_exception=True)
        return optimization_problem
