from benderopt.validation import is_parameter_value_valid, validate_search_space
from benderopt.stats import sample_generators, probability_density_function


class BaseParameter:
    def __init__(self, name, category, search_space):
        self.name = name

        if category not in ("categorical", "uniform", "normal"):
            raise ValueError("Category not in base categories: {categorical, uniform, normal}")
        self.category = category

        self.search_space = validate_search_space[self.category](search_space)

        self._args = self.search_space

    def check_value(self, value):
        return is_parameter_value_valid[self.category](value, **self._args)

    def draw(self, size):
        return sample_generators[self.category](size=size, **self._args)

    def pdf(self, samples):
        return probability_density_function[self.category](samples=samples, **self._args)

    @classmethod
    def from_dict(cls, data):
        if type(data) != dict:
            raise ValueError
        if set(data.keys()) != set(["name", "category", "search_space"]):
            raise ValueError
        return cls(**data)

    def __repr__(self):
        return self.name


class Parameter(BaseParameter):
    def __init__(self, name, category, search_space):

        if category in ("categorical", "uniform", "normal"):
            self.base_parameter = True
            return super(Parameter, self).__init__(name, category, search_space)
        elif category == "mixture":
            self.base_parameter = False
            self.name = name
            self.category = category
            self.search_space = validate_search_space[self.category](search_space)
            self.parameters = [BaseParameter(name="param_{}".format(i), **parameter)
                               for i, parameter in enumerate(self.search_space["parameters"])]
            self._args = {
                "parameters": self.parameters,
                "weights": self.search_space["weights"]
            }
        else:
            raise ValueError("Unrecognized category.")
