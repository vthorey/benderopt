from benderopt.validation import is_parameter_value_valid, validate_search_space
from benderopt.stats import sample_generators, probability_density_function


class Parameter:
    def __init__(self, name, category, search_space):
        self.name = name

        if category not in ("categorical", "uniform", "normal", "gaussian_mixture"):
            raise ValueError
        self.category = category

        self.search_space = validate_search_space[self.category](search_space)

    def check_value(self, value):
        return is_parameter_value_valid[self.category](value, self.search_space)

    def draw(self, size):
        return sample_generators[self.category](size=size, **self.search_space)

    def pdf(self, samples):
        return probability_density_function[self.category](samples=samples, **self.search_space)

    @classmethod
    def from_dict(cls, data):
        if type(data) != dict:
            raise ValueError
        if set(data.keys()) != set(["name", "category", "search_space"]):
            raise ValueError
        return cls(**data)

    def __repr__(self):
        return self.name
