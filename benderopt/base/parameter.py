from benderopt.stats import probability_density_function, sample_generators
from benderopt.validation import is_parameter_value_valid, validate_search_space
from benderopt.validation.utils import ValidationError


class Parameter:
    def __init__(self, name, category, search_space):
        self.name = name

        if category not in sample_generators.keys():
            raise ValidationError(message="Category not recognized.")

        self.category = category

        self.search_space = validate_search_space[self.category](search_space)

    def check_value(self, value):
        return is_parameter_value_valid[self.category](value, **self.search_space)

    def draw(self, size):
        return sample_generators[self.category](size=size, **self.search_space)

    def pdf(self, samples):
        return probability_density_function[self.category](samples=samples, **self.search_space)

    def numeric_transform(self, value):
        """To convert categorical values to index"""
        value_transformed = value
        if self.category == "categorical":
            value_transformed = self.search_space["values"].index(value)
        return value_transformed

    def revert_numeric_transform(self, value):
        """To convert categorical index to value"""
        value_transformed = value
        if self.category == "categorical":
            value = self.search_space["values"][value]
        return value_transformed

    @classmethod
    def from_dict(cls, data):
        if type(data) != dict:
            raise ValidationError(message="'data' must be a dict")
        if set(data.keys()) != set(["name", "category", "search_space"]):
            raise ValidationError(message="'name', 'category', 'search_space' keys are mandatory")
        return cls(**data)

    def __repr__(self):
        return self.name
