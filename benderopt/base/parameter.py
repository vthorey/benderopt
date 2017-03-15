from benderopt.validation import is_parameter_value_valid, is_search_space_valid


class Parameter:
    def __init__(self, name, category, search_space):
        self.name = name

        if category not in ("categorical", "uniform", "normal"):
            raise ValueError
        self.category = category

        if is_search_space_valid[self.category](search_space):
            self.search_space = search_space

    def check_value(self, value):
        return is_parameter_value_valid[self.category](value, self.search_space)

    @classmethod
    def from_dict(cls, data):
        if type(data) != dict:
            raise ValueError
        if set(data.keys()) != set(["name", "category", "search_space"]):
            raise ValueError
        return cls(**data)

    def __repr__(self):
        return self.name
