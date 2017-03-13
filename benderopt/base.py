from .validation import validate_data


class BaseOptimizer:

    def __init__(self, data):
        self.data = validate_data(data)

    def suggest(self, data):
        raise NotImplementedError
