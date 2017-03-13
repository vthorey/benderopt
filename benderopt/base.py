from .validation import validate_data


class BaseOptimizer:

    def __init__(self, data):
        self.data = validate_data(data)

    def suggest(self, data):
        raise NotImplementedError

    def is_sample_unique(self, sample):
        is_unique = True
        if not hasattr(self, '_trials_parameters'):
            self._trials_parameters = [trial["parameters"] for trial in self.data["trials"]]
        if sample in self._trials_parameters:
            is_unique = False
        return is_unique
