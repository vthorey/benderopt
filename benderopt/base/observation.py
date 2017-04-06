from benderopt.validation.utils import ValidationError


class Observation:
    def __init__(self, sample, loss):
        self.loss = loss
        if type(sample) != dict:
            raise ValidationError(message="Sample must be a dict of 'parameter_name': value")
        self.sample = sample

    @classmethod
    def from_dict(cls, data):
        if data.get("loss") is None:
            raise ValidationError(message="Loss is mandatory for an observation")
        if data.get("sample") is None:
            raise ValidationError(message="Sample is mandatory for an observation")
        if type(data["sample"]) != dict:
            raise ValidationError(message="Sample must be a dict of parameter_name/values")
        return cls(data["sample"], data["loss"])

    @property
    def parameters_name(self):
        return set(self.sample.keys())
