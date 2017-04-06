from benderopt.base import Observation
from benderopt.validation.utils import ValidationError
import pytest


def test_parameter_init():
    observation = Observation(sample={"alpha": 2}, loss=0.8)
    assert observation.parameters_name == set(["alpha"])


def test_parameter_from_dict():
    data = {
        "sample": {"alpha": 2},
        "loss": 0.8
    }
    observation = Observation.from_dict(data)
    assert observation.parameters_name == set(["alpha"])


def test_parameter_bad_sample_format():
    with pytest.raises(ValidationError):
        Observation(sample=[2], loss=0.8)


def test_parameter_from_dict_missing_loss():
    data = {
        "sample": {"alpha": 2},
    }
    with pytest.raises(ValidationError):
        Observation.from_dict(data)


def test_parameter_from_dict_missing_sample():
    data = {
        "loss": 0.8
    }
    with pytest.raises(ValidationError):
        Observation.from_dict(data)


def test_parameter_from_dict_bad_sample():
    data = {
        "loss": 0.8,
        "sample": [2],
    }
    with pytest.raises(ValidationError):
        Observation.from_dict(data)
