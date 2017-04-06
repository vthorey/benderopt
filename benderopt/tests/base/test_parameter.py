from benderopt.base import Parameter
from benderopt.validation.utils import ValidationError
import numpy as np
import pytest


def test_parameter_init():
    parameter = Parameter("alpha", "uniform", {"high": 5, "low": 0})
    assert parameter.name == "alpha"
    assert parameter.category == "uniform"
    assert parameter.search_space == {"high": 5, "low": 0, "step": None}
    assert parameter.check_value(2) is True
    assert len(parameter.draw(size=10)) == 10
    assert parameter.pdf(np.array([2.5])) == np.array([1 / 5])
    assert str(parameter) == "alpha"


def test_parameter_from_dict():
    data = {
        "name": "alpha",
        "category": "uniform",
        "search_space": {"high": 5, "low": 0}
    }
    parameter = Parameter.from_dict(data)
    assert parameter.name == "alpha"
    assert parameter.category == "uniform"
    assert parameter.search_space == {"high": 5, "low": 0, "step": None}
    assert parameter.check_value(2) is True
    assert len(parameter.draw(size=10)) == 10
    assert parameter.pdf(np.array([2.5])) == np.array([1 / 5])


def test_parameter_from_dict_bad_format():
    data = [1, 2]
    with pytest.raises(ValidationError):
        Parameter.from_dict(data)


def test_parameter_from_dict_missing_key():
    data = {
        "name": "alpha",
        "search_space": {"high": 5, "low": 0}
    }
    with pytest.raises(ValidationError):
        Parameter.from_dict(data)


def test_parameter_from_dict_bad_category():
    data = {
        "name": "alpha",
        "category": "lol",
        "search_space": {"high": 5, "low": 0}
    }
    with pytest.raises(ValidationError):
        Parameter.from_dict(data)
