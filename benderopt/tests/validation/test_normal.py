import pytest
from benderopt.validation.normal import validate_normal, validate_normal_value
from benderopt.validation.utils import ValidationError
import numpy as np


def test_normal_search_space_ok():
    search_space = {
        "mu": 0.5,
        "sigma": 1,
        "low": -5,
        "high": 5,
        "step": 0.1,
    }

    search_space = validate_normal(search_space)


def test_normal_search_space_not_dict():
    search_space = [0.5, 1],

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_no_mu():
    search_space = {
        "sigma": 1,
        "low": -5,
        "high": 5,
        "step": 0.1,
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_bad_mu():
    search_space = {
        "mu": [5],
        "sigma": 1,
        "low": -5,
        "high": 5,
        "step": 0.1,
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_no_sigma():
    search_space = {
        "mu": 0.5,
        "low": -5,
        "high": 5,
        "step": 0.1,
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_bad_sigma():
    search_space = {
        "mu": 5,
        "sigma": [1],
        "low": -5,
        "high": 5,
        "step": 0.1,
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_bad_low():
    search_space = {
        "mu": 0.5,
        "sigma": 1,
        "low": [-5],
        "high": 5,
        "step": 0.1,
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_bad_high():
    search_space = {
        "mu": 0.5,
        "sigma": 1,
        "low": -5,
        "high": [5],
        "step": 0.1,
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_bad_low_high():
    search_space = {
        "mu": 0.5,
        "sigma": 1,
        "low": 5,
        "high": 5,
        "step": 0.1,
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_set_low_high():
    search_space = {
        "mu": 0.5,
        "sigma": 1,
        "step": 0.1,
    }

    search_space = validate_normal(search_space)

    assert "low" in search_space.keys()
    assert search_space["low"] == -np.inf
    assert "high" in search_space.keys()
    assert search_space["high"] == np.inf


def test_normal_search_space_step():
    search_space = {
        "mu": 0.5,
        "sigma": 1,
        "low": -5,
        "high": 5,
        "step": {0.1},
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_step_high():
    search_space = {
        "mu": 0.5,
        "sigma": 1,
        "low": -5,
        "high": 5,
        "step": 5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_normal(search_space)


def test_normal_search_space_set_step():
    search_space = {
        "mu": 0.5,
        "sigma": 1,
        "low": -5,
        "high": 5,
    }

    search_space = validate_normal(search_space)
    assert "step" in search_space.keys()
    assert search_space["step"] is None


def test_normal_value():
    search_space = {
        "mu": 0,
        "sigma": 1,
        "low": -5,
        "high": 5,
        "step": 6,
        "base": 10,
    }
    assert validate_normal_value(3, **search_space) is True
    assert validate_normal_value(-13, **search_space) is False
    assert validate_normal_value(13, **search_space) is False
