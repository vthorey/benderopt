import pytest
import numpy as np
from benderopt.validation.lognormal import validate_lognormal, validate_lognormal_value
from benderopt.validation.utils import ValidationError


def test_lognormal_search_space_ok():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    search_space = validate_lognormal(search_space)


def test_lognormal_search_space_not_dict():
    search_space = [1e-5, 1e5],

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_no_mu():
    search_space = {
        # "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_bad_mu():
    search_space = {
        "mu": [1e-0],
        "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_no_sigma():
    search_space = {
        "mu": 1e-0,
        # "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_bad_sigma():
    search_space = {
        "mu": 1e-0,
        "sigma": [1e1],
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_bad_low():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": [1e-5],
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_bad_low_0():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": -1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_bad_high():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e-5,
        "high": [1e5],
        "step": 1e-6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_bad_high_low():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_no_low_high():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "step": 1e-6,
        "base": 10,
    }
    search_space = validate_lognormal(search_space)

    assert "low" in search_space.keys()
    assert search_space["low"] == 0
    assert "high" in search_space.keys()
    assert search_space["high"] == np.inf


def test_lognormal_search_space_bad_step():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
        "step": [1e-6],
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_bad_high_step():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
        "step": 1e6,
        "base": 10,
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_bad_base():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": [10],
    }

    with pytest.raises(ValidationError):
        search_space = validate_lognormal(search_space)


def test_lognormal_search_space_no_base_step():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
    }

    search_space = validate_lognormal(search_space)
    assert "base" in search_space.keys()
    assert search_space["base"] == 10
    assert "step" in search_space.keys()
    assert search_space["step"] is None


def test_lognormal_value():
    search_space = {
        "mu": 1e-0,
        "sigma": 1e1,
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }
    assert validate_lognormal_value(1.5e3, **search_space) is True
    assert validate_lognormal_value(1.5e-13, **search_space) is False
    assert validate_lognormal_value(1.5e13, **search_space) is False
