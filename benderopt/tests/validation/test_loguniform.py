import pytest
from benderopt.validation.loguniform import validate_loguniform, validate_loguniform_value
from benderopt.validation.utils import ValidationError


def test_loguniform_search_space_ok():
    search_space = {
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }

    search_space = validate_loguniform(search_space)


def test_loguniform_search_space_not_dict():
    search_space = [1e-5, 1e5],

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_no_high():
    search_space = {
        "low": 1e-5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_no_low():
    search_space = {
        "high": 1e5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_bad_high():
    search_space = {
        "low": [1e-5],
        "high": 1e5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_bad_low():
    search_space = {
        "low": 1e-5,
        "high": [1e5],
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_low_not_positive():
    search_space = {
        "low": -1e1,
        "high": 1e5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_bad_low_high():
    search_space = {
        "low": 1e6,
        "high": 1e5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_bad_step():
    search_space = {
        "low": 1e0,
        "high": 1e5,
        "step": [1]
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)

    search_space = {
        "low": 1e0,
        "high": 1e5,
        "step": 1e6,
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_bad_base():
    search_space = {
        "low": 1e0,
        "high": 1e5,
        "base": [10],
    }

    with pytest.raises(ValidationError):
        search_space = validate_loguniform(search_space)


def test_loguniform_search_space_no_step_no_base():
    search_space = {
        "low": 1e0,
        "high": 1e5,
    }
    search_space = validate_loguniform(search_space)
    assert "step" in search_space.keys()
    assert search_space["step"] is None
    assert "base" in search_space.keys()
    assert search_space["base"] == 10


def test_loguniform_value():
    search_space = {
        "low": 1e-5,
        "high": 1e5,
        "step": 1e-6,
        "base": 10,
    }
    assert validate_loguniform_value(1e0, **search_space) is True
    assert validate_loguniform_value(1e5, **search_space) is False
    assert validate_loguniform_value(1e2, **search_space) is True
    assert validate_loguniform_value(1e10, **search_space) is False
    assert validate_loguniform_value(1e-6, **search_space) is False
