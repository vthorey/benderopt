import pytest
from benderopt.validation.uniform import validate_uniform, validate_uniform_value
from benderopt.validation.utils import ValidationError


def test_uniform_search_space_ok():
    search_space = {
        "low": -5,
        "high": 5,
        "step": 0.1,
    }

    search_space = validate_uniform(search_space)


def test_uniform_search_space_not_dict():
    search_space = [-5, 5],

    with pytest.raises(ValidationError):
        search_space = validate_uniform(search_space)


def test_uniform_search_space_no_high():
    search_space = {
        "low": -5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_uniform(search_space)


def test_uniform_search_space_no_low():
    search_space = {
        "high": 5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_uniform(search_space)


def test_uniform_search_space_bad_high():
    search_space = {
        "low": [-5],
        "high": 5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_uniform(search_space)


def test_uniform_search_space_bad_low():
    search_space = {
        "low": -5,
        "high": [5],
    }

    with pytest.raises(ValidationError):
        search_space = validate_uniform(search_space)


def test_uniform_search_space_bad_low_high():
    search_space = {
        "low": 6,
        "high": 5,
    }

    with pytest.raises(ValidationError):
        search_space = validate_uniform(search_space)


def test_uniform_search_space_bad_step():
    search_space = {
        "low": 0,
        "high": 5,
        "step": [1]
    }

    with pytest.raises(ValidationError):
        search_space = validate_uniform(search_space)

    search_space = {
        "low": 0,
        "high": 5,
        "step": 6,
    }

    with pytest.raises(ValidationError):
        search_space = validate_uniform(search_space)


def test_uniform_search_space_no_step():
    search_space = {
        "low": 0,
        "high": 5,
    }

    search_space = validate_uniform(search_space)

    assert "step" in search_space.keys()
    assert search_space["step"] is None


def test_uniform_value():
    search_space = {
        "low": 0,
        "high": 5,
    }

    assert validate_uniform_value(0, **search_space) is True
    assert validate_uniform_value(5, **search_space) is False
    assert validate_uniform_value(2, **search_space) is True
    assert validate_uniform_value(10, **search_space) is False
    assert validate_uniform_value(-10, **search_space) is False
