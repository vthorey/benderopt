import pytest
from benderopt.validation.categorical import validate_categorical, validate_categorical_value
from benderopt.validation.utils import ValidationError


def test_categorical_search_space_ok():
    search_space = {
        "values": ["a", "b", "c"],
        "probabilities": [0.1, 0.2, 0.7]
    }

    search_space = validate_categorical(search_space)


def test_categorical_search_space_not_dict():
    search_space = ["a", "b", "c"],

    with pytest.raises(ValidationError):
        search_space = validate_categorical(search_space)


def test_categorical_search_space_not_values():
    search_space = {

    }

    with pytest.raises(ValidationError):
        search_space = validate_categorical(search_space)


def test_categorical_search_space_bad_values():
    search_space = {
        "values": 3
    }

    with pytest.raises(ValidationError):
        search_space = validate_categorical(search_space)


def test_categorical_search_space_bad_probas():
    search_space = {
        "values": ["a", "b", "c"],
        "probabilities": 3
    }

    with pytest.raises(ValidationError):
        search_space = validate_categorical(search_space)


def test_categorical_search_space_bad_probas_size():
    search_space = {
        "values": ["a", "b", "c"],
        "probabilities": [0.5, 0.5]
    }

    with pytest.raises(ValidationError):
        search_space = validate_categorical(search_space)


def test_categorical_search_space_wrong_probas():
    search_space = {
        "values": ["a", "b", "c"],
        "probabilities": [0.5, 0.5, 0.5]
    }

    with pytest.raises(ValidationError):
        search_space = validate_categorical(search_space)


def test_categorical_search_space_no_probas():
    search_space = {
        "values": ["a", "b", "c"],
    }

    search_space = validate_categorical(search_space)

    assert "probabilities" in search_space.keys()
    assert sum(search_space["probabilities"]) == 1


def test_categorical_value():
    search_space = {
        "values": ["a", "b", "c"],
        "probabilities": [0.1, 0.2, 0.7]
    }
    assert validate_categorical_value("b", **search_space) is True
    assert validate_categorical_value("d", **search_space) is False
