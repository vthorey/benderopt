import pytest
from benderopt.validation.mixture import validate_mixture, validate_mixture_value
from benderopt.validation.utils import ValidationError


def test_mixture_search_space_ok():
    search_space = {
        "weights": [0.5, 0.5],
        "parameters": [
            {
                "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    search_space = validate_mixture(search_space)


def test_mixture_search_space_bad():
    search_space = ["bma"]

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_no_parameters():
    search_space = {
        "weights": [0.5, 0.5],
        # "parameters": [
        #     {
        #         "category": "normal",
        #         "search_space": {
        #             "mu": 0.5,
        #             "sigma": 1,
        #             "low": -5,
        #             "high": 5,
        #             "step": 0.1,
        #         }
        #     },
        #     {
        #         "category": "categorical",
        #         "search_space": {
        #             "values": [1, 2, 3],
        #             "probabilities": [0.1, 0.2, 0.7]
        #         }
        #     }
        # ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_bad_parameters():
    search_space = {
        "weights": [0.5, 0.5],
        "parameters":
            {
                "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
        }
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_missing_category():
    search_space = {
        "weights": [0.5, 0.5],
        "parameters": [
            {
                # "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_bad_category():
    search_space = {
        "weights": [0.5, 0.5],
        "parameters": [
            {
                "category": "lol",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_missing_search_space():
    search_space = {
        "weights": [0.5, 0.5],
        "parameters": [
            {
                "category": "normal",
                # "search_space": {
                #     "mu": 0.5,
                #     "sigma": 1,
                #     "low": -5,
                #     "high": 5,
                #     "step": 0.1,
                # }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_bad_search_space():
    search_space = {
        "weights": [0.5, 0.5],
        "parameters": [
            {
                "category": "normal",
                "search_space": ["lol"]
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_error_validation():
    search_space = {
        "weights": [0.5, 0.5],
        "parameters": [
            {
                "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": 50,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_bad_weights():
    search_space = {
        "weights": 1,
        "parameters": [
            {
                "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_bad_weights_size():
    search_space = {
        "weights": [1],
        "parameters": [
            {
                "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_bad_weights_sum():
    search_space = {
        "weights": [0.25, 0.25],
        "parameters": [
            {
                "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    with pytest.raises(ValidationError):
        search_space = validate_mixture(search_space)


def test_mixture_search_space_no_weights():
    search_space = {
        "parameters": [
            {
                "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }

    search_space = validate_mixture(search_space)

    assert "weights" in search_space.keys()
    assert sum(search_space["weights"]) == 1


def test_validate_mixture_value():
    search_space = {
        "parameters": [
            {
                "category": "normal",
                "search_space": {
                    "mu": 0.5,
                    "sigma": 1,
                    "low": -5,
                    "high": 5,
                    "step": 0.1,
                }
            },
            {
                "category": "categorical",
                "search_space": {
                    "values": [1, 2, 3],
                    "probabilities": [0.1, 0.2, 0.7]
                }
            }
        ]
    }
    validate_mixture_value("lol", **search_space)
