from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
import numpy as np

np.random.seed(0)


def test_lognormal_generator():
    """Test to reassure."""
    search_space = {
        "weights": [0.25, 0.75],
        "parameters": [
            {
                "category": "uniform",
                "search_space": {
                    "low": -10,
                    "high": -5,
                    "step": None,
                }
            },
            {
                "category": "uniform",
                "search_space": {
                    "low": 0,
                    "high": 15,
                    "step": None,
                }
            }
        ]
    }
    size = 100000
    epsilon = 1e-1

    samples = sample_generators["mixture"](size=size, **search_space)
    theorical_mean = (
        search_space["weights"][0] *
        (search_space["parameters"][0]["search_space"]["high"] + search_space["parameters"][0]["search_space"]["low"]) * 0.5 +
        search_space["weights"][1] *
        (search_space["parameters"][1]["search_space"]["high"] + search_space["parameters"][1]["search_space"]["low"]) * 0.5
    )
    assert np.abs(np.mean(samples) - theorical_mean) / theorical_mean < epsilon


def test_lognormal_pdf():
    """Test to reassure."""
    search_space = {
        "weights": [0.25, 0.75],
        "parameters": [
            {
                "category": "uniform",
                "search_space": {
                    "low": -10,
                    "high": -5,
                    "step": None,
                }
            },
            {
                "category": "uniform",
                "search_space": {
                    "low": 0,
                    "high": 15,
                    "step": None,
                }
            }
        ]
    }
    epsilon = 1e-5
    samples = np.array([-15, -7, 5, 20])
    densities = probability_density_function["mixture"](samples=samples, **search_space)
    assert densities[0] == 0
    assert (densities[1] - 0.25 * 1 / (-5 + 10)) < epsilon
    assert (densities[2] - 0.75 * 1 / (15 - 0)) < epsilon
    assert densities[3] == 0
