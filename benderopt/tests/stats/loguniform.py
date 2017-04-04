from benderopt.stats import sample_generators
from benderopt.stats import probability_density_function
import numpy as np

np.random.seed(0)


def test_loguniform_generator():
    data = {
        'low': 1,
        'high': 3,
        'step': None,
        'base': 10,
    }
    size = 10000
    epsilon = 1e-2
    samples = sample_generators["loguniform"](size=size, **data)
    assert np.abs(
        np.mean(samples) -
        (data["base"] ** (0.5 * (data["high"] + data["low"])))) < epsilon
    assert np.sum(samples < data["base"] ** data["low"]) == 0
    assert np.sum(samples >= data["base"] ** data["high"]) == 0


# def test_loguniform_step_generator():
#     data = {
#         'low': 2,
#         'high': 4,
#         'step': 2,
#         'base': 10,
#     }
#     size = 10000,
#     samples = sample_generators["loguniform"](size=size, **data)
#     assert np.sum(samples % data["step"]) == 0
#     assert np.sum(samples < data["base"] ** data["low"]) == 0
#     assert np.sum(samples >= data["base"] ** data["high"]) == 0


# def test_loguniform_pdf():
#     data = {
#         'low': -2,
#         'high': 0,
#         'step': None,
#         'base': 10,
#     }
#     samples = np.arange(0.001, 2, 0.001)
#     densities = probability_density_function["loguniform"](samples=samples, **data)
#     assert np.sum(densities[samples < (data["base"] ** data["low"])]) == 0
#     assert np.sum(densities[samples >= (data["base"] ** data["high"])]) == 0
#     assert np.sum((densities[1:] - densities[:-1]) != 0) == 2
#     assert densities[densities != 0][0] == (1 / ((data["base"] ** data["high"]) -
#                                                  (data["base"] ** data["low"])))
#     assert (np.sum(densities) / len(samples) - 1) <= 1e-3


# def test_loguniform_step_pdf():
#     data = {
#         'low': 2,
#         'high': 4,
#         'step': 10,
#         'base': 10,
#     }
#     samples = np.arange(50, 5000, 2)
#     densities = probability_density_function["loguniform"](samples=samples, **data)
#     assert np.sum(densities[samples < data["low"]]) == 0
#     assert np.sum(densities[samples >= data["high"]]) == 0
#     assert np.sum((densities[1:] - densities[:-1]) != 0) == 2
#     assert np.sum(densities) == 5 * 2 / 10
