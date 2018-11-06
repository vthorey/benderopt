from benderopt.base import OptimizationProblem, Parameter, Observation
from benderopt.utils import get_test_optimization_problem
from benderopt.validation.utils import ValidationError
import pytest


def test_optimization_problem():
    parameter1 = Parameter(name="param1", category="categorical",
                           search_space={"values": ["a", "b"]})
    parameter2 = Parameter(name="param2", category="uniform", search_space={"low": 1, "high": 2})
    parameters = [parameter1, parameter2]
    optimization_problem = OptimizationProblem(parameters)
    observation1 = Observation(sample={"param1": "a", "param2": 1.5}, loss=1.5)
    optimization_problem.add_observation(observation1)
    observation2 = Observation(sample={"param1": "b", "param2": 1.8}, loss=1.8)
    optimization_problem.add_observation(observation2)
    observation3 = Observation(sample={"param1": "b", "param2": 1.05}, loss=0.1)
    optimization_problem.add_observation(observation3)

    assert type(optimization_problem.parameters) == list
    assert len(optimization_problem.observations) == 3
    assert optimization_problem.parameters_name == set(["param1", "param2"])
    assert observation1.sample in optimization_problem.samples
    assert len(optimization_problem.samples) == 3
    assert optimization_problem.best_sample == {"param1": "b", "param2": 1.05}
    assert optimization_problem.sorted_observations[0].sample == {"param1": "b", "param2": 1.05}
    assert optimization_problem.finite is False
    assert len(optimization_problem.find_observations({"param1": "b", "param2": 1.05})) == 1
    a, b = optimization_problem.observations_quantile(0.5)
    assert len(a) == 1
    assert len(b) == 2
    assert optimization_problem.get_best_k_samples(1)[0].sample == {"param1": "b", "param2": 1.05}


def test_optimization_problem_from_list():
    optimization_problem = OptimizationProblem.from_list([
        {
            "name": "param1",
            "category": "categorical",
            "search_space": {"values": ["a", "b"]}
        },
        {
            "name": "param2",
            "category": "uniform",
            "search_space": {"low": 1, "high": 2}
        }
    ])

    optimization_problem.add_observations_from_list(
        [
            {
                "loss": 1.5,
                "sample": {"param1": "a", "param2": 1.5}
            },
            {
                "loss": 1.8,
                "sample": {"param1": "b", "param2": 1.8}
            },
            {
                "loss": 0.1,
                "sample": {"param1": "b", "param2": 1.05}
            },
        ],
        raise_exception=True)

    assert type(optimization_problem.parameters) == list
    assert len(optimization_problem.observations) == 3
    assert optimization_problem.parameters_name == set(["param1", "param2"])
    assert {"param1": "b", "param2": 1.8} in optimization_problem.samples
    assert len(optimization_problem.samples) == 3
    assert optimization_problem.best_sample == {"param1": "b", "param2": 1.05}
    assert optimization_problem.sorted_observations[0].sample == {"param1": "b", "param2": 1.05}
    assert optimization_problem.finite is False
    assert len(optimization_problem.find_observations({"param1": "b", "param2": 1.05})) == 1
    a, b = optimization_problem.observations_quantile(0.5)
    assert len(a) == 1
    assert len(b) == 2
    assert optimization_problem.get_best_k_samples(1)[0].sample == {"param1": "b", "param2": 1.05}


def test_optimization_problem_from_json():
    get_test_optimization_problem()


def test_optimization_problem_bad_param():
    with pytest.raises(ValidationError):
        OptimizationProblem("lol")


def test_optimization_problem_bad_param_type():
    with pytest.raises(ValidationError):
        OptimizationProblem(["lol"])


def test_optimization_problem_add_bad_type():
    with pytest.raises(ValidationError):
        OptimizationProblem.from_list(
            {
                "name": "param1",
                "category": "categorical",
                "search_space": {"values": ["a", "b"]}
            })


def test_optimization_problem_add_bad_observation():
    optimization_problem = OptimizationProblem.from_list([
        {
            "name": "param1",
            "category": "categorical",
            "search_space": {"values": ["a", "b"]}
        },
        {
            "name": "param2",
            "category": "uniform",
            "search_space": {"low": 1, "high": 2}
        }
    ])
    observation2 = Observation(sample={"lol": "b", "param2": 1.8}, loss=1.8)
    with pytest.raises(ValidationError):
        optimization_problem.add_observation(observation2)


def test_optimization_problem_from_list_bad_type():
    optimization_problem = OptimizationProblem.from_list([
        {
            "name": "param1",
            "category": "categorical",
            "search_space": {"values": ["a", "b"]}
        },
        {
            "name": "param2",
            "category": "uniform",
            "search_space": {"low": 1, "high": 2}
        }
    ])
    with pytest.raises(ValidationError):
        optimization_problem.add_observations_from_list(
            "lol",
            raise_exception=True)


def test_optimization_problem_from_list_bad_sample_name():
    optimization_problem = OptimizationProblem.from_list([
        {
            "name": "param1",
            "category": "categorical",
            "search_space": {"values": ["a", "b"]}
        },
        {
            "name": "param2",
            "category": "uniform",
            "search_space": {"low": 1, "high": 2}
        }
    ])
    with pytest.raises(ValidationError):
        optimization_problem.add_observations_from_list(
            [
                {
                    "loss": 1.5,
                    "sample": {"param1": "a", "param2": 1.5}
                },
                {
                    "loss": 1.8,
                    "sample": {"lol": "b", "param2": 1.8}
                },
                {
                    "loss": 0.1,
                    "sample": {"param1": "b", "param2": 1.05}
                },
            ],
            raise_exception=True)


def test_optimization_problem_from_list_bad_value():
    optimization_problem = OptimizationProblem.from_list([
        {
            "name": "param1",
            "category": "categorical",
            "search_space": {"values": ["a", "b"]}
        },
        {
            "name": "param2",
            "category": "uniform",
            "search_space": {"low": 1, "high": 2}
        }
    ])
    with pytest.raises(ValidationError):
        optimization_problem.add_observations_from_list(
            [
                {
                    "loss": 1.5,
                    "sample": {"param1": "c", "param2": 1.5}
                },
                {
                    "loss": 1.8,
                    "sample": {"lol": "b", "param2": 1.8}
                },
                {
                    "loss": 0.1,
                    "sample": {"param1": "b", "param2": 1.05}
                },
            ],
            raise_exception=True)
