
# benderopt

benderopt is a black box optimization library.

For asynchronous use, a web client using this library is available in open access at [bender.dreem.com](https://bender.dreem.com)

The algorithm implemented "parzen_estimator" is similar to TPE described in:
[Bergstra, James S., et al. “Algorithms for hyper-parameter optimization.” Advances in Neural Information Processing Systems.](https://www.lri.fr/~kegl/research/PDFs/BeBaBeKe11.pdf)

# Installation

```
pip install benderopt
```
or from the sources

# Demo
Here is a comparison on 200 evaluations of a function we want to minimize. First a random estimator is used to select random evaluation point. Then the parzen_estimator implemented in benderopt is used to select evaluation points.

The function to minimize is the following: `cos(x) + cos(2 * x + 1) + cos(y)`.

The red point correspond to the location of the global minima between 0 and 2pi for x and y.

<p align="center">
  <img src ="https://s3.eu-central-1.amazonaws.com/dreem-static/bender/benderopt.gif">
</p>

The code to generate the video can be found in `benchmark/benchmark_sinus2D`

We can observe on this example that the parzen estimator tends to explore more the local minimum than the random approach. This might lead to a better optimization given a fixed number of evaluations.

# The goal
In Black box optimization, we have a function to optimize but cannot compute the gradient, and evaluation is expensive in term of time / ressource. So we want to find a good exploration-exploitation trade off to get the best hyperparameters in as few evaluations as possible.
Use case are:
- Optimization of a machine learning model (number of layers of a neural network, function of activation, etc.
- Business optimization (marketing campain, a/b testing)

# Code Minimal Example

One of the advantage of benderopt is that it uses JSON-like object representation making it easier for a user to define parameters to optimize. This also allows an easy to integratation with an asynchrounous system such as [bender.dreem.com](https://bender.dreem.com).

Here is a minimal example.

```
from benderopt import minimize
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG) # logging.INFO will print less information

# We want to minimize the sinus function between 0 and 2pi
def f(x):
    return np.sin(x)

# We define the parameters we want to optimize:
optimization_problem_parameters = [
    {
        "name": "x", 
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 2 * np.pi,
        }
    }
]

# We launch the optimization
best_sample = minimize(f, optimization_problem_parameters, number_of_evaluation=50)

print(best_sample["x"], 3 * np.pi / 2)


> 4.710390692396651 4.71238898038469
```

# Minimal Documentation:
# Optimization Problem

An optimization problem contains:

- A list of parameters (i.e. parameters with their search space)
- A list of observation (i.e. values for each parameter of the list and a corresponding loss)

We use JSON-like representation for each of them e.g.
```
optimization_problem_data = {
    "parameters": [
        {
             "name": "parameter_1", 
             "category": "uniform",
             "search_space": {"low": 0, "high": 2 * np.pi, "step": 0.1}
        },
        {
            "name": "parameter_2", 
            "category": "categorical",
            "search_space": {"values": ["a", "b", "c"]}
        }
    ],
    "observations": [
        {
            "sample": {"parameter_1": 0.4, "parameter_2": "a"},
            "loss": 0.08
        },
        {
            "sample": {"parameter_1": 3.4, "parameter_2": "a"},
            "loss": 0.1
        },
        {
            "sample": {"parameter_1": 4.1, "parameter_2": "c"},
            "loss": 0.45
        },
    ]
}

```
## Optimizer
An optimizer takes an optimization problem and suggest new_predictions.
In other words, an optimizer takes a list of parameters with their search space and a history of past evaluations to suggest a new one.

Using the `optimization_problem_data` from the previous example:
```
from benderopt.base import OptimizationProblem, Observation
from benderopt.optimizer import optimizers

optimization_problem = OptimizationProblem.from_json(optimization_problem_data)
optimizer = optimizers["parzen_estimator"](optimization_problem)
sample = optimizer.suggest()

print(sample)

> {"parameter_1": 3.9, "parameter_2": "b"}
```
Optimizers currently available are `random` and `parzen_estimator`.

Benderopt allows to add a new optimizer really easily by inheriting an optimizer from `BaseOptimizer` class.

You can check `benderopt/optimizer/random.py` for a minimal example.

## Minimize function
Minimize function shown above in the `minimal example` section implementation is quite strateforward:
```
optimization_problem = OptimizationProblem.from_list(optimization_problem_parameters)
optimizer = optimizers["parzen_estimator"](optimization_problem)
for _ in range(number_of_evaluation):
    sample = optimizer.suggest()
    loss = f(**sample)
    observation = Observation.from_dict({"loss": loss, "sample": sample})
    optimization_problem.add_observation(observation)
```
The optimization_problem's observation history list is extended with a new observations at each iteration. This allows the optimizer to take them into account for the next suggestion.

## Uniform Parameter
| parameter | type      | default | comments                                                                                                            | 
|-----------|-----------|---------|---------------------------------------------------------------------------------------------------------------------| 
| low       | mandatory | -       | lowest possible value: all values will be greater than or equal to low                                              | 
| high      | mandatory | -       | highest value: all values will be stricly less than high                                                            | 
| step      | optionnal | None    | discretize the set of possible values: all values will follow 'value = low + k * step with k belonging to [0, K]' | 

e.g.
```
    {
        "name": "x", 
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 2 * np.pi,
            # "step": np.pi / 8
        }
    }
```

## Log-Uniform Parameter
| parameter | type      | default | comments                                                                                                            | 
|-----------|-----------|---------|---------------------------------------------------------------------------------------------------------------------| 
| low       | mandatory | -       | lowest possible value: all values will be greater than or equal to low                                              | 
| high      | mandatory | -       | highest value: all values will be stricly less than high                                                            | 
| step      | optionnal | None    | "discretize the set of possible values: all values will follow 'value = low + k * step with k belonging to [0, K]'" | 
| base      | optional  | 10      | logarithmic base to use                                                                                             | 

e.g.
```
    {
        "name": "x", 
        "category": "loguniform",
        "search_space": {
            "low": 1e-4,
            "high": 1e-2,
            # "step": 1e-5,
            # "base": 10,
        }
    }
```

## Normal Parameter
| parameter | type      | default | comments                                                                                                            | 
|-----------|-----------|---------|---------------------------------------------------------------------------------------------------------------------| 
| low       | optionnal | -inf    | lowest possible value: all values will be greater than or equal to low                                              | 
| high      | optionnal | inf     | highest value: all values will be stricly less than high                                                            | 
| mu        | mandatory | -       | mean value: all values will be initially drawn following a gaussian centered at mu with sigma variance              | 
| sigma     | mandatory | -       | sigma value: all values will be initially drawn following a gaussian centered at mu with sigma variance             | 
| step      | optionnal | None    | "discretize the set of possible values: all values will follow 'value = low + k * step with k belonging to [0, K]'" | 


e.g.
```
    {
        "name": "x", 
        "category": "normal",
        "search_space": {
            # "low": 0,
            # "high": 10,
            "mu": 5,
            "sigma": 1
            # "step": 0.01,
        }
    }
```
## Log-Normal Parameter
| parameter | type      | default | comments                                                                                                            | 
|-----------|-----------|---------|---------------------------------------------------------------------------------------------------------------------| 
| low       | optionnal | -inf    | lowest possible value: all values will be greater than or equal to low                                              | 
| high      | optionnal | inf     | highest value: all values will be stricly less than high                                                            | 
| mu        | mandatory | -       | mean value: all values will be initially drawn following a gaussian centered at mu with sigma variance              | 
| sigma     | mandatory | -       | sigma value: all values will be initially drawn following a gaussian centered at mu with sigma variance             | 
| step      | optionnal | None    | "discretize the set of possible values: all values will follow 'value = low + k * step with k belonging to [0, K]'" | 
| base      | optional  | 10      | logarithmic base to use                                                                                             | 

e.g.
```
    {
        "name": "x", 
        "category": "lognormal",
        "search_space": {
            # "low": 1e-6,
            # "high": 1e0,
            "mu": 1e-3,
            "sigma": 1e-2
            # "step": 1e-7,
            # "base": 10,
        }
    }
```
## Categorical Parameter
| parameter     | type      | default                                   | comments                                                                                  | 
|---------------|-----------|-------------------------------------------|-------------------------------------------------------------------------------------------| 
| values        | mandatory | -                                         | list of categories: all values will be sampled from this list                             | 
| probabilities | optionnal | number_of_values * [1 / number_of_values] | list of probabilities: all values will be initially drawn following this probability list | 

e.g.
```
    {
        "name": "x", 
        "category": "categorical",
        "search_space": {
            "values": ["a", "b", "c", "d"],
            # "probabilities": [0.1, 0.2, 0.2, 0.2, 0.3]
        }
    }
```
