# benderopt
benderopt

- Parameter(name, category, search_space)  : A Parameter to explore to optimize the problem.
    e.g: (a, uniform , {"low": 0, "high": 1})
         (b, categorical, {"values": ["lol", "haha"]})

- Sample: an instance the parameters of an Optimization Problem
    e.g: {"a": 0.3, "b": "lol"}

- observation(sample, loss, weight): The result of a trial with a sample of parameters.
(Result is the loss)
    e.g.: ({"a": 0.3, "b": "lol}", 0.4, 1)

`
from benderopt import minimize

def f(x):
    return np.sin(x)

optimization_problem = [
    {
        "name": "x",
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 2 * np.pi,
        }
    }
]

best_sample = minimize(f, optimization_problem, number_of_evaluation=100)

print(best_sample["x"], 3 * np.pi / 2)
`