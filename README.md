# benderopt
benderopt is a black box optimization library:

```
from benderopt import minimize
import numpy as np

# We want to minimize the sinus function between 0 and 2pi
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
```