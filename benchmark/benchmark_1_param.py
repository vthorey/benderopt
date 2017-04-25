import numpy as np
from benderopt.minimizer import minimize


optimization_problem = [
    {
        "name": "x1",
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 3.1416,
        }
    },
]


def function_to_optimize(x1):
    return (np.sin(x1) - np.pi / 2) ** 2


best_sample = {
    "x1": np.pi / 2,
}


np.random.seed(0)
seeds = np.random.randint(low=0, high=2 ** 31 - 1, size=5)
methods = ["parzen_estimator", "model_based_estimator", "random"]
number_of_trials = [30, 50, 100, 200, 500]
trials = {
    method: {
        n: []
        for n in number_of_trials
    }
    for method in methods
}
for seed in seeds:
    for n in number_of_trials:
        for method in methods:
            print("N: {}".format(n))
            np.random.seed(seed)
            best = minimize(function_to_optimize,
                            optimization_problem,
                            optimizer_type=method,
                            number_of_evaluation=n)
            trials[method][n].append(function_to_optimize(**best))
        # print(trials)

results = {
    method: {
        n: {
            "mean": np.mean(values),
            "std": np.std(values),
        }
        for n, values in ns.items()
    }
    for method, ns in trials.items()
}

for n in number_of_trials:
    print("\nN:{}\n".format(n))
    print("Random: {}/{}".format(results["random"][n]["mean"], results["random"][n]["std"]))
    print("parzen: {}/{}".format(results["parzen_estimator"][n]["mean"], results["parzen_estimator"][n]["std"]))
    print("model_: {}/{}".format(results["model_based_estimator"][n]["mean"], results["model_based_estimator"][n]["std"]))
