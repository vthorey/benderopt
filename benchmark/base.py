from itertools import product
from joblib import Parallel, delayed
from benderopt import minimize
import numpy as np
import tqdm


def benchmark_simple(function_to_optimize,
                     optimization_problem,
                     target,
                     methods,
                     number_of_evaluations,
                     seeds,
                     parallel=True):
    """Aim to benchmark analytics functions of multiple variables"""
    results_tmp = {}

    assert set(target.keys()) == set([x["name"] for x in optimization_problem])
    parameters = set(target.keys())

    trials = list(product(seeds, number_of_evaluations, methods))
    print("Number of trials:", len(trials))
    if parallel is True:
        best_samples = Parallel(n_jobs=-1, verbose=8)(
            delayed(minimize)(
                function_to_optimize,
                optimization_problem,
                optimizer_type=method,
                number_of_evaluation=number_of_evaluation,
                seed=seed,
            ) for seed, number_of_evaluation, method in trials
        )
    else:
        best_samples = [
            minimize(
                function_to_optimize,
                optimization_problem,
                optimizer_type=method,
                number_of_evaluation=number_of_evaluation,
                seed=seed,
            ) for seed, number_of_evaluation, method in tqdm.tqdm(trials)
        ]

    for (seed, number_of_evaluation, method), best in zip(trials, best_samples):
        results_tmp.setdefault(method, {}).setdefault(number_of_evaluation, []).append(
            np.sqrt(np.sum([(best[parameter] - target[parameter]) ** 2 for parameter in parameters])))

    results = {
        method: {
            n: {
                "mean": np.mean(values),
                "std": np.std(values),
            }
            for n, values in ns.items()
        }
        for method, ns in results_tmp.items()
    }
    return results


def benchmark_value(function_to_optimize,
                    optimization_problem,
                    target,
                    methods,
                    number_of_evaluations,
                    seeds,
                    parallel=True):
    """Aim to benchmark analytics functions of multiple variables"""
    results_tmp = {}

    trials = list(product(seeds, number_of_evaluations, methods))
    print("Number of trials:", len(trials))
    if parallel is True:
        best_samples = Parallel(n_jobs=-1, verbose=8)(
            delayed(minimize)(
                function_to_optimize,
                optimization_problem,
                optimizer_type=method,
                number_of_evaluation=number_of_evaluation,
                seed=seed,
            ) for seed, number_of_evaluation, method in trials
        )
    else:
        best_samples = [
            minimize(
                function_to_optimize,
                optimization_problem,
                optimizer_type=method,
                number_of_evaluation=number_of_evaluation,
                seed=seed,
            ) for seed, number_of_evaluation, method in tqdm.tqdm(trials)
        ]

    for (seed, number_of_evaluation, method), best in zip(trials, best_samples):
        results_tmp.setdefault(method, {}).setdefault(number_of_evaluation, []).append(
            np.abs(function_to_optimize(**best) - target))

    results = {
        method: {
            n: {
                "mean": np.mean(values),
                "std": np.std(values),
            }
            for n, values in ns.items()
        }
        for method, ns in results_tmp.items()
    }
    return results
