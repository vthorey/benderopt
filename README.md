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

