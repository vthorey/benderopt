from .optimizer import BaseOptimizer


class RandomOptimizer(BaseOptimizer):

    def __init__(self,
                 optimization_problem,
                 authorize_duplicate=True,
                 batch=None,
                 max_retry=50):
        super(RandomOptimizer, self).__init__(optimization_problem,
                                              authorize_duplicate=True,
                                              batch=None,
                                              max_retry=50)

    def _generate_samples(self, size):
        parameters = self.optimization_problem.parameters
        draws = [
            parameter.draw(size) if size == 1 else parameter.draw(size)
            for parameter in self.optimization_problem.parameters
        ]
        names = [parameter.name for parameter in parameters]
        return [
            {
                names[i]: value
                for i, value in enumerate(draw)
            } for draw in zip(*draws)
        ]
