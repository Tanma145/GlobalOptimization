import numpy as np
from numpy.random import default_rng


class MonteCarlo():
    def __init__(self, *,
                 objective_function,
                 boundaries,
                 extr,
                 max_iterations: int = 1000):
        self.objective_function = objective_function
        self.boundaries = np.array(boundaries)  # size = (2, D)
        self.extr = extr
        self.rng = default_rng()
        self.fittest = Individual(self.rng.uniform(self.boundaries[:, 0],
                                                   self.boundaries[:, 1],
                                                   len(self.boundaries)))
        self.max_iterations = max_iterations - 1
        self.sample = [self.fittest]

    def get_sample_size(self):
        return len(self.sample)

    def generate_individual(self):
        ind = Individual(self.rng.uniform(self.boundaries[:, 0],
                                          self.boundaries[:, 1],
                                          len(self.boundaries)))
        ind.fitness = self.objective_function(ind.genome)
        self.fittest = self.compare(ind, self.fittest)
        self.sample.append(ind)

    def optimize(self):
        self.sample.clear()
        for _ in range(self.max_iterations):
            self.generate_individual()
        return self.fittest
