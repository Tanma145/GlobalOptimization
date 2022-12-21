import numpy as np
import copy
from individual import *


class Generation:
    range = 0

    def __init__(self, population,
                 boundaries,
                 population_size: int = 200,
                 mutation_power: float = 0.5,
                 crossover_probability: float = 0.95):

        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_power = mutation_power
        self.population = population
        self.boundaries = boundaries

    def get_range(self):
        return self.range

    def create_new_gen(self):
        new_gen = copy.deepcopy(self)
        parent, mutagen1, mutagen2 = np.choices(new_gen.population, k=3)
        parent_mutated = parent.mutate(mutagen1, mutagen2, self.mutation_power)
