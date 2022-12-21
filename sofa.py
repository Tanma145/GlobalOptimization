from math import *
import random
import copy
import numpy as np
from individual import *
from optimizer import Optimizer


def __inverse_probability(self, y, epsilon, x_0, x_min, x_max):
    a = (x_max - x_0) / epsilon
    b = (x_min - x_0) / epsilon
    return epsilon * p.tan(y * np.arctan(a) + (1.0 - y) * np.arctan(b)) + x_0


class SurvivalOfTheFittestAlgorithm(Optimizer):
    def __init__(self, *,
                 objective_function,
                 boundaries,
                 extr,
                 precision: float = 0.01,
                 initial_population_size: int = 1000,
                 max_iterations: int = 1000,
                 dispersion_a: float = 0.4,
                 dispersion_b: float = 2.5e-6
                 ):
        super().__init__(objective_function=objective_function,
                         boundaries=boundaries,
                         extr=extr)
        self.precision = precision
        self.initial_population_size = initial_population_size
        self.max_iterations = max_iterations
        self.dispersion_a = dispersion_a
        self.dispersion_b = dispersion_b

        self.population = []
        self.population_weights = []
        self.max = None
        self.min = None

    def __dispersion(self, k):
        kk = k - self.initial_population_size
        return kk ** (-self.dispersion_a - self.dispersion_b * kk)

    def get_population_size(self):
        return len(self.population)

    def calculate_weights(self):
        denominator = 0.0
        width = self.max.fitness - self.min.fitness
        power = len(self.population)

        for ind in self.population:
            numerator = ((ind.fitness-self.min.fitness) / width) ** power
            denominator += numerator
        for i, _ in enumerate(self.population):
            self.population_weights[i] /= denominator

    def generate_initial_population(self):
        self.population.clear()
        self.population_weights.clear()
        ind = get_random_individual(self.objective_function, self.boundaries)
        self.max = ind
        self.min = ind
        self.population.append(ind)
        self.population_weights.append(0.0)

        for _ in range(self.initial_population_size):
            ind = get_random_individual(self.objective_function,
                                        self.boundaries)
            if ind.fitness > self.max.fitness:
                self.max = ind
            if ind.fitness < self.min.fitness:
                self.min = ind

            self.population.append(ind)
            self.population_weights.append(0.0)
        self.calculate_weights()  # calculate_weights_max()

    def generate_child(self):
        base_list = random.choices(self.population,
                                   weights=self.population_weights)
        base = copy.deepcopy(base_list[0])
        for i in range(len(self.boundaries)):
            base[i] = inverse_probability(
                random.random(),
                self.__dispersion(len(self.population) + 1),
                base[i],
                self.boundaries[i][0],
                self.boundaries[i][1])
        base.calculate_fitness(self.objective_function)
        if base.fitness > self.max.fitness:
            self.max = base
        if base.fitness < self.min.fitness:
            self.min = base
        self.population.append(base)
        self.population_weights.append(0.0)
        if self.maximize:
            self.calculate_weights()  # calculate_weights_max()
        else:
            self.calculate_weights()  # calculate_weights_min()

    def optimize(self):
        self.generate_initial_population()

        while self.__dispersion(len(self.population)) > self.precision \
                and len(self.population) - self.initial_population_size \
                <= self.max_iterations:
            self.generate_child()

        if self.maximize:
            return self.max
        else:
            return self.min
