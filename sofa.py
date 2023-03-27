from math import *
import random
import copy
import numpy as np
from scipy.stats import qmc
from individual import *
from optimizer import Optimizer


def inverse_probability(y, epsilon, x_0, x_min, x_max):
    a = (x_max - x_0) / epsilon
    b = (x_min - x_0) / epsilon
    return epsilon * np.tan(y * np.arctan(a) + (1.0 - y) * np.arctan(b)) + x_0


class SurvivalOfTheFittestAlgorithm(Optimizer):
    def __init__(self, *,
                 objective_function,
                 boundaries,
                 precision: float = 0.01,
                 initial_population_size: int = 1000,
                 max_iterations: int = 1000,
                 dispersion_a: float = 0.4,
                 dispersion_b: float = 2.5e-6,
                 ):
        super().__init__(objective_function=objective_function,
                         boundaries=boundaries)
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
        kk = k - self.initial_population_size + 1
        return kk ** (-self.dispersion_a - self.dispersion_b * kk)

    def get_population_size(self):
        return len(self.population)

    def calculate_weights(self):
        width = self.max.fitness - self.min.fitness
        power = len(self.population)

        if len(self.population) == 1:
            self.population_weights[0] = 1
            return

        for i, ind in enumerate(self.population):
            self.population_weights[i] = ((ind.fitness - self.min.fitness)
                                          / width) ** power

    def new_generate_initial_population(self):
        match self.init:
            case 'latinhypercube':
                sampler = qmc.LatinHypercube(d=len(self.boundaries))
            case _:
                raise ValueError("Wrong init value")
        sample = sampler.random(n=self.initial_population_size)
        for s, bounds in zip(sample, self.boundaries):
            pass

        self.population = list(sample)
        self.calculate_weights()

    def generate_initial_population(self):
        self.population.clear()
        self.population_weights.clear()

        ind = get_random_individual(self.objective_function, self.boundaries)
        self.max = ind
        self.min = ind
        self.population.append(ind)
        self.population_weights.append(0.0)

        for _ in range(self.initial_population_size - 1):
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
        # выбираем случайную особь с учётом весов
        base = random.choices(self.population,
                                   weights=self.population_weights,
                                   k=1)[0]

        mutant = np.empty_like(base)

        # мутация
        for i in range(len(self.boundaries)):
            mutant[i] = inverse_probability(
                random.random(),
                self.__dispersion(len(self.population) + 1),
                base[i],
                self.boundaries[i][0],
                self.boundaries[i][1])

        # считаем фитнес новой особи и проверяем на максимум/минимум
        mutant.calculate_fitness(self.objective_function)
        if mutant.fitness > self.max.fitness:
            self.max = mutant
        if mutant.fitness < self.min.fitness:
            self.min = mutant

        # добавляем особь к популяции
        self.population.append(mutant)
        self.population_weights.append(0.0)

        # пересчитываем веса
        self.calculate_weights()

    def optimize(self):
        self.generate_initial_population()

        while (self.__dispersion(len(self.population)) > self.precision
               and len(self.population) < self.max_iterations):
            self.generate_child()

        return self.max
