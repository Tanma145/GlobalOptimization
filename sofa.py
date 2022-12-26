from math import *
import random
import copy
import numpy as np
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
        kk = k - self.initial_population_size + 1
        return kk ** (-self.dispersion_a - self.dispersion_b * kk)

    def get_population_size(self):
        return len(self.population)

    def calculate_weights(self):
        denominator = 0.0
        width = self.max.fitness - self.min.fitness
        power = len(self.population)
        if self.initial_population_size == 1:
            self.population_weights[0] = 1
            return

        for i, ind in enumerate(self.population):
            numerator = ((ind.fitness-self.min.fitness) / width) ** power
            denominator += numerator
            self.population_weights[i] = numerator
        for pw in self.population_weights:
            pw /= denominator

    def generate_initial_population(self):
        self.population.clear()
        self.population_weights.clear()
        ind = get_random_individual(self.objective_function, self.boundaries)
        self.max = ind
        self.min = ind
        self.population.append(ind)
        self.population_weights.append(0.0)

        for _ in range(1, self.initial_population_size):
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
        base_list = random.choices(self.population,
                                   weights=self.population_weights,
                                   k=1)
        # надо копировать иначе будет меняться уже существующая особь
        base = copy.deepcopy(base_list[0])

        # мутация
        for i in range(len(self.boundaries)):
            base[i] = inverse_probability(
                random.random(),
                self.__dispersion(len(self.population) + 1),
                base[i],
                self.boundaries[i][0],
                self.boundaries[i][1])

        # считаем фитнес новой особи и проверяем на максимум/минимум
        base.calculate_fitness(self.objective_function)
        if base.fitness > self.max.fitness:
            self.max = base
        if base.fitness < self.min.fitness:
            self.min = base

        # добавляем особь к популяции
        self.population.append(base)
        self.population_weights.append(0.0)

        # пересчитываем веса
        if self.extr == "max":
            self.calculate_weights()  # calculate_weights_max()
        elif self.extr == "min":
            self.calculate_weights()  # calculate_weights_min()

    def optimize(self):
        self.generate_initial_population()

        while (self.__dispersion(len(self.population)) > self.precision
               and len(self.population) + self.initial_population_size <= self.max_iterations):
            self.generate_child()

        # outdated
        if self.extr == "max":
            return self.max
        elif self.extr == "min":
            return self.min
