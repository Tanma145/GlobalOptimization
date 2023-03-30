from math import *
import random
import numpy as np
from scipy.stats import qmc


def inverse_probability(y, epsilon, x_0, x_min, x_max):
    a = (x_max - x_0) / epsilon
    b = (x_min - x_0) / epsilon
    return epsilon * np.tan(y * np.arctan(a) + (1.0 - y) * np.arctan(b)) + x_0


class SurvivalOfTheFittestAlgorithm:
    def __init__(self, *,
                 objective_function,
                 boundaries,
                 precision: float = 0.01,
                 initial_population_size: int = 1000,
                 init='latinhypercube',
                 max_iterations: int = 1000,
                 dispersion_a: float = 0.4,
                 dispersion_b: float = 2.5e-6,
                 ):
        self.objective_function = objective_function
        self.boundaries = np.array(boundaries)  # size = (2, D)
        self.precision = precision
        self.initial_population_size = initial_population_size
        self.init = init  # change?
        self.max_iterations = max_iterations
        self.dispersion_a = dispersion_a
        self.dispersion_b = dispersion_b

        self.dimension = boundaries.shape[1]
        self.population = []
        self.fitness_list = []
        self.population_weights = []
        self.max = None  # change?
        self.min = None
        self.max_point = None

    def __dispersion(self, k):
        kk = k - self.initial_population_size + 1
        return kk ** (-self.dispersion_a - self.dispersion_b * kk)

    def get_population_size(self):
        return len(self.population)

    def calculate_weights(self):
        height = self.max - self.min
        power = len(self.population)

        if len(self.population) == 1:
            self.population_weights[0] = 1
            return

        for i, _ in enumerate(self.population):
            self.population_weights[i] = (
                (self.fitness_list[i] - self.min) / height) ** power

    def generate_initial_population(self):
        self.fitness_list = [0] * self.initial_population_size
        self.population_weights = [0] * self.initial_population_size

        if isinstance(self.init, str):
            if self.init == 'latinhypercube':
                sampler = qmc.LatinHypercube(d=self.dimension)
                scale_arr = self.boundaries[1] - self.boundaries[0]
                shift_arr = self.boundaries[0]
                self.population = (shift_arr + sampler.random(
                    n=self.initial_population_size)*scale_arr)
            else:
                raise ValueError("Unacceptable init_population.")
        else:
            if (np.shape(self.init)
                    == (self.initial_population_size, self.dimension)):
                self.population = self.init
            else:
                raise ValueError("Initial population shape doesn't match"
                                 "dimension and/or initial population size.")

        self.max = self.objective_function(self.population[0])
        self.max_point = self.population[0]
        self.min = self.objective_function(self.population[0])

        for i in range(self.initial_population_size):
            objf = self.objective_function(self.population[i])
            self.fitness_list[i] = objf
            if objf > self.max:
                self.max = objf
                self.max_point = self.population[i]
            if objf < self.min:
                self.min = objf

        self.population = list(self.population)
        self.calculate_weights()

    def generate_child(self):
        # Селекция
        # случайно выбираем базовую особь с учётом весов
        base = random.choices(self.population,
                                   weights=self.population_weights,
                                   k=1)[0]

        mutant = np.empty_like(base)

        # Мутация
        for i in range(len(self.boundaries)):
            mutant[i] = inverse_probability(
                random.random(),
                self.__dispersion(len(self.population) + 1),
                base[i],
                self.boundaries[i][0],
                self.boundaries[i][1])

        # считаем фитнес новой особи и проверяем на максимум/минимум
        mutant_fitness = self.objective_function(mutant)
        if mutant_fitness > self.fitness_list[self.index_max]:
            self.max = mutant_fitness
            self.max_point = mutant
        if mutant_fitness < self.fitness_list[self.index_min]:
            self.min = mutant_fitness

        # добавляем особь к популяции
        self.population.append(mutant)
        self.population_weights.append(0.0)
        self.fitness_list.append(mutant_fitness)

        # пересчитываем веса
        self.calculate_weights()

    def optimize(self):
        self.generate_initial_population()

        while (self.__dispersion(len(self.population)) > self.precision
               and len(self.population) < self.max_iterations):
            self.generate_child()

        return self.max_point, self.max
