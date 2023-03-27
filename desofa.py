import numpy as np
from numpy.random import default_rng
import random
from scipy.stats import qmc


def inverse_probability(y, epsilon, x_0, x_min, x_max):
    a = (x_max - x_0) / epsilon
    b = (x_min - x_0) / epsilon
    return epsilon * np.tan(y * np.arctan(a) + (1.0 - y) * np.arctan(b)) + x_0


v_inverse_probability = np.vectorize(inverse_probability)


class DifEnvSOFA:
    def __init__(self, *,
                 objective_function,
                 boundaries,
                 _lambda,
                 pop_division,
                 precision,
                 maxiter,
                 population_size,
                 init='latinhypercube',
                 init_generation=0,
                 seed=0):
        self.objective_function = objective_function
        self.boundaries = np.array(boundaries)  # size = (2, D)
        self._lambda = _lambda
        self.pop_division = pop_division
        self.precision = precision
        self.maxiter = maxiter
        self.population_size = population_size
        self.init = init  # change?
        self.generation = init_generation
        self.selection_probabilities = np.empty(population_size)
        self.seed = seed

        self.dimension = boundaries.shape[1]
        self.population = np.empty((population_size, len(boundaries)))
        self.fitness_arr = np.empty(population_size)
        self.crossover_base = np.full(self.population_size
                                      - self.pop_division, 0.9)
        self.crossover_probabilities = np.full(self.population_size
                                               - self.pop_division, 0.9)
        self.dispersions_base = np.full(self.population_size
                                       - self.pop_division, 1.)
        self.dispersions = np.full(self.population_size
                                   - self.pop_division, 1.)
        self.index_max = 0  # change?
        self.index_min = 0
        self.rng = default_rng()

    def init_population(self):
        if isinstance(self.init, str):
            if self.init == 'latinhypercube':
                sampler = qmc.LatinHypercube(d=self.dimension)
                scale_arr = self.boundaries[1] - self.boundaries[0]
                shift_arr = self.boundaries[0]
                self.population = (
                        shift_arr
                        + sampler.random(n=self.population_size)*scale_arr)
            else:
                raise ValueError("Unacceptable init_population.")
        else:
            if (np.shape(self.init)
                    == (self.population_size, self.dimension)):
                self.population = self.init_population
            else:
                raise ValueError("Initial population shape doesn't match"
                                 "other parameters.")

        for i in range(self.population_size):
            self.fitness_arr[i] = self.objective_function(self.population[i])

        self.update_max_min()
        self.calculate_selection_probabilities()

    def update_max_min(self):
        self.index_max = self.fitness_arr.argmax()
        self.index_min = self.fitness_arr.argmin()

    def selection_power(self):  # change
        return (self.generation * self.population_size + 1)**(1/self._lambda)

    def dispersion(self, i):
        if i < self.pop_division:
            d = (self.generation*self.population_size + i + 1)**(-0.5/self.dimension)
        else:
            state = random.random()
            if state < 0.1:
                d = (self.generation*self.population_size + i + 1) ** (-0.5)
            else:
                d = self.dispersions_base[i - self.pop_division]
        return d

    def calculate_selection_probabilities(self):
        height = self.fitness_arr[self.index_max] - self.fitness_arr[self.index_min]
        self.selection_probabilities = (((self.fitness_arr - self.fitness_arr[self.index_min])/height)
                                        ** self.selection_power())

    def mutate(self, i):
        rand_arr = self.rng.uniform(size=self.dimension)
        # mutant = v_inverse_probability(rand_arr,
        #                                self.dispersion(i),
        #                                self.population[i],
        #                                self.boundaries[0],
        #                                self.boundaries[1])
        mutant = np.empty(self.dimension)
        for j in range(self.dimension):
            mutant[j] = inverse_probability(rand_arr[j],
                                            self.dispersion(i),
                                            self.population[i][j],
                                            self.boundaries[0][j],
                                            self.boundaries[1][j])
        return mutant

    def crossover(self, i, base_agent):
        crossed_agent = self.population[i].copy()
        ensurance = random.randint(0, self.dimension - 1)
        for j in range(self.dimension):
            state1 = random.random()
            state2 = random.random()
            k = i - self.pop_division
            if state1 < 0.1:
                self.crossover_probabilities[k] = random.random()
            if state2 < self.crossover_probabilities[k] or j == ensurance:
                crossed_agent[j] = base_agent[j]
        return crossed_agent

    def next_generation(self):
        base_agent_index = random.choices(range(self.population_size),
                                          weights=self.selection_probabilities,
                                          k=1)[0]

        # main population
        for i in range(self.pop_division):
            mutant = self.mutate(i)
            mutant_fitness = self.objective_function(mutant)
            if mutant_fitness > self.fitness_arr[i]:
                self.population[i] = mutant
                self.fitness_arr[i] = mutant_fitness

        # supporting population
        crossover_base = self.mutate(base_agent_index)

        for i in range(self.pop_division + 1, self.population_size):
            crossed = self.crossover(i, crossover_base)
            crossed_fitness = self.objective_function(crossed)
            if crossed_fitness > self.fitness_arr[i]:
                self.population[i] = crossed
                self.fitness_arr[i] = crossed_fitness

                j = i - self.pop_division
                self.crossover_base[j] = self.crossover_probabilities[j]
                self.dispersions_base[j] = self.dispersions[j]

        self.update_max_min()
        self.calculate_selection_probabilities()
        self.generation += 1

    def maximize(self):
        self.init_population()
        while self.generation * self.population_size < self.maxiter:
            self.next_generation()

        return (self.population[self.index_max],
                self.fitness_arr[self.index_max])

