import numpy as np


def get_random_individual(func, bounds):
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    x = np.random.uniform(lb, ub, len(bounds))
    ind = Individual(x, func(x))
    return ind


class Individual:
    def __init__(self, genome: np.array, fitness=0.0):
        self.genome = genome
        self.fitness = fitness

    def __add__(self, other):
        return Individual(self.genome + other.genome)

    def __sub__(self, other):
        return Individual(self.genome - other.genome)

    def __mul__(self, other):
        return Individual(self.genome * other)

    def __getitem__(self, key):
        return self.genome[key]

    def __setitem__(self, key, value):
        self.genome[key] = value

    # добавить оператор сравнения или нет
