from individual import *


class DifferentialEvolution:
    def __init__(self,
                 population_size: int = 200,
                 mutation_power: float = 0.5,
                 crossover_probability: float = 0.95):
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_power = mutation_power
