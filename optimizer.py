import numpy as np


class Optimizer:
    def __init__(self, *, objective_function, boundaries):
        # переписать в словарь
        self.objective_function = objective_function
        self.boundaries = np.array(boundaries)
        self.fittest = None
