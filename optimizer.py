import numpy as np


class Optimizer:
    def __init__(self, *, objective_function, boundaries, extr="max"):
        # переписать в словарь
        match extr:
            case "max":
                self.compare = lambda x, y: x if x.fitness > y.fitness else y
            case "min":
                self.compare = lambda x, y: x if x.fitness < y.fitness else y
            case _:
                raise Exception("Unsupported state. Supported states: 'min', 'max'")
        self.objective_function = objective_function
        self.boundaries = np.array(boundaries)
        self.fittest = None
