from individual import Individual


class WeightedIndividual(Individual):
    def __init__(self, genome, fitness, weight):
        super().__init__(genome, fitness)
        self.weight = weight
