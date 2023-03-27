import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import sys
import cProfile
import pstats

from individual import Individual
from monte_carlo import MonteCarlo
from sofa import *
from desofa import DifEnvSOFA
import test_functions as tf


if __name__ == '__main__':
    # optimization
    # - parameters
    objective_function = tf.icicle_function
    boundaries = np.array([[0., 0.],
                           [2., 2.]])
    boundaries_x = (0., 2.)
    boundaries_y = (0., 2.)
    maxiter = 50000

    with cProfile.Profile() as pr:
        solver1 = DifEnvSOFA(objective_function=objective_function,
                             boundaries=boundaries,
                             _lambda=100,
                             pop_division=5,
                             precision=0.0,
                             maxiter=maxiter,
                             population_size=20)
        opt_point, opt = solver1.maximize()

    stats1 = pstats.Stats(pr)
    stats1.sort_stats(pstats.SortKey.TIME)
    stats1.print_stats()
    stats1.dump_stats(filename='profiling_DifEnvSOFA.prof')

    # with cProfile.Profile() as pr:
    #     solver = SurvivalOfTheFittestAlgorithm(
    #         objective_function=objective_function,
    #         boundaries=(boundaries_x, boundaries_y),
    #         precision=0.001,
    #         initial_population_size=100,
    #         max_iterations=2000,
    #         dispersion_a=0.4,
    #         dispersion_b=2.5e-6
    #     )
    #     optimum = solver.optimize()
    #

    print(opt_point, opt)
    print(solver1.population_size * solver1.generation)
    print(solver1.population.std())

    # graphics

    # scatter plot
    fig, ax = plt.subplots()
    points = np.array([(agent[0], agent[1]) for agent in solver1.population])
    color = [fit for fit in solver1.fitness_arr]
    sctr = ax.scatter(points[:, 0],
                      points[:, 1],
                      c=color,
                      cmap="jet",
                      alpha=0.2)
    ax.scatter(opt_point[0],
               opt_point[1],
               c='k',
               marker='X',
               s=500,
               alpha=0.5)
    ax.set_xlim(boundaries_x)
    ax.set_ylim(boundaries_y)
    ax.set_title(objective_function.__name__)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    s = "Оптимум {0:.4f} найден в " \
        "точке ({1:.4f}, {2:.4f})".format(opt,
                                          opt_point[0],
                                          opt_point[1])
    box = {'facecolor': 'white',
           'edgecolor': 'black',
           'boxstyle': 'round',
           'pad': 0.9}
    x_length = boundaries_x[1] - boundaries_x[0]
    y_length = boundaries_y[1] - boundaries_y[0]
    x_margin, y_margin = 0.05, 0.1
    plt.text(boundaries_x[0] + x_margin * x_length,
             boundaries_y[0] + y_margin * y_length,
             s,
             bbox=box)
    fig.colorbar(sctr, ax=ax)
    fig.tight_layout()
    plt.show()
