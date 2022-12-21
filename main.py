import numpy as np
import matplotlib.pyplot as plt
import sys

from individual import Individual
from monte_carlo import MonteCarlo
from sofa import *
import test_functions as tf


if __name__ == '__main__':
    # optimization
    # - parameters
    max_iter = 10000
    ips = 1
    precision = 0.05
    objective_function = tf.icicle_function
    boundaries_x = (0., 2.)
    boundaries_y = (0., 2.)

    opt1 = MonteCarlo(
        objective_function=objective_function,
        boundaries=(boundaries_x, boundaries_y),
        extr="max",
        max_iterations=max_iter)
    opt = SurvivalOfTheFittestAlgorithm(
        objective_function=objective_function,
        boundaries=(boundaries_x, boundaries_y),
        extr="max",
        precision=0.001,
        initial_population_size=1000,
        max_iterations=1000
    )
    optimum = opt1.optimize()
    print(optimum.genome, optimum.fitness)
    print(opt1.get_sample_size())

    # graphics

    # scatter plot
    fig, ax = plt.subplots()
    points = np.array([(ind[0], ind[1]) for ind in opt1.sample])
    color = [xx.fitness for xx in opt1.sample]
    sctr = ax.scatter(points[:, 0],
                      points[:, 1],
                      c=color,
                      cmap="jet",
                      alpha=0.2)
    ax.scatter(optimum[0],
               optimum[1],
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
        "точке ({1:.4f}, {2:.4f})".format(optimum.fitness,
                                          optimum.genome[0],
                                          optimum.genome[1])
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
    plt.colorbar(sctr, ax=ax)
    fig.tight_layout()
    plt.show()
