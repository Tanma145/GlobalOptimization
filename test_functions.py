from math import *


def test_function1(arg):
    x = arg[0]
    y = arg[1]
    return (sin(x) + 3 * cos(x) + sin(y) + 3 * cos(y)) ** 2 * (x - 0.5 * y)


def test_function2(arg):
    x = arg[0]
    y = arg[1]
    return (sin(x) + 3 * cos(x) + sin(y) + 3 * cos(y)) * (x - 0.5 * y)


def rosenbrock_function(arg):
    x = arg[0]
    y = arg[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def icicle_function(arg):
    x = arg[0]
    y = arg[1]
    return 1 + sin(10 * x) + cos(2 * x) + cos(2 * x + 2 * y) + cos(2 * y)\
           + sin(20 * y) + y ** 2


def horrific_function(arg):
    x = arg[0]
    y = arg[1]
    res = 0
    sum1 = 0
    sum2 = 0
    a = ((-0.940, -0.536, -0.743),
         (-0.502,  0.804,  0.769),
         (-0.428, -0.789,  0.204))
    b = ((0.590,  0.160, -0.681),
         (0.387,  0.945, -0.195),
         (-0.231,  0.152,  0.295))
    c = ((-0.896, -0.613, -0.463),
         (0.038, -0.428, -0.714),
         (0.103,  0.741, -0.317))
    d = ((-0.754, -0.558, -0.989),
         (-0.702,  0.881,  0.397),
         (-0.056,  0.085, -0.616))

    for i in range(3):
        for j in range(3):
            sum1 += (a[i][j] * sin(i * pi * (x - 1 / 2))
                             * sin(j * pi * (y - 1 / 2))
                     + b[i][j] * cos(i * pi * (x - 1 / 2))
                               * cos(j * pi * (y - 1 / 2)))
            sum2 += (c[i][j] * sin(i * pi * (x - 1 / 2))
                             * sin(j * pi * (y - 1 / 2))
                     + d[i][j] * cos(i * pi * (x - 1 / 2))
                               * cos(j * pi * (y - 1 / 2)))

    return sqrt(sum1 ** 2 + sum2 ** 2)


