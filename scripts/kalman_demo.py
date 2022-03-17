from math import sin, cos, pi

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace

from ai_umpire import KalmanFilterB

if __name__ == '__main__':
    x = []
    y = []
    for theta in linspace(0, 1.8 * pi, 10):
        # r = theta ** 1.1
        x.append(cos(theta))
        y.append(sin(theta))

    measurements = np.c_[x, y]

    n_variables = 2
    mu_p = np.zeros((2, 1))
    # mu_m = np.ones((n_variables, 1))
    psi = np.identity(2)
    phi = np.ones((n_variables, 1))

    print(measurements)

    # kf = KalmanFilterB(measurements=measurements, )
    plt.plot(x, y)
    plt.show()
