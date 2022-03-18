from math import sin, cos, pi

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace

from ai_umpire import KalmanFilterB

if __name__ == "__main__":
    x = []
    y = []
    for theta in linspace(0, 1.8 * pi, 10):
        # r = theta ** 1.1
        x.append(cos(theta))
        y.append(sin(theta))

    measurements = np.c_[x, y]

    n_variables = 2
    n_measurements = measurements[0].shape[0]
    mu_p = np.zeros((n_variables, 1))
    mu_m = np.zeros((n_measurements, 1))
    psi = np.identity(n_variables)
    phi = np.eye(
        n_measurements, n_variables
    )  # Temporary, should relate data to state e.g. through a projection
    sigma_p = np.ones((n_variables, n_variables))
    sigma_m = np.ones((n_variables, n_variables))

    print("mu_p:\n", mu_p)
    print("mu_m:\n", mu_m)
    print("psi:\n", psi)
    print("phi:\n", phi)
    print("sigma_p:\n", sigma_p)
    print("sigma_m:\n", sigma_m)
    print("Start".center(40, "-"))

    kf = KalmanFilterB(
        measurements=measurements,
        sigma_m=sigma_m,
        sigma_p=sigma_p,
        phi=phi,
        psi=psi,
        mu_m=mu_m,
        mu_p=mu_p,
    )
    _, _ = kf.step()
    _, _ = kf.step()
    plt.plot(x, y)
    plt.show()
