from math import sin, cos, pi
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from numpy import linspace

from ai_umpire import KalmanFilter

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 5
sim_frames_path: Path = root_dir_path / "sim_frames" / f"sim_{sim_id}_frames"
sim_blurred_frames_path: Path = (
    root_dir_path / "blurred_frames" / f"sim_{sim_id}_blurred"
)

if __name__ == "__main__":
    # Check that ball data file exists
    data_file_path = root_dir_path / "ball_pos" / f"sim_{sim_id}.csv"
    if not data_file_path.exists():
        raise IOError("Data file not found")

    col_names = ["x", "y", "z"]
    ball_pos = pd.DataFrame(pd.read_csv(data_file_path), columns=col_names)
    x = []
    y = []
    for theta in linspace(0, 1.8 * pi, 10):
        # r = theta ** 1.1
        x.append(cos(theta))
        y.append(sin(theta))

    # x = ball_pos["x"]
    # y = ball_pos["y"]

    rng = np.random.default_rng()

    measurements = np.c_[x, y]
    noisy_measurements = measurements + rng.normal(
        0, 0.2, size=(measurements.shape[0], 2)
    )

    n_variables = 2
    n_measurement_vals = measurements[0].shape[0]
    mu_p = np.zeros((n_variables, 1))
    mu_m = np.zeros((n_measurement_vals, 1))
    psi = np.identity(n_variables)
    phi = np.eye(
        n_measurement_vals, n_variables
    )  # Temporary, should relate data to state e.g. through a projection
    # sigma_p = np.ones((states_dim, states_dim))
    # sigma_m = np.ones((states_dim, states_dim))
    sigma_p = np.identity(n_variables) * 3
    sigma_m = np.identity(n_variables) * 4

    print("mu_p:\n", mu_p)
    print("mu_m:\n", mu_m)
    print("psi:\n", psi)
    print("phi:\n", phi)
    print("sigma_p:\n", sigma_p)
    print("sigma_m:\n", sigma_m)
    print("Start".center(40, "-"))

    kf = KalmanFilter(
        measurements=noisy_measurements,
        sigma_m=sigma_m,
        sigma_p=sigma_p,
        phi=phi,
        psi=psi,
        mu_m=mu_m,
        mu_p=mu_p,
    )

    mu_list = []
    cov_list = []

    for i in range(measurements.shape[0]):
        mu, cov = kf.step()
        mu_list.append(mu)
        cov_list.append(cov)
        print(f"Step #{kf._t}: Prob of mu = {kf.prob_of_point(kf.mu)}")

    print("End".center(40, "-"))

    pred = list(zip([x.item() for x, y in mu_list], [y.item() for x, y in mu_list]))
    sigma = 0.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharex="all", sharey="all")

    ax2.plot(x, y, "--k", label="GT", markersize=5)
    ax1.plot(
        noisy_measurements[:, 0],
        noisy_measurements[:, 1],
        "-k",
        label="Noisy Measurements",
    )
    ax2.plot(
        [x for x, _ in pred],
        [y for _, y in pred],
        "-rx",
        label="Pred",
        alpha=0.5,
        markersize=10,
    )

    # Draw confidence ellipses
    for cov, pred in zip(cov_list, pred):
        n_std = 1.0
        mean_x = pred[0]
        mean_y = pred[1]

        rad_x = np.sqrt(cov[0, 0])  # Standard deviation of x
        rad_y = np.sqrt(cov[1, 1])  # # Standard deviation of y

        ellipse = Ellipse(
            (mean_x, mean_y),
            width=rad_x * sigma,
            height=rad_y * sigma,
            fill=False,
            linestyle="--",
            edgecolor="blue",
            alpha=0.5,
        )
        ax2.add_patch(ellipse)

    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax2.set_xlabel("$x$")
    fig.suptitle(f"Temporal Model = Simple Brownian, $\sigma = {sigma}$")

    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    plt.show()
