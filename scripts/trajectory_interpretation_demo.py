from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace, pi, cos, sin

from ai_umpire import KalmanFilter
from ai_umpire.trajectory_interpretation import TrajectoryInterpreter

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 5
sim_length = 2.0
sim_step_sz = 0.005
n_rendered_frames = int(sim_length / sim_step_sz)
desired_fps = 50
n_frames_to_avg = int(n_rendered_frames / desired_fps)

if __name__ == "__main__":
    # Check that ball data file exists
    data_file_path = root_dir_path / "ball_pos" / f"sim_{sim_id}.csv"
    if not data_file_path.exists():
        raise IOError("Data file not found")

    ball_pos_WC = pd.DataFrame(pd.read_csv(data_file_path), columns=["x", "y", "z"])
    ball_pos_blurred_WC = ball_pos_WC.iloc[
        n_frames_to_avg::n_frames_to_avg, :
    ].reset_index(drop=True)

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
    noisy_measurements = measurements + rng.normal(0, 0.2, size=(measurements.shape[0], 2))

    n_variables = 2
    n_measurement_vals = measurements[0].shape[0]
    mu_p = np.zeros((n_variables, 1))
    mu_m = np.zeros((n_measurement_vals, 1))
    psi = np.identity(n_variables)
    phi = np.eye(
        n_measurement_vals, n_variables
    )  # Temporary, should relate data to state e.g. through a projection
    # sigma_p = np.ones((n_variables, n_variables))
    # sigma_m = np.ones((n_variables, n_variables))
    sigma_p = np.identity(n_variables) * 3
    sigma_m = np.identity(n_variables) * 4

    kf = KalmanFilter(
        measurements=noisy_measurements,
        sigma_m=sigma_m,
        sigma_p=sigma_p,
        phi=phi,
        psi=psi,
        mu_m=mu_m,
        mu_p=mu_p,
    )

    ti = TrajectoryInterpreter(kf)
    ti.in_out_prob(n_dim_samples=[6, 2], sampling_area_size=[3, 1])
