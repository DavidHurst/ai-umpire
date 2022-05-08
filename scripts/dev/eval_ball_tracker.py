import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ai_umpire import KalmanFilter
from ai_umpire.util import (
    FIELD_BOUNDING_BOXES,
    plot_bb,
    get_sim_ball_pos,
    get_init_ball_pos,
    approximate_homography,
)

ROOT_DIR_PATH = Path() / "data"
SIM_ID = 0
SIM_FRAMES_PATH: Path = ROOT_DIR_PATH / "frames" / f"sim_{SIM_ID}"
SIM_LENGTH: float = 2.0
SIM_STEP_SIZE: float = 0.005
N_RENDERED_IMAGES: int = int(SIM_LENGTH / SIM_STEP_SIZE)
DESIRED_FPS: int = 50
N_FRAMES_TO_AVERAGE: int = int(N_RENDERED_IMAGES / DESIRED_FPS)

plt.rcParams["figure.figsize"] = (5.5, 4.5)

if __name__ == "__main__":
    vid_dir_path = ROOT_DIR_PATH / "videos"
    vid_fname = f"sim_{SIM_ID}.mp4"
    ball_pos_true = get_sim_ball_pos(SIM_ID, ROOT_DIR_PATH, N_FRAMES_TO_AVERAGE)

    rng = np.random.default_rng()

    # Add noise to true ball positions to simulate measurements
    noisy_gt = np.copy(ball_pos_true) + rng.normal(
        0, 0.5, size=(ball_pos_true.shape[0], 3)
    )

    # Obtain initial ball position (will be in image coords) and project into world coords
    # init_ball_pos_ic = get_init_ball_pos(vid_dir_path, vid_fname)
    # h = approximate_homography(video_path=vid_dir_path / vid_fname)
    # init_ball_pos_ic_homog = np.reshape(np.append(init_ball_pos_ic, 1), (3, 1))
    # init_ball_pos_wc = h @ init_ball_pos_ic_homog
    # init_ball_pos_wc /= init_ball_pos_wc[-1]
    # init_ball_pos_wc = np.array(init_ball_pos_wc) * 5  # Scale
    # init_ball_pos_wc[-1] = 0
    init_ball_pos_wc = np.append(np.array([-1.71358755, -0.85250551, 0.]), np.ones(6))

    # Kalman filter will track ball over time and smooth noise in the detections_IC
    measurements_dim = 3
    states_dim = 3 * 3
    delta_t = 1 / DESIRED_FPS

    # Define the temporal model's parameters
    psi = np.identity(states_dim)
    psi[0, 1] = delta_t
    psi[0, 2] = 0.5 * (delta_t ** 2)
    psi[1, 2] = delta_t
    psi[3, 4] = delta_t
    psi[3, 4] = 0.5 * (delta_t ** 2)
    psi[4, 5] = delta_t
    psi[6, 7] = delta_t
    psi[6, 7] = delta_t
    psi[7, 8] = 0.5 * (delta_t ** 2)
    psi[4, 5] = delta_t
    mu_p = np.zeros((states_dim, 1))
    sigma_p = np.identity(states_dim)

    # Define measurement model's parameters
    phi = np.zeros((measurements_dim, states_dim))
    for i in range(phi.shape[0]):
        phi[i, i] = 1
    mu_m = np.zeros((measurements_dim, 1))
    sigma_m = np.identity(measurements_dim) * 50

    print("KF Internal model's parameters".ljust(70, "-"))
    print(f"psi: \n{psi}:")
    print(f"mu_p: \n{mu_p}:")
    print(f"sigma_p: \n{sigma_p}:")
    print(f"phi: \n{phi}:")
    print(f"mu_m: \n{mu_m}:")
    print(f"sigma_m: \n{sigma_m}:")

    kf = KalmanFilter(
        init_mu=init_ball_pos_wc.reshape((states_dim, 1)),
        n_variables=states_dim,
        measurements=noisy_gt,
        sigma_m=sigma_m,
        sigma_p=sigma_p,
        phi=phi,
        psi=psi,
        mu_m=mu_m,
        mu_p=mu_p,
    )

    mean_tracking_err = 0.0
    mean_noisy_gt_err = 0.0
    state_pos_preds = []
    for i in range(noisy_gt.shape[0]):
        mu, cov = kf.step()

        gt_x = ball_pos_true[i][0]
        gt_y = ball_pos_true[i][1]
        gt_z = ball_pos_true[i][2]

        state_pos_preds.append(mu[:3].T.squeeze())

        # Calculate KF prediction error
        tracking_error = math.sqrt(
            ((gt_x - mu[0]) ** 2)
            + ((gt_y - mu[1]) ** 2)
            + ((gt_z - mu[2]) ** 2)
        )
        noisy_gt_error = math.sqrt(
            ((gt_x - noisy_gt[i][0]) ** 2)
            + ((gt_y - noisy_gt[i][1]) ** 2)
            + ((gt_z - noisy_gt[i][2]) ** 2)
        )

        mean_tracking_err += tracking_error / noisy_gt.shape[0]
        mean_noisy_gt_err += noisy_gt_error / noisy_gt.shape[0]
    state_pos_preds = np.array(state_pos_preds)

    print(f"Mean tracking error = {mean_tracking_err:.4f}")
    print(f"Noisy gt error      = {mean_noisy_gt_err:.4f}")

    fig = plt.figure()
    ax = Axes3D(fig, elev=15, azim=-140, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.grid(False)
    ax.set_xlim3d(-4, 4)
    ax.set_zlim3d(0, 7)
    ax.set_ylim3d(-6, 6)
    ax.set_xlabel("$x$")
    ax.set_zlabel("$y$")
    ax.set_ylabel("$z$")

    for bb_name in FIELD_BOUNDING_BOXES.keys():
        if not bb_name.startswith(("left", "back")):
            plot_bb(
                bb_name=bb_name,
                ax=ax,
                bb_face_annotation="",
                show_vertices=False,
                show_annotation=False,
            )

    ax.plot3D(
        ball_pos_true[:, 0],
        ball_pos_true[:, 1],
        ball_pos_true[:, 2],
        "--",
        label="GT",
        alpha=0.5,
        c="g",
        zdir="y",
    )
    ax.plot3D(
        state_pos_preds[:, 0],
        state_pos_preds[:, 1],
        state_pos_preds[:, 2],
        "-o",
        label="KF Preds.",
        alpha=0.5,
        c="r",
        zdir="y",
    )
    ax.plot3D(
        noisy_gt[:, 0],
        noisy_gt[:, 1],
        noisy_gt[:, 2],
        "-x",
        label="Noisy GT",
        alpha=0.5,
        c="b",
        zdir="y",
    )
    plt.legend()
    # plt.savefig("eval_ball_tracker.png")
    plt.show()
