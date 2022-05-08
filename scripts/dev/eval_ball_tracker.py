from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ai_umpire import KalmanFilter
from ai_umpire.util import (
    FIELD_BOUNDING_BOXES,
    plot_bb, get_sim_ball_pos, get_init_ball_pos, approximate_homography,
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
    noisy_gt = np.copy(ball_pos_true) + rng.normal(0, 0.3, size=(ball_pos_true.shape[0], 3))
    print(noisy_gt[0].shape[0])

    # Obtain initial ball position (will be in image coords) and project into world coords
    init_ball_pos_ic = get_init_ball_pos(vid_dir_path, vid_fname)
    h = approximate_homography(video_path=vid_dir_path / vid_fname)
    init_ball_pos_ic_homog = np.reshape(np.append(init_ball_pos_ic, 1), (3, 1))
    init_ball_pos_wc = h @ init_ball_pos_ic_homog
    init_ball_pos_wc /= init_ball_pos_wc[-1]
    init_ball_pos_wc = np.array(init_ball_pos_wc) * 5  # Scale
    init_ball_pos_wc[-1] = 0

    # Kalman filter will track ball over time and smooth noise in the detections_IC
    n_variables = 3
    n_measurement_vals = noisy_gt[0].shape[0]
    mu_p = np.zeros((n_variables, 1))
    mu_m = np.zeros((n_measurement_vals, 1))
    psi = np.identity(n_variables)
    phi = np.eye(n_measurement_vals, n_variables)
    # sigma_p = np.ones((n_variables, n_variables))
    # sigma_m = np.ones((n_variables, n_variables))
    sigma_p = np.identity(n_variables) * 3
    sigma_m = np.identity(n_variables) * 4

    # kf = KalmanFilter(
    #     init_mu=init_ball_pos_wc,
    #     n_variables=n_variables,
    #     measurements=noisy_gt,
    #     sigma_m=sigma_m,
    #     sigma_p=sigma_p,
    #     phi=phi,
    #     psi=psi,
    #     mu_m=mu_m,
    #     mu_p=mu_p,
    # )
    #
    # mu, cov = kf.step()
    # print(f"mu = {mu}")
    # print(f"cov = {cov}")

    # tracking_error = math.sqrt(
    #     ((x[i] - reproj_pt[0]) ** 2)
    #     + ((y[i] - reproj_pt[1]) ** 2)
    #     + ((z[i] - reproj_pt[2]) ** 2)
    # )

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
        zorder=4,
    )
    ax.scatter3D(
        init_ball_pos_wc[0],
        init_ball_pos_wc[1],
        init_ball_pos_wc[2],
        label="Init. Ball Pos.",
        marker="o",
        c="r"
    )
    # ax.plot3D(
    #     noisy_gt[:, 0],
    #     noisy_gt[:, 1],
    #     noisy_gt[:, 2],
    #     "-o",
    #     label="Noisy GT",
    #     alpha=0.5,
    #     c="b",
    #     zdir="y",
    #     zorder=4,
    # )
    plt.legend()
    # plt.savefig("eval_ball_tracker.png")
    plt.show()
