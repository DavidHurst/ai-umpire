from pathlib import Path

import numpy as np

from ai_umpire import KalmanFilter, TrajectoryInterpreter
from ai_umpire.util import get_sim_ball_pos, approximate_homography, get_init_ball_pos

ROOT_DIR_PATH = Path() / "data"
SIM_ID = 0
SIM_FRAMES_PATH: Path = ROOT_DIR_PATH / "frames" / f"sim_{SIM_ID}"
SIM_LENGTH: float = 2.0
SIM_STEP_SIZE: float = 0.005
N_RENDERED_IMAGES: int = int(SIM_LENGTH / SIM_STEP_SIZE)
DESIRED_FPS: int = 50
N_FRAMES_TO_AVERAGE: int = int(N_RENDERED_IMAGES / DESIRED_FPS)

if __name__ == '__main__':
    vid_dir_path = ROOT_DIR_PATH / "videos"
    vid_fname = f"sim_{SIM_ID}.mp4"
    ball_pos_true = get_sim_ball_pos(SIM_ID, ROOT_DIR_PATH, N_FRAMES_TO_AVERAGE)

    rng = np.random.default_rng()

    # Add noise to true ball positions to simulate measurements
    noisy_gt = np.copy(ball_pos_true) + rng.normal(
        0, 0.5, size=(ball_pos_true.shape[0], 3)
    )

    # Obtain initial ball position (will be in image coords) and project into world coords
    init_ball_pos_ic = get_init_ball_pos(vid_dir_path, vid_fname)
    h = approximate_homography(video_path=vid_dir_path / vid_fname)
    init_ball_pos_ic_homog = np.reshape(np.append(init_ball_pos_ic, 1), (3, 1))
    init_ball_pos_wc = h @ init_ball_pos_ic_homog
    init_ball_pos_wc /= init_ball_pos_wc[-1]
    init_ball_pos_wc = np.array(init_ball_pos_wc) * 5  # Scale
    init_ball_pos_wc[-1] = 0
    init_ball_pos_wc = np.append(init_ball_pos_wc, np.zeros(6))

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
    transition_noise = rng.normal(0, 0.035, size=(states_dim, states_dim))
    sigma_p = np.identity(states_dim) + transition_noise

    # Define measurement model's parameters
    phi = np.zeros((measurements_dim, states_dim))
    for i in range(phi.shape[0]):  # Set main diagonal to ones
        phi[i, i] = 1
    mu_m = np.zeros((measurements_dim, 1))
    measurement_noise = rng.normal(0, 0.07, size=(measurements_dim, measurements_dim))
    sigma_m = (np.identity(measurements_dim) * 30) + measurement_noise

    kf = KalmanFilter(
        init_mu=init_ball_pos_wc.reshape((states_dim, 1)),
        measurements=noisy_gt,
        sigma_m=sigma_m,
        sigma_p=sigma_p,
        phi=phi,
        psi=psi,
        mu_m=mu_m,
        mu_p=mu_p,
    )

    ti = TrajectoryInterpreter(
        kalman_filter=kf, n_dim_samples=[5, 3, 5], n_std_devs_to_sample=1
    )
    trajectory_label = ti.classify_trajectory(0.65, visualise=False)

    print(f"The trajectory has been interpreted as - {trajectory_label}")
