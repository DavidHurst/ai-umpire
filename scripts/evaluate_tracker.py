from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ai_umpire import SimVideoGen, Tracker, KalmanFilter

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 5
sim_frames_path: Path = root_dir_path / "sim_frames" / f"sim_{sim_id}_frames"
sim_blurred_frames_path: Path = (
    root_dir_path / "blurred_frames" / f"sim_{sim_id}_blurred"
)
vid_dir: Path = root_dir_path / "videos"
sim_length = 2.0
sim_step_sz = 0.005
n_rendered_frames = int(sim_length / sim_step_sz)
desired_fps = 50
n_frames_to_avg = int(n_rendered_frames / desired_fps)


if __name__ == "__main__":
    # Generate video from simulation frames if it does not already exist
    if not (vid_dir / f"sim_{sim_id}.mp4").exists():
        print(f"Generating video for sim id {sim_id}")
        vid_gen = SimVideoGen(root_dir=root_dir_path)
        vid_gen.convert_frames_to_vid(sim_id, 50)

    # Check that ball data file exisits
    data_file_path = root_dir_path / "ball_pos" / f"sim_{sim_id}.csv"
    if not data_file_path.exists():
        raise IOError("Data file not found")

    col_names = ["x", "y", "z"]
    ball_pos = pd.DataFrame(pd.read_csv(data_file_path), columns=col_names)

    x = [
        -393.66,
        -375.93,
        -351.04,
        -328.96,
        -299.35,
        -273.36,
        -245.89,
        -222.58,
        -198.03,
        -174.17,
        -146.32,
        -123.72,
        -103.47,
        -78.23,
        -52.63,
        -23.34,
        25.96,
        49.72,
        76.94,
        95.38,
        119.83,
        144.01,
        161.84,
        180.56,
        201.42,
        222.62,
        239.4,
        252.51,
        266.26,
        271.75,
        277.4,
        294.12,
        301.23,
        291.8,
        299.89,
    ]
    y = [
        300.4,
        301.78,
        295.1,
        305.19,
        301.06,
        302.05,
        300,
        303.57,
        296.33,
        297.65,
        297.41,
        299.61,
        299.6,
        302.39,
        295.04,
        300.09,
        294.72,
        298.61,
        294.64,
        284.88,
        272.82,
        264.93,
        251.46,
        241.27,
        222.98,
        203.73,
        184.1,
        166.12,
        138.71,
        119.71,
        100.41,
        79.76,
        50.62,
        32.99,
        2.14,
    ]
    ball_x = ball_pos["x"]
    ball_y = ball_pos["y"]

    kf = KalmanFilter(num_variables=6, init_estimate_uncertainty=10)

    measurements_example = np.array([np.array([[x[i]], [y[i]]]) for i in range(len(x))])
    ball_true = np.array([[[ball_x[i]], [ball_y[i]]] for i in range(len(ball_x))])

    noise = 0 + 0.02 * np.random.randn(
        ball_true.shape[0], ball_true.shape[1], ball_true.shape[2]
    )
    measurements_ball = ball_true + noise

    preds = kf.process_measurements(measurements_ball)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5), sharey="all", sharex="all")
    # axes.plot(x, y, '--g', label="Measurements")
    # axes.plot([pos[0] for pos in preds], [pos[1] for pos in preds], '-+b', label='Estimate')
    axes[0].plot(ball_x, ball_y, "-g", label="True")
    axes[0].set_title("True Position")

    axes[1].plot(
        [p[0] for p in measurements_ball],
        [p[1] for p in measurements_ball],
        "-b",
        label="Measurement",
    )
    axes[1].set_title("Noisy Measurement (True + N(0, 0.02)")

    axes[2].plot(
        [pos[0] for pos in preds],
        [pos[1] for pos in preds],
        "-r",
        label="Estimate",
    )
    axes[2].set_title("Estimated Position")

    for i in range(0, len(ball_x), 20):
        axes[0].annotate(i, (ball_x[i], ball_y[i]))

    for ax in axes:
        ax.legend()
        ax.set_ylabel("y")
        ax.set_xlabel("x")

    plt.tight_layout()
    plt.show()
