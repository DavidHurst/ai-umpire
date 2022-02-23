import math
import statistics
from pathlib import Path
from typing import Tuple, List

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai_umpire import SimVideoGen, Localiser
from ai_umpire.util import wc_to_ic

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

    # Check that ball data file exists
    data_file_path = root_dir_path / "ball_pos" / f"sim_{sim_id}.csv"
    if not data_file_path.exists():
        raise IOError("Data file not found")

    ball_pos_WC = pd.DataFrame(pd.read_csv(data_file_path), columns=["x", "y", "z"])
    ball_pos_blurred_WC = ball_pos_WC.iloc[
        n_frames_to_avg::n_frames_to_avg, :
    ].reset_index(drop=True)

    # Visualise blurred pos
    # for i in range(len(ball_pos_blurred_WC)):
    #     fig, axes = plt.subplots(figsize=(15, 7))
    #     im1 = cv2.cvtColor(
    #         cv2.imread(
    #             str(sim_blurred_frames_path / f"frame{str(i).zfill(5)}.png"),
    #             cv2.IMREAD_COLOR,
    #         ),
    #         cv2.COLOR_BGR2RGB,
    #     )
    #
    #     ball_x_ic, ball_y_ic = wc_to_ic(
    #         ball_pos_blurred_WC["x"][i], ball_pos_blurred_WC["y"][i], ball_pos_blurred_WC["z"][i], [1024, 768]
    #     )
    #
    #     axes.annotate(
    #         f'(x_IC, y_IC, z_WC) = ({ball_x_ic:.2f}, {ball_y_ic:.2f}, {ball_pos_blurred_WC["z"][i]:.2f})',
    #         (ball_x_ic, ball_y_ic),
    #         xytext=(-300, 0),
    #         textcoords="offset points",
    #         bbox=dict(boxstyle="round", fc="0.7"),
    #         arrowprops=dict(arrowstyle="->", color="green", linewidth=2),
    #     )
    #
    #     axes.imshow(im1)
    #     axes.set_title(f"Frame #{i}")
    #     axes.axis("off")
    #     plt.tight_layout()
    #     # plt.savefig(f"ball_true_{i}.png")
    #     plt.show()

    # Generate ball candidates per frame in video
    loc = Localiser(root_dir_path)
    frame_detections = loc.get_ball_candidates_contour(
        sim_id=sim_id,
        morph_op="open",
        morph_op_iters=1,
        morph_op_SE_shape=(5, 5),
        blur_kernel_size=(51, 51),
        blur_sigma_x=2,
        binary_thresh_low=235,
    )

    # Measure performance of detector, metric is Euclidean distance for x and y
    avg_euclid_dists = []
    min_euclid_dists = []
    max_euclid_dists = []

    min_z_estimate = []
    closest_det_z_estimate = []
    for i in range(len(frame_detections)):
        # Calculate Euclidean distances from true pos to detected positions
        ball_x_ic, ball_y_ic = wc_to_ic(
            ball_pos_blurred_WC["x"][i],
            ball_pos_blurred_WC["y"][i],
            ball_pos_blurred_WC["z"][i],
            [1024, 768],
        )
        euclid_dists = [
            math.sqrt(((det_x - ball_x_ic) ** 2) + ((det_y - ball_y_ic) ** 2))
            for (det_x, det_y, _) in frame_detections[i]
        ]

        avg_euclid_dists.append(sum(euclid_dists) / len(euclid_dists))
        min_euclid_dists.append(min(euclid_dists))
        max_euclid_dists.append(max(euclid_dists))

        frame_z_estimates = [z_est for (_, _, z_est) in frame_detections[i]]
        closest_det_idx = euclid_dists.index(min(euclid_dists))

        min_z_estimate.append(min(frame_z_estimates))
        closest_det_z_estimate.append(frame_z_estimates[closest_det_idx])

    # Plot performance
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    frame_nums = np.arange(0, len(frame_detections))
    axes[0].plot(frame_nums, avg_euclid_dists, "b-", label="Mean")
    axes[0].fill_between(
        frame_nums,
        min_euclid_dists,
        max_euclid_dists,
        label="Range",
        color="blue",
        alpha=0.2,
        edgecolor="green",
    )

    axes[0].set_title("Detection Error - (x, y)")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Euclidean Distance Between True & Det. (Pixels)")

    # Compute correlation
    z_df = pd.DataFrame(
        {
            "z": ball_pos_blurred_WC["z"].to_numpy()[:-1],
            "Area Min": min_z_estimate,
            "Best Detection": closest_det_z_estimate,
        }
    )
    area_min_corr = z_df.corr().iloc[0]["Area Min"]
    best_det_corr = z_df.corr().iloc[0]["Best Detection"]

    axes[1].scatter(
        ball_pos_blurred_WC["z"].to_numpy()[:-1],
        min_z_estimate,
        marker="x",
        color="red",
        label=f"z vs. Contour w/ Min Area - Corr.={area_min_corr:.2f}",
        alpha=0.5,
    )
    axes[1].scatter(
        ball_pos_blurred_WC["z"].to_numpy()[:-1],
        closest_det_z_estimate,
        marker="x",
        color="green",
        label=f"z vs. Closest Detection - Corr.={best_det_corr:.2f}",
        alpha=0.5,
    )
    axes[1].set_title("Detection Error - z (True z vs. Contour Area)")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("Contour Area")

    fig.suptitle("Contour Detection Performance")
    for ax in axes:
        ax.legend()
    plt.show()
