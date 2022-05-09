"""
Evaluates the performance of the ball detector
"""
import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ai_umpire import BallDetector
from ai_umpire.util import (
    wc_to_ic,
    load_sim_ball_pos,
)

ROOT_DIR_PATH = Path() / "data"
VID_DIR_PATH: Path = ROOT_DIR_PATH / "videos"
SIM_LEN = 2.0  # In seconds
SIM_STEP_SZ = 0.005  # In seconds
N_RENDERED_FRAMES = int(SIM_LEN / SIM_STEP_SZ)
DESIRED_FPS = 50
N_FRAMES_TO_AVG = int(N_RENDERED_FRAMES / DESIRED_FPS)

plt.rcParams["figure.figsize"] = (8, 4.5)

if __name__ == "__main__":
    for i in range(4):
        # Generate video from simulation frames if it does not already exist
        video_fname: str = f"sim_{i}.mp4"
        # video_fname: str = "sim_0_comparable.mp4"
        if not (VID_DIR_PATH / video_fname).exists():
            raise FileNotFoundError(f"Video file for sim ID {i}not found.")

        detector = BallDetector(ROOT_DIR_PATH)
        filtered_dets = detector.get_filtered_ball_detections(
            vid_fname=video_fname,
            sim_id=i,
            morph_op="close",
            morph_op_iters=1,
            morph_op_se_shape=(20, 20),
            blur_kernel_size=(21, 21),
            blur_sigma=1,
            binary_thresh=120,
            struc_el_shape=cv.MORPH_RECT,
            min_ball_travel_dist=2,
            max_ball_travel_dist=70,
            min_det_area=2,
            max_det_area=25,
            disable_progbar=False,
            visualise=["none"],
        )

        ball_pos_true = load_sim_ball_pos(i, ROOT_DIR_PATH, N_FRAMES_TO_AVG)

        # Obtain true ball positions in image space
        euclid_dists = []
        true_ball_pos_IC = []
        for j in range(len(filtered_dets)):
            ball_x_ic, ball_y_ic = wc_to_ic(
                ball_pos_true[j],
                [720, 1280],
            )

            true_ball_pos_IC.append((ball_x_ic, ball_y_ic, ball_pos_true[i][2]))

            dist_to_true = math.sqrt(
                ((ball_x_ic - filtered_dets[j][0]) ** 2)
                + ((ball_y_ic - filtered_dets[j][1]) ** 2)
            )
            euclid_dists.append(dist_to_true)

        # Plot true ball pos and filtered detections
        first_frame = cv.imread(
            str(ROOT_DIR_PATH / "frames" / f"sim_{i}" / "frame00000.jpg")
        )
        plt.imshow(cv.cvtColor(first_frame, cv.COLOR_BGR2RGB))

        # Visualise detection performance
        plt.plot(
            [x for x, y, z in true_ball_pos_IC],
            [y for _, y, _ in true_ball_pos_IC],
            label="Ball True",
            color="green",
        )

        plt.scatter(
            [x for x, y, z in filtered_dets],
            [y for _, y, _ in filtered_dets],
            label="Detections",
            color="blue",
            alpha=0.5,
            marker="x",
        )

        plt.tight_layout()
        plt.legend()
        plt.axis("off")
        plt.show()

        # Quantify z estimate performance in terms of pearson corr
        z_estimate_corr = (
            pd.DataFrame(
                {
                    "z": ball_pos_true[:, 2][:-1],
                    "Z Estimate": np.sqrt(np.array([z for _, _, z in filtered_dets])),
                }
            )
            .corr()
            .iloc[0]["Z Estimate"]
        )

        print(f"Sim ID {i}".ljust(60, "-"))
        print(f"  Mean Euclid distance   = {sum(euclid_dists) / len(euclid_dists):.4f}")
        print(f"  Z estimate correlation = {z_estimate_corr:.4f}")
