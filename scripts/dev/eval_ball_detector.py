import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ai_umpire import BallDetector
from ai_umpire.util import (
    wc_to_ic,
    get_sim_ball_pos,
)

ROOT_DIR_PATH = Path() / "data"
SIM_ID = 0
VID_DIR_PATH: Path = ROOT_DIR_PATH / "videos"
SIM_LEN = 2.0  # In seconds
SIM_STEP_SZ = 0.005  # In seconds
N_RENDERED_FRAMES = int(SIM_LEN / SIM_STEP_SZ)
DESIRED_FPS = 50
N_FRAMES_TO_AVG = int(N_RENDERED_FRAMES / DESIRED_FPS)

plt.rcParams["figure.figsize"] = (8, 4.5)

if __name__ == "__main__":
    # Generate video from simulation frames if it does not already exist
    video_fname: str = f"sim_{SIM_ID}.mp4"
    # video_fname: str = "sim_0_comparable.mp4"
    if not (VID_DIR_PATH / video_fname).exists():
        raise FileNotFoundError(f"Video file for sim ID {SIM_ID}not found.")

    # {'morph_iters': 1, 'morph_op_SE_shape': (20, 20), 'blur_kernel_size': (33, 33), 'blur_strengt
    #     h': 1, 'binarize_thresh_low': 110}
    # {'morph_iters': 1, 'morph_op_SE_shape': (22, 22), 'blur_kernel_size': (11, 11), 'blur_strengt
    #     h': 2, 'binarize_thresh_low': 110}

    detector = BallDetector(ROOT_DIR_PATH)
    filtered_dets = detector.get_filtered_ball_detections(
        vid_fname=video_fname,
        sim_id=SIM_ID,
        morph_op="close",
        morph_op_iters=1,
        morph_op_se_shape=(21, 21),
        blur_kernel_size=(33, 33),
        blur_sigma=1,
        binary_thresh=120,
        struc_el_shape=cv.MORPH_RECT,
        min_ball_travel_dist=1,
        max_ball_travel_dist=70,
        min_det_area=1,
        max_det_area=30,
        disable_progbar=False,
    )

    ball_pos_true = get_sim_ball_pos(SIM_ID, ROOT_DIR_PATH, N_FRAMES_TO_AVG)

    # Obtain true ball positions in image space
    euclid_dists = []
    true_ball_pos_IC = []
    for i in range(len(filtered_dets)):
        ball_x_ic, ball_y_ic = wc_to_ic(
            ball_pos_true[i],
            [720, 1280],
        )

        true_ball_pos_IC.append((ball_x_ic, ball_y_ic, ball_pos_true[i][2]))

        dist_to_true = math.sqrt(
            ((ball_x_ic - filtered_dets[i][0]) ** 2)
            + ((ball_y_ic - filtered_dets[i][1]) ** 2)
        )
        euclid_dists.append(dist_to_true)

    # Plot true ball pos and filtered detections
    first_frame = cv.imread(
        str(ROOT_DIR_PATH / "frames" / f"sim_{SIM_ID}" / "frame00000.jpg")
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

    # Quantify x,y performance in terms of Euclidean distance from true
    print(
        f"Mean Euclid dist for sim id {SIM_ID} = {sum(euclid_dists) / len(euclid_dists)}"
    )

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
    print(
        f"Pearson's correlation coefficient between true z and estimated for sim id {SIM_ID} = {z_estimate_corr}"
    )
