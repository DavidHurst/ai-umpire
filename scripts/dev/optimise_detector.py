import math
import random
from pathlib import Path
from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_umpire import VideoGenerator, BallDetector
from ai_umpire.util import wc_to_ic

ROOT_DIR_PATH = Path() / "data"
SIM_ID = 0
FRAMES_DIR_PATH: Path = ROOT_DIR_PATH / "frames" / f"sim_{SIM_ID}"
VID_DIR_PATH: Path = ROOT_DIR_PATH / "videos"
VID_FNAME = f"sim_{SIM_ID}.mp4"
SIM_LEN = 2.0  # In seconds
SIM_STEP_SZ = 0.005  # In seconds
N_RENDERED_FRAMES = int(SIM_LEN / SIM_STEP_SZ)
DESIRED_FPS = 50
N_FRAMES_TO_AVG = int(N_RENDERED_FRAMES / DESIRED_FPS)

DIST_PENALTY = 1000
Z_ESTIMATE_PENALTY = 250


def eval_detector(
    morph_iters: int,
    morph_op_SE_shape: Tuple,
    blur_kernel_size: Tuple,
    blur_strength: int,
    binarize_thresh_low: int,
    visualise: bool = False,
) -> Tuple:
    # Generate ball candidates per frame in video
    detector = BallDetector(ROOT_DIR_PATH)
    all_detections = detector.get_ball_detections(
        vid_fname=VID_FNAME,
        sim_id=SIM_ID,
        morph_op="close",
        morph_op_iters=morph_iters,
        morph_op_se_shape=morph_op_SE_shape,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_strength,
        binary_thresh=binarize_thresh_low,
        struc_el_shape=cv.MORPH_RECT,
        disable_progbar=True,
    )

    # Measure performance of detector, metric is Euclidean distance for x and y,
    # correlation between z surrogate and true z for z
    avg_euclid_dists = []
    min_euclid_dists = []
    max_euclid_dists = []

    z_surrogate_closest_dets = []
    for i in range(len(all_detections)):
        # Calculate Euclidean distances from true pos to detected positions
        ball_pos_true = np.array(
            [
                ball_pos_blurred_WC["x"][i],
                ball_pos_blurred_WC["y"][i],
                ball_pos_blurred_WC["z"][i],
            ]
        )
        ball_x_ic, ball_y_ic = wc_to_ic(
            ball_pos_true,
            [720, 1280],
        )
        euclid_dists = [
            math.sqrt(((det_x - ball_x_ic) ** 2) + ((det_y - ball_y_ic) ** 2))
            if det_x != -1
            else DIST_PENALTY
            for (det_x, det_y, _) in all_detections[i]
        ]

        avg_euclid_dists.append(sum(euclid_dists) / len(euclid_dists))
        min_euclid_dists.append(min(euclid_dists))
        max_euclid_dists.append(max(euclid_dists))

        # Penalise z if no detection made.
        if euclid_dists[euclid_dists.index(min(euclid_dists))] == DIST_PENALTY:
            z_surrogate_closest_dets.append(Z_ESTIMATE_PENALTY)
        else:
            frame_z_estimates = [z_estim for (_, _, z_estim) in all_detections[i]]
            closest_det_idx = euclid_dists.index(min(euclid_dists))
            z_surrogate_closest_dets.append(frame_z_estimates[closest_det_idx])

    mean_mean_dist = sum(avg_euclid_dists) / len(avg_euclid_dists)
    mean_min_dist = sum(min_euclid_dists) / len(min_euclid_dists)
    sqrt_z_surrogate_closest_dets = np.sqrt(np.array(z_surrogate_closest_dets))

    ball_true_z = ball_pos_blurred_WC["z"].to_numpy()[:-1]

    # Compute correlation, evaluation of z estimate
    closest_dets_corr = (
        pd.DataFrame(
            {
                "z": ball_true_z,
                "Best Detection": sqrt_z_surrogate_closest_dets,
            }
        )
        .corr()
        .iloc[0]["Best Detection"]
    )

    if visualise:
        # Plot performance
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        frame_nums = np.arange(0, len(all_detections))
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

        axes[1].scatter(
            ball_pos_blurred_WC["z"].to_numpy()[:-1],
            sqrt_z_surrogate_closest_dets,
            marker="x",
            color="green",
            label=f"z vs. Closest Detection - Corr.={closest_dets_corr:.2f}",
            alpha=0.5,
        )
        axes[1].set_title("Detection Error - z (True z vs. Contour Area (Sqrt.))")
        axes[1].set_xlabel("z")
        axes[1].set_ylabel("Contour Area")

        fig.suptitle("Contour Detection Performance")
        for ax in axes:
            ax.legend()
        plt.show()

    return mean_mean_dist, mean_min_dist, closest_dets_corr


if __name__ == "__main__":
    # Generate video from simulation frames if it does not already exist
    if not (VID_DIR_PATH / f"sim_{SIM_ID}.mp4").exists():
        print(f"Generating video for sim id {SIM_ID}")
        vid_gen = VideoGenerator(root_dir=ROOT_DIR_PATH)
        vid_gen.convert_frames_to_vid(SIM_ID, 50)

    # Check that ball data file exists
    data_file_path = ROOT_DIR_PATH / "ball_pos" / f"sim_{SIM_ID}.csv"
    if not data_file_path.exists():
        raise IOError("Data file not found")

    ball_pos_WC = pd.DataFrame(pd.read_csv(data_file_path), columns=["x", "y", "z"])
    ball_pos_blurred_WC = ball_pos_WC.iloc[
        N_FRAMES_TO_AVG::N_FRAMES_TO_AVG, :
    ].reset_index(drop=True)

    # Perform random search of hyperparameters
    rng = np.random.default_rng()

    morph_iters_set_set = list(rng.integers(1, 3, 3))

    sizes = rng.integers(10, 40, 4)
    morph_op_SE_shape_set = list(zip(sizes, sizes))

    sizes = [random.randrange(11, 51, 10) for _ in range(4)]
    blur_kernel_size_set = list(zip(sizes, sizes))

    blur_strength_set = list(rng.integers(1, 4, 3))

    binarize_thresh_low_set = [random.randrange(100, 130, 10) for _ in range(2)]

    print("Randomly chosen hyperparameter values:")
    print("Opening iterations: ".ljust(35, " "), morph_iters_set_set)
    print("Morph. op. shape: ".ljust(35, " "), morph_op_SE_shape_set)
    print("Blur kernel size:".ljust(35, " "), blur_kernel_size_set)
    print("Blur strength:".ljust(35, " "), blur_strength_set)
    print("Lower bound of binary threshold:".ljust(35, " "), binarize_thresh_low_set)

    n_configs = np.prod(
        [
            len(morph_iters_set_set),
            len(morph_op_SE_shape_set),
            len(blur_kernel_size_set),
            len(blur_strength_set),
            len(binarize_thresh_low_set),
        ]
    )

    hparam_config = 1
    optimal_model_params = {
        "morph_iters": 0,
        "morph_op_SE_shape": (),
        "blur_kernel_size": (),
        "blur_strength": 0,
        "binarize_thresh_low": 0,
    }
    optimal_model_scores = {
        "mean_mean_dist": float("inf"),
        "closest_dets_corr": float("-inf"),
    }
    objective_scaling_values = np.array([0.1, 100])
    objective_weights = np.array([1, 0.7])

    _, _, _ = eval_detector(
        morph_iters=1,
        morph_op_SE_shape=(21, 21),
        blur_kernel_size=(33, 33),
        blur_strength=1,
        binarize_thresh_low=120,
        visualise=True,
    )

    exit()

    # Scalarise optimisation objectives to remove the need for multi-objective optimisation, maximising objective here
    scalarised_objective = float("-inf")
    for morph_iters in morph_iters_set_set:
        for SE_shape in morph_op_SE_shape_set:
            for kernel_sz in blur_kernel_size_set:
                for blur_strength in blur_strength_set:
                    for thresh in binarize_thresh_low_set:
                        param_vals = (
                            f"morph_iters:{morph_iters}, SE_shape:{SE_shape}, kernel_sz:{kernel_sz}, "
                            f"blur_strength:{blur_strength}, thresh:{thresh}"
                        )
                        print(
                            f"Trial configuration #{hparam_config}/{n_configs}".ljust(
                                80, "-"
                            ),
                            f"\nParams = {param_vals}",
                        )

                        # Run and score detector
                        mean_dist, _, z_corr = eval_detector(
                            morph_iters=morph_iters,
                            morph_op_SE_shape=SE_shape,
                            blur_kernel_size=kernel_sz,
                            blur_strength=blur_strength,
                            binarize_thresh_low=thresh,
                            visualise=False,
                        )

                        curr_scalarised_objectives = np.sum(
                            np.array([-mean_dist, z_corr])
                            * objective_scaling_values
                            * objective_weights
                        )
                        print(
                            "Metrics Scaled + Weighted".ljust(25, " "),
                            np.array([-mean_dist, z_corr])
                            * objective_scaling_values
                            * objective_weights,
                        )
                        print(
                            f"Scalarized objective (maximising): "
                            f"\n   Current: {curr_scalarised_objectives:.4f}"
                            f"\n   Best:    {scalarised_objective:.4f}"
                        )

                        if curr_scalarised_objectives > scalarised_objective:
                            print(f"[i] New best configuration found.")
                            optimal_model_params = {
                                "morph_iters": morph_iters,
                                "morph_op_SE_shape": SE_shape,
                                "blur_kernel_size": kernel_sz,
                                "blur_strength": blur_strength,
                                "binarize_thresh_low": thresh,
                            }
                            optimal_model_scores = {
                                "mean_mean_dist": mean_dist,
                                "closest_dets_corr": z_corr,
                            }
                            scalarised_objective = curr_scalarised_objectives

                        hparam_config += 1
                        print("-" * 80, "\n")

    print("Optimal hyperparameter values found by grid search:\n", optimal_model_params)
    print("Optimal model performance:\n", optimal_model_scores)
    print(f"Optimal model scalarised objective score = {scalarised_objective}")
    with open("optimal_model_params.txt", "w") as f:
        f.write(str(optimal_model_params))

    _, _, _ = eval_detector(
        morph_iters=optimal_model_params["morph_iters"],
        morph_op_SE_shape=optimal_model_params["morph_op_SE_shape"],
        blur_kernel_size=optimal_model_params["blur_kernel_size"],
        blur_strength=optimal_model_params["blur_strength"],
        binarize_thresh_low=optimal_model_params["binarize_thresh_low"],
        visualise=True,
    )
