import json
import math
from pathlib import Path
import random
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai_umpire import SimVideoGen, Detector
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
img_dims = [1024, 768]

dist_penalty = 1000
z_surrogate_penalty = 250


def eval_contour_detector(
    opening_iters: int,
    morph_op_SE_shape: Tuple,
    blur_kernel_size: Tuple,
    blur_strength: int,
    binarize_thresh_low: int,
    visualise: bool = False,
) -> Tuple:
    # Generate ball candidates per frame in video
    detector = Detector(root_dir_path)
    frame_detections = detector.get_ball_candidates_contour(
        sim_id=sim_id,
        morph_op="close",
        morph_op_iters=opening_iters,
        morph_op_SE_shape=morph_op_SE_shape,
        blur_kernel_size=blur_kernel_size,
        blur_sigma_x=blur_strength,
        binary_thresh_low=binarize_thresh_low,
        struc_el_shape=cv2.MORPH_RECT,
        disable_progbar=True,
    )

    # Measure performance of detector, metric is Euclidean distance for x and y,
    # correlation between z surrogate and true z for z
    avg_euclid_dists = []
    min_euclid_dists = []
    max_euclid_dists = []

    z_surrogate_closest_dets = []
    for i in range(len(frame_detections)):
        # Calculate Euclidean distances from true pos to detected positions
        ball_x_ic, ball_y_ic = wc_to_ic(
            ball_pos_blurred_WC["x"][i],
            ball_pos_blurred_WC["y"][i],
            ball_pos_blurred_WC["z"][i],
            img_dims,
        )
        euclid_dists = [
            math.sqrt(((det_x - ball_x_ic) ** 2) + ((det_y - ball_y_ic) ** 2))
            if det_x != float("inf")
            else dist_penalty
            for (det_x, det_y, _) in frame_detections[i]
        ]

        avg_euclid_dists.append(sum(euclid_dists) / len(euclid_dists))
        min_euclid_dists.append(min(euclid_dists))
        max_euclid_dists.append(max(euclid_dists))

        if euclid_dists[euclid_dists.index(min(euclid_dists))] == dist_penalty:
            z_surrogate_closest_dets.append(z_surrogate_penalty)
        else:
            frame_z_estimates = [z_estim for (_, _, z_estim) in frame_detections[i]]
            closest_det_idx = euclid_dists.index(min(euclid_dists))
            z_surrogate_closest_dets.append(frame_z_estimates[closest_det_idx])

    mean_mean_dist = sum(avg_euclid_dists) / len(avg_euclid_dists)
    mean_min_dist = sum(min_euclid_dists) / len(min_euclid_dists)

    # Compute correlation, evaluation of z estimate
    closest_dets_corr = (
        pd.DataFrame(
            {
                "z": ball_pos_blurred_WC["z"].to_numpy()[:-1],
                "Best Detection": z_surrogate_closest_dets,
            }
        )
        .corr()
        .iloc[0]["Best Detection"]
    )

    if visualise:
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

        axes[1].scatter(
            ball_pos_blurred_WC["z"].to_numpy()[:-1],
            z_surrogate_closest_dets,
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

    # Perform random search of hyperparameters
    # opening_iters_set_set = [6]
    # morph_op_SE_shape_set = [(8, 8)]
    # blur_kernel_size_set = [(21, 21), (41, 41), (61, 61)]
    # blur_strength_set = [7]
    # binarize_thresh_low_set = [200, 230, 250]
    rng = np.random.default_rng()

    opening_iters_set_set = list(rng.integers(1, 10, 3))

    sizes = rng.integers(2, 9, 3)
    morph_op_SE_shape_set = list(zip(sizes, sizes))

    sizes = [random.randrange(31, 91, 10) for _ in range(3)]
    blur_kernel_size_set = list(zip(sizes, sizes))

    blur_strength_set = list(rng.integers(4, 12, 3))

    binarize_thresh_low_set = [random.randrange(200, 250, 10) for _ in range(3)]

    print("Randomly chosen hyperparameter values:")
    print("Opening iterations: ".ljust(35, " "), opening_iters_set_set)
    print("Morph. op. shape: ".ljust(35, " "), morph_op_SE_shape_set)
    print("Blur kernel size:".ljust(35, " "), blur_kernel_size_set)
    print("Blur strength:".ljust(35, " "), blur_strength_set)
    print("Lower bound of binary threshold:".ljust(35, " "), binarize_thresh_low_set)

    n_configs = np.prod(
        [
            len(opening_iters_set_set),
            len(morph_op_SE_shape_set),
            len(blur_kernel_size_set),
            len(blur_strength_set),
            len(binarize_thresh_low_set),
        ]
    )

    hparam_config = 1
    optimal_model_params = {
        "opening_iters": 0,
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
    objective_weights = np.array([1, 0.8])

    # Scalarise optimisation objectives to remove the need for multi-objective optimisation, maximising objective here
    scalarised_objective = float("-inf")
    for open_iters in opening_iters_set_set:
        for SE_shape in morph_op_SE_shape_set:
            for kernel_sz in blur_kernel_size_set:
                for blur_strength in blur_strength_set:
                    for thresh in binarize_thresh_low_set:
                        param_vals = (
                            f"open_iters:{open_iters}, SE_shape:{SE_shape}, kernel_sz:{kernel_sz}, "
                            f"blur_strength:{blur_strength}, thresh:{thresh}"
                        )
                        print(
                            f"Trial configuration #{hparam_config}/{n_configs}".ljust(
                                80, "-"
                            ),
                            f"\nParams = {param_vals}",
                        )

                        # Run and score detector
                        mean_dist, _, z_corr = eval_contour_detector(
                            opening_iters=open_iters,
                            morph_op_SE_shape=SE_shape,
                            blur_kernel_size=kernel_sz,
                            blur_strength=blur_strength,
                            binarize_thresh_low=thresh,
                            visualise=False,
                        )

                        curr_scalarised_objectives = np.sum(
                            np.array([-mean_dist, abs(z_corr)])
                            * objective_scaling_values
                            * objective_weights
                        )
                        print(
                            "Metrics Scaled + Weighted".ljust(25, " "),
                            np.array([-mean_dist, abs(z_corr)])
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
                                "opening_iters": open_iters,
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

    _, _, _ = eval_contour_detector(
        opening_iters=optimal_model_params["opening_iters"],
        morph_op_SE_shape=optimal_model_params["morph_op_SE_shape"],
        blur_kernel_size=optimal_model_params["blur_kernel_size"],
        blur_strength=optimal_model_params["blur_strength"],
        binarize_thresh_low=optimal_model_params["binarize_thresh_low"],
        visualise=True,
    )
