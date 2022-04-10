from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt

from ai_umpire.util import (
    wc_to_ic,
    HALF_COURT_LENGTH,
    HALF_COURT_WIDTH,
    WALL_HEIGHT,
    CAM_EXTRINSICS_HOMOG,
)

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 5
sim_length = 2.0
sim_step_sz = 0.005
n_rendered_frames = int(sim_length / sim_step_sz)
desired_fps = 50
n_frames_to_avg = int(n_rendered_frames / desired_fps)
img_dims = [1024, 768]

if __name__ == "__main__":
    # Check that ball data file exists
    video_file_path = root_dir_path / "videos" / f"sim_{sim_id}.mp4"
    blurred_frames_file_path = (
            root_dir_path / "blurred_frames" / f"sim_{sim_id}_blurred"
    )
    if not video_file_path.exists() or not blurred_frames_file_path.exists():
        raise IOError("Data file not found")

    # Check that ball data file exists
    data_file_path = root_dir_path / "ball_pos" / f"sim_{sim_id}.csv"
    if not data_file_path.exists():
        raise IOError("Data file not found")

    ball_pos_WC = pd.DataFrame(pd.read_csv(data_file_path), columns=["x", "y", "z"])
    ball_pos_blurred_WC = ball_pos_WC.iloc[
                          n_frames_to_avg::n_frames_to_avg, :
                          ].reset_index(drop=True)

    # Obtain projection matrix
    first_frame_path = blurred_frames_file_path / "frame00000.png"
    first_frame = cv.imread(str(first_frame_path), 1)
    first_frame_grey = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Manually initialise initial ball position
    # cv.namedWindow("img")

    # def clicked(event, x, y, flags, param):
    #     if event == cv.EVENT_LBUTTONDBLCLK:
    #         print(f"Clicked at ({x}, {y})")
    #         cv.circle(first_frame_grey, (x, y), 5, (255, 0, 0))
    #         image_coords.append([x, y])
    #
    # cv.setMouseCallback("img", clicked)

    # print("Double-Click: Top-Left -> Bottom-Left -> Top-Right -> Bottom-Right")

    # while True:
    #     # display the image and wait for a keypress
    #     cv.imshow("img", first_frame_grey)
    #     key = cv.waitKey(1) & 0xFF
    #
    #     if key == ord("c") or len(image_coords) >= 4:
    #         break

    # Initial ball position for sim id 5, to save time
    front_wall_image_coords: np.ndarray = np.array(
        [
            [330.0, 232.0],  # Front-Wall Top-Left
            [329.0, 555.0],  # Front-Wall Bottom Left
            [694.0, 233.0],  # Front-Wall Top Right
            [695.0, 556.0],  # Front-Wall Bottom Right
        ],
        dtype="float32",
    )
    front_wall_world_coords: np.ndarray = np.array(
        [
            [-HALF_COURT_WIDTH, WALL_HEIGHT, HALF_COURT_LENGTH],  # Front-Wall Top Left
            [-HALF_COURT_WIDTH, 0, HALF_COURT_LENGTH],  # Front-Wall Bottom Left
            [HALF_COURT_WIDTH, WALL_HEIGHT, HALF_COURT_LENGTH],  # Front-Wall Top Right
            [HALF_COURT_WIDTH, 0, HALF_COURT_LENGTH],  # Front-Wall Bottom Right
        ],
        dtype="float32",
    )

    # Approximate camera matrix
    # ToDo: Approximation might be bad, try RANSAC solver
    camera_matrix = cv.initCameraMatrix2D(
        [front_wall_world_coords], [front_wall_image_coords], first_frame_grey.shape
    )

    dist_coeffs = np.zeros((4, 1))  # Minimal distortion so can pass as null

    # Estimate camera extrinsics
    # ToDo: Approximation might be bad, try RANSAC PnP
    success, rotation_vector, translation_vector = cv.solvePnP(
        front_wall_world_coords,
        front_wall_image_coords,
        camera_matrix,
        dist_coeffs,
        flags=0,
    )

    print(
        f"world coords shape={front_wall_world_coords.shape}, dtype={front_wall_world_coords.dtype}):\n",
        front_wall_world_coords,
    )
    print(
        f"image coords shape={front_wall_image_coords.shape}, dtype={front_wall_image_coords.dtype}):\n",
        front_wall_image_coords,
    )
    print(
        f"camera matrix shape={camera_matrix.shape}, dtype={camera_matrix.dtype}):\n",
        camera_matrix,
    )

    for i in range(len(ball_pos_blurred_WC)):
        ball_pos_to_convert = np.array(
            [
                ball_pos_blurred_WC["x"][i],
                ball_pos_blurred_WC["y"][i],
                ball_pos_blurred_WC["z"][i],
            ]
        )
        print("-" * 55)
        print(
            f"Frame #{i}, world coords to convert:",
            ball_pos_to_convert,
        )

        ball_x_ic, ball_y_ic = wc_to_ic(
            ball_pos_to_convert[0],
            ball_pos_to_convert[1],
            ball_pos_to_convert[2],
            img_dims,
        )

        ball_pos_2D, _ = cv.projectPoints(
            ball_pos_to_convert,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs,
        )

        ball_pos_2D = ball_pos_2D[0, 0]

        print(f"Converted to image coords using POV mat  = {[ball_x_ic, ball_y_ic]}")
        print(f"Converted to image coords using solvePnP = {ball_pos_2D}")

        frame_num = f"{i}".zfill(5)
        frame_path = blurred_frames_file_path / f"frame{frame_num}.png"
        frame = cv.imread(str(frame_path))

        plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        plt.scatter(ball_pos_2D[0], ball_pos_2D[1],  label="solvePnP", marker="^", s=130, alpha=0.6)
        plt.scatter(ball_x_ic, ball_y_ic,  label="POV", marker="s", s=130, alpha=0.6)

        plt.legend()
        plt.tight_layout()
        plt.axis("off")
        plt.show()
