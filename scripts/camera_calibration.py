import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt

from ai_umpire.util import (
    wc_to_ic,
    HALF_COURT_LENGTH,
    HALF_COURT_WIDTH,
    WALL_HEIGHT,
    extract_frames_from_vid,
)

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 5
sim_length = 2.0
sim_step_sz = 0.005
n_rendered_frames = int(sim_length / sim_step_sz)
desired_fps = 50
n_frames_to_avg = int(n_rendered_frames / desired_fps)
img_dims = [1024, 768]


class FourCoordsStore:
    def __init__(self):
        self._coords: List = []

    def img_clicked(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(first_frame, (x, y), 7, (255, 0, 0))
            self._coords.append([x, y])

    def click_pos(self) -> np.ndarray:
        return np.array(self._coords, dtype="float32")


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

    # Load first frame of video, so we can get 4 points to approximate homography
    frames = extract_frames_from_vid(video_file_path)
    first_frame = frames[0].copy()
    first_frame_grey = np.mean(first_frame, -1)

    # Manually initialise initial ball position
    cv.namedWindow("First Frame")
    coords_store = FourCoordsStore()
    cv.setMouseCallback("First Frame", coords_store.img_clicked)

    print("Double-Click: Top-Left -> Bottom-Left -> Top-Right -> Bottom-Right")

    while True:
        # Display the image and wait for wither exit via escape key or 4 coordinates clicked
        cv.imshow("First Frame", first_frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("c") or coords_store.click_pos().shape[0] == 4:
            break
    cv.destroyAllWindows()

    front_wall_image_coords = coords_store.click_pos()
    # front_wall_image_coords: np.ndarray = np.array(
    #     [
    #         [330.0, 232.0],  # Front-Wall Top-Left
    #         [329.0, 555.0],  # Front-Wall Bottom Left
    #         [694.0, 233.0],  # Front-Wall Top Right
    #         [695.0, 556.0],  # Front-Wall Bottom Right
    #     ],
    #     dtype="float32",
    # )
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
    # ToDo: Approximation of cam matrix isn't perfect, resulting it slightly-off projection matrix
    camera_matrix = cv.initCameraMatrix2D(
        [front_wall_world_coords], [front_wall_image_coords], first_frame_grey.shape
    )

    dist_coeffs = np.zeros((4, 1))  # Minimal/no distortion so can pass as null

    # Estimate camera extrinsic
    # ToDo: Approximation might be bad, try RANSAC PnP
    _, rotation_vector, translation_vector = cv.solvePnP(
        front_wall_world_coords,
        front_wall_image_coords,
        camera_matrix,
        dist_coeffs,
        flags=0,
    )

    # Compute mean of reprojection error
    tot_error = 0
    total_points = 0
    for i in range(front_wall_world_coords.shape[0]):
        reprojected_points, _ = cv.projectPoints(front_wall_world_coords[i], rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        reprojected_points = reprojected_points.reshape(-1, 2)
        tot_error += np.sum(np.abs(front_wall_image_coords[i] - reprojected_points) ** 2)
        total_points += len(front_wall_world_coords[i])
    mean_error = np.sqrt(tot_error / total_points)
    print("Mean reprojection error: ", mean_error)

    # Calculate projection matrix, P = [cameraMatrix] . [rotationMatrix | translationVector]
    rot_mtx, _ = cv.Rodrigues(rotation_vector)  # Convert rotation from vector notation to matrix
    projection_matrix = camera_matrix @ np.c_[rot_mtx, translation_vector]

    for i in range(len(ball_pos_blurred_WC)):
        ball_pos_to_convert = np.array(
            [
                ball_pos_blurred_WC["x"][i],
                ball_pos_blurred_WC["y"][i],
                ball_pos_blurred_WC["z"][i],
            ]
        )
        print(f"Frame #{i}".ljust(50, "-"))
        print(
            f"World coords to convert:",
            ball_pos_to_convert,
        )

        ball_x_ic, ball_y_ic = wc_to_ic(
            ball_pos_to_convert[0],
            ball_pos_to_convert[1],
            ball_pos_to_convert[2],
            img_dims,
        )

        ball_pos_to_convert_homog = np.append(np.reshape(ball_pos_to_convert, (3, 1)), 1)
        pos_ic_proj_mtx_homog = projection_matrix @ ball_pos_to_convert_homog
        pos_ic_proj_mtx = [pos_ic_proj_mtx_homog[0] / pos_ic_proj_mtx_homog[-1], pos_ic_proj_mtx_homog[1] / pos_ic_proj_mtx_homog[-1]]

        print(f"IC conversion: POV Matrix        = {[ball_x_ic, ball_y_ic]}")
        print(f"IC conversion: Projection Matrix = {pos_ic_proj_mtx}")

        error = math.sqrt(
            ((ball_x_ic - pos_ic_proj_mtx[0]) ** 2) + ((ball_y_ic - pos_ic_proj_mtx[1]) ** 2)
        )
        print(f"Projection matrix error (Euclid distance to true) = {error:.2f}")

        plt.imshow(cv.cvtColor(frames[i], cv.COLOR_BGR2RGB))
        plt.scatter(
            pos_ic_proj_mtx[0],
            pos_ic_proj_mtx[1],
            label="Proj. Mtx.",
            marker="^",
            s=130,
            alpha=0.6,
        )
        plt.scatter(ball_x_ic, ball_y_ic, label="POV Mtx.", marker="s", s=130, alpha=0.6)

        plt.legend()
        plt.tight_layout()
        plt.axis("off")
        plt.show()
