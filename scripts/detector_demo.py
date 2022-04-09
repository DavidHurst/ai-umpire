from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv

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

    # Detector params
    opening_iters = 3
    morph_op_SE_shape = (3, 3)
    blur_kernel_size = (61, 61)
    blur_strength = 2
    binarize_thresh_low = 230

    # Generate ball candidates per frame in video
    # loc = Localiser(root_dir_path)
    # frame_detections = loc.get_ball_candidates_contour(
    #     sim_id=sim_id,
    #     morph_op="open",
    #     morph_op_iters=opening_iters,
    #     morph_op_SE_shape=morph_op_SE_shape,
    #     blur_kernel_size=blur_kernel_size,
    #     blur_sigma_x=blur_strength,
    #     binary_thresh_low=binarize_thresh_low,
    #     disable_progbar=False,
    # )

    # i = 0
    # ball_x_ic, ball_y_ic = wc_to_ic(
    #     ball_pos_blurred_WC["x"][i],
    #     ball_pos_blurred_WC["y"][i],
    #     ball_pos_blurred_WC["z"][i],
    #     img_dims,
    # )

    # det = frame_detections[0][0]
    # print(f"Pos IC = ({ball_x_ic}, {ball_y_ic})")
    # ball_pos_IC = np.reshape(
    #     np.array([ball_x_ic, ball_y_ic, 2, 1]), (1, 4)
    # )
    # print(f"pos_ic @ cam mat", ball_pos_IC @ CAM_TFORM_MAT_INV)

    # Obtain projection matrix
    first_frame_path = blurred_frames_file_path / "frame00000.png"
    first_frame = cv.imread(str(first_frame_path), 1)
    first_frame_grey = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    world_coords: np.ndarray = np.array(
        [
            [-HALF_COURT_WIDTH, WALL_HEIGHT, HALF_COURT_LENGTH],  # Top Left
            [-HALF_COURT_WIDTH, 0, HALF_COURT_LENGTH],  # Bottom Left
            [HALF_COURT_WIDTH, WALL_HEIGHT, HALF_COURT_LENGTH],  # Top Right
            [HALF_COURT_WIDTH, 0, HALF_COURT_LENGTH],  # Bottom Right
        ]
    )

    # image_coords = []

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

    image_coords = np.array(
        [[330.0, 232.0], [329.0, 555.0], [694.0, 233.0], [695.0, 556.0]]
    )

    image_coords = image_coords.astype("float32")
    world_coords = world_coords.astype("float32")
    camera_matrix = cv.initCameraMatrix2D(
        [world_coords], [image_coords], first_frame_grey.shape
    )

    print(
        f"world coords shape={world_coords.shape}, dtype={world_coords.dtype}):\n",
        world_coords,
    )
    print(
        f"image coords shape={image_coords.shape}, dtype={image_coords.dtype}):\n",
        image_coords,
    )
    print(
        f"camera matrix shape={camera_matrix.shape}, dtype={camera_matrix.dtype}):\n",
        camera_matrix,
    )

    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera([world_coords], [image_coords], first_frame_grey.shape[::-1],
    #                                                   camera_matrix, None)
    # print("Camera matrix: \n")
    # print(mtx)
    # print("dist: \n")
    # print(dist)
    # print("rvecs: \n")
    # print(rvecs)
    # print("tvecs: \n")
    # print(tvecs)

    # Find homography between image plane and world coordinates front-wall plane (defined by user clicks)
    homography, _ = cv.findHomography(world_coords, image_coords, method=cv.RANSAC)
    homography_inv = np.linalg.inv(homography)

    pov_cam_extrinsics = CAM_EXTRINSICS_HOMOG

    print(
        f"homography shape={homography.shape}, dtype={homography.dtype}:\n{homography}"
    )
    print(
        f"homography_inv shape={homography_inv.shape}, dtype={homography_inv.dtype}:\n{homography_inv}"
    )
    print(
        f"pov_cam_extrinsics shape={pov_cam_extrinsics.shape}, dtype={pov_cam_extrinsics.dtype}:\n{pov_cam_extrinsics}"
    )

    print("-" * 55)
    print(
        "World coords to convert:",
        (
            ball_pos_blurred_WC["x"][0],
            ball_pos_blurred_WC["y"][0],
            ball_pos_blurred_WC["z"][0],
        ),
    )

    ball_x_ic, ball_y_ic = wc_to_ic(
        ball_pos_blurred_WC["x"][0],
        ball_pos_blurred_WC["y"][0],
        ball_pos_blurred_WC["z"][0],
        img_dims,
    )

    print(f"Converted to IC (Using POV mat)= ({ball_x_ic}, {ball_y_ic})")

    projective_transform = camera_matrix @ homography
    print(
        f"projective_transform shape={projective_transform.shape}, dtype={projective_transform.dtype}:\n{projective_transform}"
    )

    pt_3d_2d = (
        np.array(
            [
                ball_pos_blurred_WC["x"][0],
                ball_pos_blurred_WC["y"][0],
                ball_pos_blurred_WC["z"][0],
            ]
        )
        @ projective_transform
    )
    pos_ic_coefs: np.ndarray = np.array(
        [
            0.5 + (pt_3d_2d[0] / pt_3d_2d[-1]),
            0.5 - (pt_3d_2d[1] / pt_3d_2d[-1]),
        ]
    )
    pos_ic: np.ndarray = np.reshape(img_dims, (2, 1)) * pos_ic_coefs
    print(f"Converted to IC (Using homography)= ({pos_ic})")
