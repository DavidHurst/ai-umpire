import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ai_umpire import BallDetector
from ai_umpire.util import (
    extract_frames_from_vid,
    SinglePosStore,
    plot_bb,
    FIELD_BOUNDING_BOXES,
    HALF_COURT_WIDTH,
    HALF_COURT_LENGTH,
    WALL_HEIGHT,
    SERVICE_LINE_HEIGHT,
    FRONT_WALL_OUT_LINE_HEIGHT,
)
from ai_umpire.util.util import (
    calibrate_camera,
    FourCoordsStore,
    wc_to_ic,
    transform_nums_to_range,
)

ROOT_DIR_PATH: Path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
SIM_ID: int = 5
SIM_LENGTH: float = 2.0
SIM_STEP_SIZE: float = 0.005
N_RENDERED_IMAGES: int = int(SIM_LENGTH / SIM_STEP_SIZE)
DESIRED_FPS: int = 50
N_FRAMES_TO_AVERAGE: int = int(N_RENDERED_IMAGES / DESIRED_FPS)
FRONT_WALL_WORLD_COORDS: np.ndarray = np.array(
    [
        [
            -HALF_COURT_WIDTH,
            FRONT_WALL_OUT_LINE_HEIGHT,
            HALF_COURT_LENGTH,
        ],  # Front Wall Line Left
        [
            -HALF_COURT_WIDTH,
            SERVICE_LINE_HEIGHT,
            HALF_COURT_LENGTH,
        ],  # Service Line Left
        [
            HALF_COURT_WIDTH,
            FRONT_WALL_OUT_LINE_HEIGHT,
            HALF_COURT_LENGTH,
        ],  # Front Wall Line Right
        [
            HALF_COURT_WIDTH,
            SERVICE_LINE_HEIGHT,
            HALF_COURT_LENGTH,
        ],  # Service Line Right
    ],
    dtype="float32",
)

plt.rcParams["figure.figsize"] = (10, 10)

if __name__ == "__main__":
    video_fname = f"sim_{SIM_ID}.mp4"
    video_file = ROOT_DIR_PATH / "videos" / video_fname

    # Manually initialise the initial ball position, this will be used for detection filtering and Kalman initialisation
    # Load first frame of video, so we can get 4 points to approximate the inverse of the camera matrix
    frames = extract_frames_from_vid(video_file)
    first_frame = frames[0].copy()
    first_frame_grey = np.mean(first_frame, -1)
    click_store = SinglePosStore(first_frame)
    cv.namedWindow("First Frame")
    cv.setMouseCallback("First Frame", click_store.img_clicked)

    print(
        "Click the ball, since it is a streak, click on the end of the streak you believe to be the leading end."
    )

    while True:
        # Display the first frame and wait for a keypress
        cv.imshow("First Frame", first_frame)
        key = cv.waitKey(1) & 0xFF

        # Press esc to exit or once clicked, exit
        if key == 27 or click_store.click_pos() is not None:
            break
    cv.destroyAllWindows()

    print(f"Initial ball position set to {click_store.click_pos()}")

    # Get filtered detections from ball detector
    first_frame_ball_pos = click_store.click_pos()
    detector = BallDetector(root_dir=ROOT_DIR_PATH)
    measurements = detector.get_filtered_ball_detections(
        sim_id=SIM_ID,
        vid_fname=video_fname,
        morph_op="close",
        morph_op_iters=11,
        morph_op_se_shape=(2, 2),
        blur_kernel_size=(31, 31),
        blur_sigma=3,
        binary_thresh=130,
        struc_el_shape=cv.MORPH_RECT,
        disable_progbar=False,
        init_ball_pos=first_frame_ball_pos,
        min_ball_travel_dist=5,
        max_ball_travel_dist=130,
        min_det_area=1.0,
        max_det_area=40.0,
    )
    print(f"Measurements: shape={measurements.shape}, measurements: \n{measurements}")

    # Transform z measurements to be in the range [-0.5COURT_LENGTH, 0.5COURT_LENGTH]
    old_z = measurements[:, -1]
    tformed_z = transform_nums_to_range(
        old_z, [np.min(old_z), np.max(old_z)], [-HALF_COURT_LENGTH, HALF_COURT_LENGTH]
    )
    tformed_z = np.reshape(tformed_z, (tformed_z.shape[0], 1))

    print(f"Transformed z shape={tformed_z.shape} = {tformed_z}")

    # Camera calibration: obtain the coordinates of the 4 corners of the front wall which
    # will be used to derive the inverse of camera projection matrix for 2D->3D
    coords_store = FourCoordsStore(first_frame)
    cv.namedWindow("First Frame")
    cv.setMouseCallback("First Frame", coords_store.img_clicked)

    print(
        "Double-click at these locations on the front wall in the order presented:\n"
        "   1. Front Wall Line Left\n"
        "   2. Service Line Left\n"
        "   3. Front Wall Line Right\n"
        "   4. Service Line Left"
    )

    while True:
        # Display the image and wait for wither exit via escape key or 4 coordinates clicked
        cv.imshow("First Frame", first_frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("c") or coords_store.click_pos().shape[0] == 4:
            break
    cv.destroyAllWindows()

    front_wall_image_coords = coords_store.click_pos()

    print(
        f"""Image coordinates of the four corners of the Front-Wall set to:
        Front Wall Line Left  = {front_wall_image_coords[0]} 
        Service Line Left     = {front_wall_image_coords[1]} 
        Front Wall Line Right = {front_wall_image_coords[2]}
        Service Line Left     = {front_wall_image_coords[3]}"""
    )

    # Get true ball positions (output at time of simulation data exporting)
    BALL_POS_TRUE = ROOT_DIR_PATH / "ball_pos" / f"sim_{SIM_ID}.csv"
    ball_pos_WC = pd.DataFrame(pd.read_csv(BALL_POS_TRUE), columns=["x", "y", "z"])

    # Get position of ball in frames (different from true pos due to blurring by averaging frames)
    ball_pos_frames_WC = ball_pos_WC.iloc[
        N_FRAMES_TO_AVERAGE::N_FRAMES_TO_AVERAGE, :
    ].reset_index(drop=True)

    x = ball_pos_frames_WC["x"]
    y = ball_pos_frames_WC["y"]
    z = ball_pos_frames_WC["z"]

    # Derive projection matrix using 4 known image coordinates and their corresponding world coordinates
    cam_intrinsics, rot_mtx, t_vec = calibrate_camera(
        FRONT_WALL_WORLD_COORDS, front_wall_image_coords, first_frame_grey.shape
    )

    # Calculate inverse of intrinsics matrix, the simplest case, no skew and distortion = 1
    cam_intrinsics_inv = np.identity(3)
    cam_intrinsics_inv[-1, -1] = cam_intrinsics[0, 0]
    cam_intrinsics_inv[0, -1] = -cam_intrinsics[0, -1]
    cam_intrinsics_inv[1, -1] = -cam_intrinsics[1, -1]
    cam_intrinsics_inv = (1 / cam_intrinsics[0, 0]) * cam_intrinsics_inv

    # Calculate the inverse of the rotation matrix
    rot_mtx_inv = np.linalg.inv(rot_mtx)

    print(f"t_z = {t_vec[-1]}")

    # Estimate homography
    homography = (
        cam_intrinsics
        @ np.c_[
            np.reshape(rot_mtx[:, 0], (3, 1)), np.reshape(rot_mtx[:, 1], (3, 1)), t_vec
        ]
    )
    homography /= t_vec[-1]  # Normalise
    print(f"homography from parameters:\n{homography}")

    h, _ = cv.findHomography(
        front_wall_image_coords, FRONT_WALL_WORLD_COORDS, method=cv.RANSAC
    )
    print("opencv homography:\n", h)

    measurements_WC_h = []
    mean_hom_dist = 0.0
    mean_hom_p_dist = 0.0
    for i in range(len(x)):
        pos_wc = np.array([x[i], y[i], z[i]])
        pos_ic = wc_to_ic(pos_wc, list(first_frame.shape[:2]))
        xy_homog = np.reshape(np.append(pos_ic, 1), (3, 1))

        scale = 4

        # OpenCV homography
        homography_WC = h @ xy_homog
        homography_WC /= homography_WC[-1]
        homography_WC *= scale
        homography_WC[-1] = z[i]

        # Parameter homography
        homography_WC_p = homography @ xy_homog
        homography_WC_p /= homography_WC_p[-1]
        homography_WC_p[-1] = z[i]

        h_dist_2d = math.sqrt(
            ((x[i] - homography_WC[0]) ** 2)
            + ((y[i] - homography_WC[1]) ** 2)
            + ((z[i] - homography_WC[2]) ** 2)
        )
        h_p_dist_2d = math.sqrt(
            ((x[i] - homography_WC_p[0]) ** 2)
            + ((y[i] - homography_WC_p[1]) ** 2)
            + ((z[i] - homography_WC_p[2]) ** 2)
        )

        # print(f"Measurement #{i}".ljust(50, "-"))
        # print(f"Pos IC = {pos_ic}")
        # print(f"Pos WC                             = {pos_wc}")
        # print(f"Pos WC (OpenCV Hom.), d={h_dist_2d:.2f},   = {homography_WC.T.squeeze()}")
        # print(f"Pos WC (Param. Hom.), d={h_p_dist_2d:.2f}, = {homography_WC_p.T.squeeze()}")

        measurements_WC_h.append(homography_WC)
        mean_hom_dist += h_dist_2d / len(x)
        mean_hom_p_dist += h_p_dist_2d / len(x)

    measurements_WC_h = np.array(measurements_WC_h).squeeze()

    print(f"Mean hom. dist  = {mean_hom_dist:.2f}m")
    print(f"Mean hom. p dist = {mean_hom_p_dist:.2f}m")

    fig = plt.figure()
    ax = Axes3D(fig, elev=20, azim=-140, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.grid(False)
    ax.set_xlim3d(-4, 4)
    ax.set_zlim3d(0, 7)
    ax.set_ylim3d(-6, 6)
    ax.set_xlabel("$x$")
    ax.set_zlabel("$y$")
    ax.set_ylabel("$z$")

    for bb_name in FIELD_BOUNDING_BOXES.keys():
        plot_bb(
            bb_name=bb_name,
            ax=ax,
            bb_face_annotation="",
            show_vertices=False,
            show_annotation=False,
        )

    ax.plot3D(
        x,
        y,
        z,
        "-o",
        label="GT",
        alpha=0.5,
        c="g",
        zdir="y",
        markersize=10,
        zorder=4,
    )
    # ax.plot3D(
    #     measurements_WC[:, 0],
    #     measurements_WC[:, 1],
    #     measurements_WC[:, 2],
    #     "-o",
    #     alpha=0.5,
    #     label="P2L",
    #     c="b",
    #     zdir="y",
    #     markersize=10,
    #     zorder=5,
    # )
    ax.plot3D(
        measurements_WC_h[:, 0],
        measurements_WC_h[:, 1],
        measurements_WC_h[:, 2],
        "-o",
        alpha=0.5,
        label="Homography OpenCV",
        c="r",
        zdir="y",
        markersize=10,
        zorder=5,
    )
    plt.legend()
    plt.show()
