import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

from ai_umpire import BallDetector
from ai_umpire.util import (
    extract_frames_from_vid,
    SinglePosStore,
    plot_bb,
    FIELD_BOUNDING_BOXES,
    HALF_COURT_WIDTH,
    HALF_COURT_LENGTH,
    SERVICE_LINE_HEIGHT,
    FRONT_WALL_OUT_LINE_HEIGHT,
)
from ai_umpire.util.util import (
    calibrate_camera,
    FourCoordsStore,
    wc_to_ic,
    transform_nums_to_range,
)

ROOT_DIR_PATH: Path = Path() / "data"
SIM_ID: int = 0
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

plt.rcParams["figure.figsize"] = (5.5, 4.5)

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
    detections_IC = detector.get_filtered_ball_detections(
        vid_fname=video_fname,
        sim_id=SIM_ID,
        morph_op="close",
        morph_op_iters=1,
        morph_op_se_shape=(21, 21),
        blur_kernel_size=(33, 33),
        blur_sigma=1,
        binary_thresh=120,
        disable_progbar=False,
        struc_el_shape=cv.MORPH_RECT,
        min_ball_travel_dist=1,
        max_ball_travel_dist=70,
        min_det_area=1,
        max_det_area=30,
        init_ball_pos=first_frame_ball_pos,
    )
    # print(f"Measurements: shape={detections_IC.shape}, measurements: \n{detections_IC}")

    # Transform z detections_IC to be in the range [-0.5COURT_LENGTH, 0.5COURT_LENGTH]
    old_z = detections_IC[:, -1]
    tformed_z = transform_nums_to_range(
        old_z, [np.min(old_z), np.max(old_z)], [-HALF_COURT_LENGTH, HALF_COURT_LENGTH]
    )
    tformed_z = np.reshape(tformed_z, (tformed_z.shape[0], 1))
    smoothed_tformed_z = savgol_filter(tformed_z.squeeze(), len(tformed_z) // 3 + 1, 3)

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
        # Display the image and wait for either exit via escape key or 4 coordinates clicked
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

    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), smoothed_tformed_z, label="Smoothed")
    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), tformed_z, label="Not Smoothed")
    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), z[1:], label="GT")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Derive projection matrix using 4 known image coordinates and their corresponding world coordinates
    cam_intrinsics, rot_mtx, t_vec = calibrate_camera(
        FRONT_WALL_WORLD_COORDS, front_wall_image_coords, first_frame_grey.shape
    )

    h, _ = cv.findHomography(
        front_wall_image_coords, FRONT_WALL_WORLD_COORDS, method=cv.RANSAC
    )

    gt_reprojected = []
    det_projected = []
    gt_reproj_mean_error = 0.0
    det_proj_mean_error = 0.0
    for i in range(len(detections_IC)):
        # Project true ball positions from world coordinates to image coordinates
        pos_wc = np.array([x[i], y[i], z[i]])
        pos_ic = wc_to_ic(pos_wc, list(first_frame.shape[:2]))
        gt_xy_homog = np.reshape(np.append(pos_ic, 1), (3, 1))

        # Homogenise detection
        det_x = detections_IC[i][0]
        det_y = detections_IC[i][1]
        det_xy_homog = np.reshape(np.append([det_x, det_y], 1), (3, 1))

        scale = 1  # Scale constant for homography

        # OpenCV homography on true pos
        reproj_pt = h @ gt_xy_homog
        reproj_pt /= reproj_pt[-1]
        reproj_pt *= scale
        reproj_pt[-1] = z[i]

        # OpenCV homography on detections
        det_proj_pt = h @ det_xy_homog
        det_proj_pt /= det_proj_pt[-1]
        det_proj_pt *= scale
        det_proj_pt[-1] = z[i]

        # Calculate error in projection/reprojection
        reproj_error = math.sqrt(
            ((x[i] - reproj_pt[0]) ** 2)
            + ((y[i] - reproj_pt[1]) ** 2)
            + ((z[i] - reproj_pt[2]) ** 2)
        )
        det_proj_error = math.sqrt(
            ((x[i] - det_proj_pt[0]) ** 2)
            + ((y[i] - det_proj_pt[1]) ** 2)
            + ((z[i] - det_proj_pt[2]) ** 2)
        )

        gt_reprojected.append(reproj_pt)
        det_projected.append(det_proj_pt)
        gt_reproj_mean_error += reproj_error / len(x)
        det_proj_mean_error += det_proj_error / len(x)

    gt_reprojected = np.array(gt_reprojected).squeeze()
    det_projected = np.array(det_projected).squeeze()

    print(f"GTs reprojection error = {gt_reproj_mean_error:.2f}m")
    print(f"Dets. projection error = {det_proj_mean_error:.2f}m")

    fig = plt.figure()
    ax = Axes3D(fig, elev=15, azim=-140, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.grid(False)
    ax.set_xlim3d(-4, 4)
    ax.set_zlim3d(0, 7)
    ax.set_ylim3d(-6, 6)
    ax.set_xlabel("$x$")
    ax.set_zlabel("$y$")
    ax.set_ylabel("$z$")

    for bb_name in FIELD_BOUNDING_BOXES.keys():
        if not bb_name.startswith(("left", "back")):
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
        zorder=4,
    )
    # ax.plot3D(
    #     det_projected[:, 0],
    #     det_projected[:, 1],
    #     det_projected[:, 2],
    #     "-x",
    #     alpha=0.5,
    #     label="GT Reproj.",
    #     c="b",
    #     zdir="y",
    #     markersize=10,
    #     zorder=5,
    # )
    ax.plot3D(
        gt_reprojected[:, 0],
        gt_reprojected[:, 1],
        gt_reprojected[:, 2],
        "-^",
        alpha=0.5,
        label="GT Reproj.",
        c="r",
        zdir="y",
        zorder=5,
    )
    plt.legend()
    plt.savefig("eval_cam_calib.png")
    plt.show()
