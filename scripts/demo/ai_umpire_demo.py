import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pychrono as chrono
import cv2 as cv
from pathlib import Path
from random import sample, choice, uniform, randint
from typing import List, Tuple

from ai_umpire import (
    MatchSimulator,
    VideoGenerator,
    BallDetector,
    KalmanFilter,
    TrajectoryInterpreter,
)
from ai_umpire.util import (
    extract_frames_from_vid,
    HALF_COURT_WIDTH,
    HALF_COURT_LENGTH,
    WALL_HEIGHT,
    calibrate_camera,
    SinglePosStore,
    FourCoordsStore,
)


ROOT_DIR_PATH: Path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
SIM_ID: int = 5
SIM_LENGTH: float = 2.0
SIM_STEP_SIZE: float = 0.005
N_RENDERED_IMAGES: int = int(SIM_LENGTH / SIM_STEP_SIZE)
DESIRED_FPS: int = 50
N_FRAMES_TO_AVERAGE: int = int(N_RENDERED_IMAGES / DESIRED_FPS)
START_X_POS: List[int] = [-2, -1, 0, 1]
START_Z_POS: List[int] = [-2, -1, 0, 1]
FRONT_WALL_WORLD_COORDS: np.ndarray = np.array(
    [
        [-HALF_COURT_WIDTH, WALL_HEIGHT, HALF_COURT_LENGTH],  # Front-Wall Top Left
        [-HALF_COURT_WIDTH, 0, HALF_COURT_LENGTH],  # Front-Wall Bottom Left
        [HALF_COURT_WIDTH, WALL_HEIGHT, HALF_COURT_LENGTH],  # Front-Wall Top Right
        [HALF_COURT_WIDTH, 0, HALF_COURT_LENGTH],  # Front-Wall Bottom Right
    ],
    dtype="float32",
)

if __name__ == "__main__":
    random.seed(1234)
    video_fname = f"sim_{SIM_ID}.mp4"
    # If no POV-Ray data has been generated for this sim id, generate data and render images
    pov_data_file = ROOT_DIR_PATH / "generated_povray" / f"sim_{SIM_ID}_povray"
    if not pov_data_file.exists():
        # Run simulation and export state and object texture data to POV-Ray format
        players_x: List[int] = sample(START_X_POS, 2)
        players_z: List[int] = sample(START_Z_POS, 2)
        sim = MatchSimulator(
            sim_id=SIM_ID,
            root=ROOT_DIR_PATH,
            sim_step_sz=0.005,
            ball_init_pos=chrono.ChVectorD(
                choice([-3, 3, -2.5, 2.5]), uniform(0.2, 0.8), randint(-4, -2)
            ),
            ball_vel=chrono.ChVectorD(randint(-5, 5), randint(6, 15), randint(7, 25)),
            ball_acc=chrono.ChVectorD(1, 2, 10),
            ball_rot_dt=chrono.ChQuaternionD(0, 0, 0.0436194, 0.9990482),
            p1_init_x=players_x[0],
            p1_init_z=players_z[0],
            p1_vel=chrono.ChVectorD(-1, 0, 1),
            p2_init_x=players_x[1],
            p2_init_z=players_z[1],
            p2_vel=chrono.ChVectorD(1, 0, -1),
        )
        sim.run_sim(SIM_LENGTH)

    # ---------------------------------------------- #
    # 1. Manually run POV script in POV-Ray application
    # 2. Copy generated images to root/sim_frames/sim_SIM_ID_frames file
    # ToDo: Automate this^
    # ---------------------------------------------- #

    # If no video has been generated for this sim id, generate video
    video_file = ROOT_DIR_PATH / "videos" / video_fname
    if not video_file.exists():
        # Generate video from images rendered by POV Ray of
        vid_gen = VideoGenerator(root_dir=ROOT_DIR_PATH)
        vid_gen.convert_frames_to_vid(SIM_ID, DESIRED_FPS)

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

    plt.imshow(cv.cvtColor(frames[0], cv.COLOR_BGR2RGB))
    plt.scatter(measurements[:, 0], measurements[:, 1], s=measurements[:, 2] * 2)
    plt.show()

    # Camera calibration: obtain the coordinates of the 4 corners of the front wall which
    # will be used to derive the inverse of camera projection matrix for 2D->3D
    coords_store = FourCoordsStore(first_frame)
    cv.namedWindow("First Frame")
    cv.setMouseCallback("First Frame", coords_store.img_clicked)

    print(
        "Double-Click on the corners of the Front-Wall in this order: Top-Left -> Bottom-Left -> "
        "Top-Right -> Bottom-Right"
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
    Top-Left     = {front_wall_image_coords[0]} 
    Bottom-Left  = {front_wall_image_coords[1]} 
    Top-Right    = {front_wall_image_coords[2]}
    Bottom-Right = {front_wall_image_coords[3]}"""
    )

    # Derive projection matrix using 4 known image coordinates and their corresponding world coordinates
    cam_intrinsics, rot_mtx, t_vec = calibrate_camera(
        FRONT_WALL_WORLD_COORDS, front_wall_image_coords, first_frame_grey.shape
    )

    # Calculate inverse of intrinsics matrix, simplest case, no skew and distortion = 1
    cam_intrinsics_inv = np.identity(3)
    cam_intrinsics_inv[-1, -1] = cam_intrinsics[0, 0]
    cam_intrinsics_inv[0, -1] = -cam_intrinsics[0, -1]
    cam_intrinsics_inv[1, -1] = -cam_intrinsics[1, -1]
    cam_intrinsics_inv = (1 / cam_intrinsics[0, 0]) * cam_intrinsics_inv

    # Calculate the inverse of the rotation matrix
    rot_mtx_inv = np.linalg.inv(rot_mtx)

    # # Get true ball positions (output at time of simulation data exporting)
    BALL_POS_TRUE = ROOT_DIR_PATH / "ball_pos" / f"sim_{SIM_ID}.csv"
    ball_pos_WC = pd.DataFrame(pd.read_csv(BALL_POS_TRUE), columns=["x", "y", "z"])

    # Get position of ball in frames (different from true pos due to blurring by averaging frames)
    ball_pos_frames_WC = ball_pos_WC.iloc[
        N_FRAMES_TO_AVERAGE::N_FRAMES_TO_AVERAGE, :
    ].reset_index(drop=True)

    x = ball_pos_frames_WC["x"]
    y = ball_pos_frames_WC["y"]
    z = ball_pos_frames_WC["z"]

    # Convert detections_IC from image coordinates to world coordinates for KF
    cam_x, cam_y = cam_intrinsics[0, -1], cam_intrinsics[1, -1]
    measurements_WC = []
    for i, m in enumerate(measurements):
        print(f"Measurement #{i}".ljust(40, "-"))
        # Calculate the point to line transformation, given by:
        # w . [rot_mtx.T] [cam_intrinsics^-1] [x, y, 1].T - [rot_mtx^-1] [t_vec]
        w = 1  # Typical to use 1 or the distance from the camera to the point, our z surrogate in this case
        xy_homog = np.reshape(np.append(m[:-1], 1), (3, 1))
        m_WC = w * rot_mtx.T @ cam_intrinsics_inv @ xy_homog - rot_mtx_inv @ t_vec

        # ToDo: Z needs to be normalised to court depth range

        measurements_WC.append([m_WC[0].item(), m_WC[1].item(), m[2]])
        # print(f"In IC: shape={m.shape} \n{m}")
        # print(f"In WC: shape={m_WC.shape} \n{m_WC}")
        # print(f"True WC: \n{[x[i], y[i], z[i]]}")

    measurements_WC = np.array(measurements_WC).squeeze()
    print(f"Measurements WC, shape={measurements_WC.shape} \n{measurements_WC}")

    rng = np.random.default_rng(111)

    # Generate noisy detections_IC to demonstrate KF and trajectory interpretation
    # detections_IC = np.c_[x, y, z]
    # noisy_measurements = detections_IC + rng.normal(
    #     0, 0.02, size=(detections_IC.shape[0], 3)
    # )

    # Convert manually selected ball position to world coordinates so we can initialise KF with it
    init_ball_pos = click_store.click_pos()
    init_ball_pos_homog = np.reshape(np.append(init_ball_pos, 1), (3, 1))
    init_mu_WC = (
        1 * rot_mtx.T @ cam_intrinsics_inv @ init_ball_pos_homog - rot_mtx_inv @ t_vec
    )
    init_mu_WC[-1] = 0
    print(f"Init mu: {init_mu_WC}")

    # Kalman filter will track ball over time and smooth noise in the detections_IC
    n_variables = 3
    n_measurement_vals = measurements_WC[0].shape[0]
    mu_p = np.zeros((n_variables, 1))
    mu_m = np.zeros((n_measurement_vals, 1))
    psi = np.identity(n_variables)
    phi = np.eye(
        n_measurement_vals, n_variables
    )  # Temporary, should relate data to state e.g. through a projection matrix
    # sigma_p = np.ones((n_variables, n_variables))
    # sigma_m = np.ones((n_variables, n_variables))
    sigma_p = np.identity(n_variables) * 3
    sigma_m = np.identity(n_variables) * 4

    kf = KalmanFilter(
        init_mu=init_mu_WC,
        n_variables=n_variables,
        measurements=measurements_WC,
        sigma_m=sigma_m,
        sigma_p=sigma_p,
        phi=phi,
        psi=psi,
        mu_m=mu_m,
        mu_p=mu_p,
    )

    # Using Kalman filter, give probabilistic interpretation of trajectory by sampling points around predicted true ball
    # position and calculating whether they went out of court or syated in
    ti = TrajectoryInterpreter(
        kalman_filter=kf, n_dim_samples=[10, 10, 10], n_std_devs_to_sample=1
    )
    ti.interpret_trajectory(visualise=True, save=False)
