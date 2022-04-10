import math
import random

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
    Detector,
    KalmanFilter,
    TrajectoryInterpreter,
)


ROOT_DIR_PATH: Path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
SIM_ID: int = 6
SIM_LENGTH: float = 2.0
SIM_STEP_SIZE: float = 0.005
N_RENDERED_IMAGES: int = int(SIM_LENGTH / SIM_STEP_SIZE)
DESIRED_FPS: int = 50
N_FRAMES_TO_AVERAGE: int = int(N_RENDERED_IMAGES / DESIRED_FPS)
IMG_DIMS: List[int] = [1024, 768]
START_X_POS: List[int] = [-2, -1, 0, 1]
START_Z_POS: List[int] = [-2, -1, 0, 1]


# ToDo: Incorporate this function into ball detector class
def filter_ball_detections(
    frame_detections: List[List],
    init_ball_pos: Tuple[float, float],
    *,
    min_ball_travel_dist: float = 5,
    max_ball_travel_dist: float = 130,
    min_det_area: float = 2.0,
    max_det_area: float = 65.0,
) -> List[List]:
    filtered_dets = []

    def get_frame_detections_com(frame_num: int) -> Tuple[float, float]:
        """com = Center of Mass"""
        prev_frame_accepted_dets_x = [x for x, _, _ in filtered_dets[frame_num]]
        prev_frame_accepted_dets_y = [y for _, y, _ in filtered_dets[frame_num]]
        com_x = sum(prev_frame_accepted_dets_x) / len(prev_frame_accepted_dets_x)
        com_y = sum(prev_frame_accepted_dets_y) / len(prev_frame_accepted_dets_y)

        return com_x, com_y

    for i in range(len(frame_detections)):
        print(f"Frame #{i}", "-" * 40)
        if len(filtered_dets) > 0:
            print(f"Accumulated filtered dets:")
            for d in filtered_dets:
                if len(d) > 1:
                    print(" " * 8, f"{len(d)} dets below:")
                    for d_ in d:
                        print(" " * 8, d_)
                else:
                    print(" " * 4, d)
        dets = frame_detections[i]

        # Filter detections base on their size, i.e. filter out the player detections and noise
        dets = [(x, y, z) for x, y, z in dets if min_det_area < z < max_det_area]

        print(f">> Num dets filtered by size = {len(frame_detections[i]) - len(dets)}")
        frame_num = f"{i}".zfill(5)
        frame_path = (
            ROOT_DIR_PATH
            / "blurred_frames"
            / f"sim_{SIM_ID}_blurred"
            / f"frame{frame_num}.png"
        )
        frame = cv.imread(str(frame_path), 1)

        if i > 0:
            velocity_constrained_dets = []
            for curr_frame_det in dets:
                for prev_frame_det in filtered_dets[i - 1]:

                    curr_x, curr_y = curr_frame_det[0], curr_frame_det[1]
                    prev_x, prev_y = prev_frame_det[0], prev_frame_det[1]

                    euclid_dist_between_prev_and_curr_dets = math.sqrt(
                        ((curr_x - prev_x) ** 2) + ((curr_y - prev_y) ** 2)
                    )
                    in_acceptable_range_of_motion = (
                        min_ball_travel_dist
                        < euclid_dist_between_prev_and_curr_dets
                        < max_ball_travel_dist
                    )

                    if (
                        in_acceptable_range_of_motion
                        and curr_frame_det not in filtered_dets[i - 1]
                    ):
                        cv.circle(frame, (curr_x, curr_y), 10, (0, 255, 0))
                        velocity_constrained_dets.append(curr_frame_det)
                        # print(f">> Added {curr_frame_det}")
                    else:
                        cv.circle(frame, (curr_x, curr_y), 10, (0, 0, 255))
            if len(velocity_constrained_dets) == 0:
                # If no detections satisfy the velocity constraint, add the detection closest to the center of mass of
                # the previous frame's acceptable detections
                print(
                    "[i] No dets added, added det closest to center of mass (com) of prev acceptable dets."
                )

                com_x, com_y = get_frame_detections_com(frame_num=i - 1)

                dets_dist_to_com = [
                    math.sqrt(((x - com_x) ** 2) + ((y - com_y) ** 2))
                    for x, y, _ in dets
                ]
                closest_det = dets[dets_dist_to_com.index(min(dets_dist_to_com))]
                filtered_dets.append([closest_det])

                # print(f">> Added {closest_det}")

                frame_ = frame.copy()
                cv.circle(frame_, (int(com_x), int(com_y)), 5, (0, 0, 255), -1)
                cv.circle(frame_, (closest_det[0], closest_det[1]), 5, (0, 255, 0))
                # cv.imshow(f"Frame #{i} - Center of mass & closest det to com", frame_)
                cv.waitKey(0)
            else:
                filtered_dets.append(list(set(velocity_constrained_dets)))
        else:
            # Find the closest detection to manually initialised initial ball position
            dists = [
                math.sqrt(((x - init_ball_pos[0]) ** 2) + ((y - init_ball_pos[1]) ** 2))
                for x, y, _ in dets
            ]
            closest_det = dets[dists.index(min(dists))]

            for det in dets:
                if det == closest_det:
                    cv.circle(frame, (det[0], det[1]), 10, (0, 255, 0))
                else:
                    cv.circle(frame, (det[0], det[1]), 10, (0, 0, 255))
            print(f">> First frame, adding closest det -> {closest_det}")

            filtered_dets.append([closest_det])

        cv.imshow(f"Frame #{i}", frame)
        cv.waitKey(0)

    return filtered_dets


class ClickStore:
    def __init__(self):
        self._click_pos: Tuple[int, int] = None

    def img_clicked(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(first_frame, (x, y), 7, (255, 0, 0))
            self._click_pos = (x, y)

    def click_pos(self) -> Tuple:
        return self._click_pos


if __name__ == "__main__":
    random.seed(1234)
    # If no POV-Ray data has been generated for this sim id, generate data and render images
    pov_data_file = ROOT_DIR_PATH / "generated_povray" / f"sim_{SIM_ID}_povray"
    if not pov_data_file.exists():
        # Sun simulation and export state and object texture data to POV-Ray format
        players_x: List[int] = sample(START_X_POS, 2)
        players_z: List[int] = sample(START_Z_POS, 2)
        sim = MatchSimulator(
            sim_id=SIM_ID,
            root=ROOT_DIR_PATH,
            step_sz=0.005,
            ball_origin=chrono.ChVectorD(
                choice([-3, 3, -2.5, 2.5]), uniform(0.2, 0.8), randint(-4, -2)
            ),
            ball_speed=chrono.ChVectorD(randint(-5, 5), randint(6, 15), randint(7, 25)),
            ball_acc=chrono.ChVectorD(-1, 2, 10),
            ball_rot_dt=chrono.ChQuaternionD(0, 0, 0.0436194, 0.9990482),
            p1_pos_x=players_x[0],
            p1_pos_z=players_z[0],
            p1_speed=chrono.ChVectorD(-1, 0, 1),
            p2_pos_x=players_x[1],
            p2_pos_z=players_z[1],
            p2_speed=chrono.ChVectorD(1, 0, -1),
        )
        sim.run_sim(SIM_LENGTH)

    # ---------------------------------------------- #
    # 1. Manually run POV script in POV-Ray application
    # 2. Copy generated images to root/sim_frames/sim_SIM_ID_frames file
    # ToDo: Automate this^
    # ---------------------------------------------- #

    # If no video has been generated for this sim id, generate video
    video_file = ROOT_DIR_PATH / "videos" / f"sim_{SIM_ID}.mp4"
    if not video_file.exists():
        # Generate video from images rendered by POV Ray of
        vid_gen = VideoGenerator(root_dir=ROOT_DIR_PATH)
        vid_gen.convert_frames_to_vid(SIM_ID, DESIRED_FPS)

    # Generate candidate ball detections in each frame
    detector = Detector(root_dir=ROOT_DIR_PATH)
    # Parameters obtained through random hyperparameter search
    all_detections = detector.get_ball_candidates_contour(
        sim_id=SIM_ID,
        morph_op="close",
        morph_op_iters=8,
        morph_op_SE_shape=(8, 8),
        blur_kernel_size=(81, 81),
        blur_sigma_x=4,
        binary_thresh_low=190,
        struc_el_shape=cv.MORPH_RECT,
        disable_progbar=False,
    )

    # Manually initialise the initial ball position, this will be used for detection filtering and Kalman initialisation
    first_frame_path = (
        ROOT_DIR_PATH / "blurred_frames" / f"sim_{SIM_ID}_blurred" / "frame00000.png"
    )
    first_frame = cv.imread(str(first_frame_path), 1)
    # first_frame_grey = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    click_store = ClickStore()
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

    # Filter detections based on initial ball position, size and maximum distance the ball could potentially travel
    # first_frame_ball_pos = click_store.click_pos()
    # filtered_ball_candidates = filter_ball_detections(
    #     all_detections, first_frame_ball_pos
    # )

    # Temporary solution until KF is used to filter candidate detections:
    # Arbitrarily select first detection in frame detections if more than one detection present.
    # This is in order to get one detection per frame to form the measurements for the KF.
    # measurements = np.array([detection[0] for detection in filtered_ball_candidates])
    # print(f"Measurement shape={measurements.shape}")

    # ---------------------------------------------- #
    # Candidate filtering is fragile so to demo tracking and trajectory-interpretation,
    # use true ball positions with added noise
    # ToDo: Either add another check for no candidates and use CoM again or reduce pre-processing stringency
    # ---------------------------------------------- #

    # Get true ball positions (output at time of simulation data exporting)
    BALL_POS_TRUE = ROOT_DIR_PATH / "ball_pos" / f"sim_{SIM_ID}.csv"
    ball_pos_WC = pd.DataFrame(pd.read_csv(BALL_POS_TRUE), columns=["x", "y", "z"])
    ball_pos_blurred_WC = ball_pos_WC.iloc[
        N_FRAMES_TO_AVERAGE::N_FRAMES_TO_AVERAGE, :
    ].reset_index(drop=True)

    x = ball_pos_blurred_WC["x"]
    y = ball_pos_blurred_WC["y"]
    z = ball_pos_blurred_WC["z"]

    rng = np.random.default_rng(111)

    measurements = np.c_[x, y, z]
    noisy_measurements = measurements + rng.normal(
        0, 0.02, size=(measurements.shape[0], 3)
    )

    # Track ball over time using Kalman filter and give probabilistic interpretation of trajectory
    n_variables = 3
    n_measurement_vals = measurements[0].shape[0]
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
        n_variables=n_variables,
        measurements=noisy_measurements,
        sigma_m=sigma_m,
        sigma_p=sigma_p,
        phi=phi,
        psi=psi,
        mu_m=mu_m,
        mu_p=mu_p,
    )

    ti = TrajectoryInterpreter(
        kalman_filter=kf, n_dim_samples=[10, 10, 10], n_std_devs_to_sample=1
    )
    ti.interpret_trajectory(visualise=True, save=False)
