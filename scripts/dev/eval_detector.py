import math
from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ai_umpire import BallDetector
from ai_umpire.util import wc_to_ic, extract_frames_from_vid, SinglePosStore

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

    # Obtain the ball's initial position in the video from the user
    frames = extract_frames_from_vid(ROOT_DIR_PATH / "videos" / video_fname)
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
        init_ball_pos=click_store.click_pos(),
    )

    data_file_path = ROOT_DIR_PATH / "ball_pos" / f"sim_{SIM_ID}.csv"
    ball_pos_WC = pd.DataFrame(pd.read_csv(data_file_path), columns=["x", "y", "z"])
    ball_pos_blurred_WC = ball_pos_WC.iloc[
        N_FRAMES_TO_AVG::N_FRAMES_TO_AVG, :
    ].reset_index(drop=True)

    # Obtain true ball positions in image space
    euclid_dists = []
    true_ball_pos_IC = []
    for i in range(len(filtered_dets)):
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

        true_ball_pos_IC.append((ball_x_ic, ball_y_ic, ball_pos_blurred_WC["z"][i]))

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
    ball_true_z = ball_pos_blurred_WC["z"].to_numpy()[:-1]
    z_estimate_corr = (
        pd.DataFrame(
            {
                "z": ball_true_z,
                "Z Estimate": np.sqrt(np.array([z for _, _, z in filtered_dets])),
            }
        )
        .corr()
        .iloc[0]["Z Estimate"]
    )
    print(
        f"Pearson's correlation coefficient between true z and estimated for sim id {SIM_ID} = {z_estimate_corr}"
    )
