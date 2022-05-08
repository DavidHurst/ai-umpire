import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ai_umpire.util import (
    extract_frames_from_vid,
    plot_bb,
    FIELD_BOUNDING_BOXES,
    HALF_COURT_WIDTH,
    HALF_COURT_LENGTH,
    SERVICE_LINE_HEIGHT,
    FRONT_WALL_OUT_LINE_HEIGHT,
)
from ai_umpire.util.util import (
    wc_to_ic,
    approximate_homography,
    get_sim_ball_pos,
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
    video_file_path = ROOT_DIR_PATH / "videos" / video_fname

    # Get filtered detections from ball detector
    # detector = BallDetector(root_dir=ROOT_DIR_PATH)
    # detections_IC = detector.get_filtered_ball_detections(
    #     vid_fname=video_fname,
    #     sim_id=SIM_ID,
    #     morph_op="close",
    #     morph_op_iters=1,
    #     morph_op_se_shape=(21, 21),
    #     blur_kernel_size=(33, 33),
    #     blur_sigma=1,
    #     binary_thresh=120,
    #     disable_progbar=False,
    #     struc_el_shape=cv.MORPH_RECT,
    #     min_ball_travel_dist=1,
    #     max_ball_travel_dist=70,
    #     min_det_area=1,
    #     max_det_area=30,
    # )
    # print(f"Detections: shape={detections_IC.shape}, noisy_gt: \n{detections_IC}")

    # Transform z detections_IC to be in the range [-0.5COURT_LENGTH, 0.5COURT_LENGTH]
    # old_z = detections_IC[:, -1]
    # tformed_z = transform_nums_to_range(
    #     old_z, [np.min(old_z), np.max(old_z)], [-HALF_COURT_LENGTH, HALF_COURT_LENGTH]
    # )
    # tformed_z = np.reshape(tformed_z, (tformed_z.shape[0], 1))
    # smoothed_tformed_z = savgol_filter(tformed_z.squeeze(), len(tformed_z) // 3 + 1, 3)

    ball_pos_true = get_sim_ball_pos(SIM_ID, ROOT_DIR_PATH, N_FRAMES_TO_AVERAGE)
    print(f"ball pos true shape = {ball_pos_true.shape}")
    print(ball_pos_true)

    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), smoothed_tformed_z, label="Smoothed")
    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), tformed_z, label="Not Smoothed")
    # plt.plot(np.linspace(0, 10, len(smoothed_tformed_z)), z[1:], label="GT")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    frames = extract_frames_from_vid(video_file_path, disable_progbar=True)
    first_frame = frames[0].copy()

    # Derive projection matrix using 4 known image coordinates and their corresponding world coordinates
    # cam_intrinsics, rot_mtx, t_vec = calibrate_camera(
    #     FRONT_WALL_WORLD_COORDS, front_wall_image_coords, first_frame_grey.shape
    # )

    h = approximate_homography(video_path=video_file_path)

    gt_reprojected = []
    # det_projected = []
    gt_reproj_mean_error = 0.0
    # det_proj_mean_error = 0.0
    for i in range(ball_pos_true.shape[0]):
        # Project true ball positions from world coordinates to image coordinates
        gt_pos_wc = ball_pos_true[i]
        gt_pos_ic = wc_to_ic(gt_pos_wc, list(first_frame.shape[:-1]))
        gt_xy_homog = np.reshape(np.append(gt_pos_ic, 1), (3, 1))

        # Homogenise detection
        # det_x = detections_IC[i][0]
        # det_y = detections_IC[i][1]
        # det_xy_homog = np.reshape(np.append([det_x, det_y], 1), (3, 1))

        scale = 5  # Scale constant for homography

        # OpenCV homography on true pos
        reproj_pt = h @ gt_xy_homog
        reproj_pt /= reproj_pt[-1]
        reproj_pt *= scale
        reproj_pt[-1] = gt_pos_wc[-1]

        # OpenCV homography on detections
        # det_proj_pt = h @ det_xy_homog
        # det_proj_pt /= det_proj_pt[-1]
        # det_proj_pt *= scale
        # det_proj_pt[-1] = gt_pos_wc[-1]

        # Calculate error in projection/reprojection
        reproj_error = math.sqrt(
            ((gt_pos_wc[0] - reproj_pt[0]) ** 2)
            + ((gt_pos_wc[1] - reproj_pt[1]) ** 2)
            + ((gt_pos_wc[2] - reproj_pt[2]) ** 2)
        )
        # det_proj_error = math.sqrt(
        #     ((gt_pos_wc[0] - det_proj_pt[0]) ** 2)
        #     + ((gt_pos_wc[1] - det_proj_pt[1]) ** 2)
        #     + ((gt_pos_wc[2] - det_proj_pt[2]) ** 2)
        # )

        gt_reprojected.append(reproj_pt)
        # det_projected.append(det_proj_pt)
        gt_reproj_mean_error += reproj_error / ball_pos_true.shape[0]
        # det_proj_mean_error += det_proj_error / ball_pos_true.shape[0]

    gt_reprojected = np.array(gt_reprojected).squeeze()
    # det_projected = np.array(det_projected).squeeze()

    print(f"GTs reprojection error = {gt_reproj_mean_error:.2f}m")
    # print(f"Dets. projection error = {det_proj_mean_error:.2f}m")

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
        ball_pos_true[:, 0],
        ball_pos_true[:, 1],
        ball_pos_true[:, 2],
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
