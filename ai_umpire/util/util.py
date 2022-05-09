import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import pi
from numpy.linalg import inv, det
from scipy.spatial import Delaunay
from tqdm import tqdm

__all__ = [
    "extract_frames_from_vid",
    "difference_frames",
    "blur_frames",
    "binarize_frames",
    "apply_morph_op",
    "wc_to_ic",
    "multivariate_norm_pdf",
    "gen_grid_of_points",
    "CAM_EXTRINSICS_HOMOG",
    "calibrate_camera",
    "SinglePosStore",
    "FourCoordsStore",
    "plot_bb",
    "point_bb_collided",
    "transform_nums_to_range",
    "approximate_homography",
    "get_sim_ball_pos",
    "get_init_ball_pos",
]

from ai_umpire.util import (
    FIELD_BOUNDING_BOXES,
    HALF_COURT_LENGTH,
    BB_DEPTH,
    HALF_COURT_WIDTH,
    FRONT_WALL_OUT_LINE_HEIGHT,
    SERVICE_LINE_HEIGHT,
)

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

# Extracted from POV-Ray, will only work with wc_to_ic function for a image resolution of [852, 480]
CAM_EXTRINSICS_HOMOG: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5625, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 3.0, -16.0, 1.0],
    ]
)

CAM_EXTRINSICS_HOMOG_INV: np.ndarray = np.linalg.inv(CAM_EXTRINSICS_HOMOG)


def gen_grid_of_points(
    center: np.ndarray,
    n_dim_samples: list,
    sampling_area_size: list,
) -> np.ndarray:
    if center.shape[0] != 3:
        raise ValueError("Expecting 3D point for center.")
    if len(sampling_area_size) != center.shape[0]:
        raise ValueError("You must provide a sample area size for each dimension.")
    if len(n_dim_samples) != center.shape[0]:
        raise ValueError("You must provide a number of samples each dimension.")
    x = np.linspace(
        center[0] - (sampling_area_size[0] / 2),
        center[0] + (sampling_area_size[0] / 2),
        n_dim_samples[0],
    )
    y = np.linspace(
        center[1] - (sampling_area_size[1] / 2),
        center[1] + (sampling_area_size[1] / 2),
        n_dim_samples[1],
    )
    z = np.linspace(
        center[2] - (sampling_area_size[2] / 2),
        center[2] + (sampling_area_size[2] / 2),
        n_dim_samples[2],
    )

    sample_x = []
    sample_y = []
    sample_z = []
    for p1 in x:
        for p2 in y:
            for p3 in z:
                sample_x.append(p1)
                sample_y.append(p2)
                sample_z.append(p3)

    return np.c_[np.array(sample_x), np.array(sample_y), np.array(sample_z)]


def multivariate_norm_pdf(x: np.array, mu: np.array, sigma: np.array) -> float:
    if x.shape[0] != mu.shape[0]:
        raise ValueError("Mean and sample dimensions incompatible.")
    if sigma.shape != (x.shape[0], x.shape[0]):
        raise ValueError("Non-square covariance matrix.")
    if det(sigma) == 0:
        raise ValueError("The covariance matrix can't be singular.")

    numerator = np.exp(-0.5 * (x - mu).T @ inv(sigma) @ (x - mu))
    denominator = np.sqrt((2 * pi) ** x.shape[0] * det(sigma))
    return (numerator / denominator).item()


def wc_to_ic(
    pos_wc: np.ndarray, img_dims: List[int], *, m: np.ndarray = CAM_EXTRINSICS_HOMOG_INV
) -> Tuple[int, int]:
    """
    Project the given world coordinate point into image coordinates
    Only works for synthetic videos in 720p as the camera matrix has been extracted specifically for that camera.
    """
    # Check the provided dimensions are in the format that OpenCV loaded images are in
    if img_dims[0] > img_dims[1]:
        warnings.warn(
            "Warning, expecting image dimensions given as height x width, possible opposite provided."
        )
    img_dims.reverse()
    pos_x_wc, pos_y_wc, pos_z_wc = pos_wc[0], pos_wc[1], pos_wc[2]

    # Homogensise
    homog_ball_wc: np.ndarray = np.reshape(
        np.array([pos_x_wc, pos_y_wc, pos_z_wc, 1]), (1, 4)
    )
    homog_tformed_ball_wc: np.ndarray = homog_ball_wc @ m
    tformed_ball_wc: np.ndarray = homog_tformed_ball_wc[
        :, :-1
    ]  # Convert homogenous to Cartesian
    pos_ic_coefs: np.ndarray = np.array(
        [
            0.5 + (tformed_ball_wc[:, 0] / tformed_ball_wc[:, -1]),
            0.5 - (tformed_ball_wc[:, 1] / tformed_ball_wc[:, -1]),
        ]
    )

    pos_ic: np.ndarray = np.reshape(img_dims, (2, 1)) * pos_ic_coefs

    return pos_ic[0].item(), pos_ic[1].item()


def binarize_frames(
    frames: np.ndarray,
    thresh_low: int,
    thresh_high: int = 255,
    disable_progbar: bool = False,
) -> np.ndarray:
    """
    Performs binary thresholding on the given images
    :param frames: Images to binarize
    :param thresh_low: Lower-bound of intensity threshold
    :param thresh_high: Upper-bound of intensity threshold
    :param disable_progbar: Whether to show the progress bar
    :return: Binarizes frames
    """
    binary_frames: List[np.ndarray] = []
    for i in tqdm(
        range(frames.shape[0]), desc="Binarizing frames", disable=disable_progbar
    ):
        frame: np.ndarray = frames[i]

        # Normalise frame and convert to greyscale
        normalised_frame: np.ndarray = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX)
        normalised_greyscale_frame: np.ndarray = np.mean(
            normalised_frame, axis=2
        ).astype(np.uint8)

        # Binarize frame with Otsu's method
        _, binary_frame = cv.threshold(
            normalised_greyscale_frame,
            thresh_low,
            thresh_high,
            cv.THRESH_BINARY,
        )
        binary_frames.append(binary_frame)

    return np.array(binary_frames)


def blur_frames(
    frames: np.ndarray,
    kernel_sz: Tuple[int, int],
    sigma_x: int,
    disable_progbar: bool = False,
) -> np.ndarray:
    """
    Apply Gaussian blur to the given frames
    :param frames: Images to blur
    :param kernel_sz: Shape/size of the Gaussiain kernel to use
    :param sigma_x: Effective strength of the blurring
    :param disable_progbar: Whether to show the progress bar
    :return: Blurred frames
    """
    blurred_frames: List[np.ndarray] = []
    for i in tqdm(
        range(frames.shape[0]), desc="Blurring frames", disable=disable_progbar
    ):
        blurred_frames.append(cv.GaussianBlur(frames[i], kernel_sz, sigma_x))

    return np.array(blurred_frames)


def apply_morph_op(
    frames: np.ndarray,
    morph_op: str,
    n_iter: int,
    kernel_shape: Tuple[int, int],
    struc_el: np.ndarray = cv.MORPH_RECT,
    disable_progbar: bool = False,
) -> np.ndarray:
    """
    Applies the specified morphological operator to the given frames
    :param frames: Images to apply morphological opeator to
    :param morph_op: Morphological operation to apply
    :param n_iter: Number of times to apply the operation
    :param kernel_shape: Shape of the structing element
    :param struc_el: Structuring element to use
    :param disable_progbar: Whether to show the progress bar
    :return: Frames with the morphological operator applied to them
    """
    morph_ops: Dict[str, np.ndarray] = {
        "erode": cv.MORPH_ERODE,
        "open": cv.MORPH_OPEN,
        "dilate": cv.MORPH_DILATE,
        "close": cv.MORPH_CLOSE,
    }
    if morph_op not in morph_ops.keys():
        e: ValueError = ValueError(f"Supported morphological operators are {morph_ops}")
        logging.exception(e)
        raise e

    morph_op_frames: List[np.ndarray] = []
    for i in tqdm(
        range(frames.shape[0]),
        desc=f"Applying morph. op. ({morph_op})",
        disable=disable_progbar,
    ):
        morph_op_frame = cv.morphologyEx(
            src=frames[i],
            op=morph_ops[morph_op],
            kernel=cv.getStructuringElement(struc_el, kernel_shape),
            iterations=n_iter,
        )
        morph_op_frames.append(morph_op_frame)

    return np.array(morph_op_frames)


def difference_frames(frames: np.ndarray, disable_progbar: bool = False) -> np.ndarray:
    """
    Difference the provided frames using a window of three frames, subtracting the first in the window from the second
    and the third from the second, combining the two subtraction with a binary AND operation
    :param frames: Images to difference
    :param disable_progbar: Whether to show the progress bar
    :return: Differenced frames, 2 fewer than provided due to windowing process
    """
    foreground_segmented_frames: List[np.ndarray] = []

    for i in tqdm(
        range(1, frames.shape[0] - 1),
        desc="Differencing frames",
        disable=disable_progbar,
    ):
        preceding: np.ndarray = frames[i - 1]
        current: np.ndarray = frames[i]
        succeeding: np.ndarray = frames[i + 1]

        diff = cv.bitwise_and(current - preceding, succeeding - current)

        foreground_segmented_frames.append(diff)

    return np.array(foreground_segmented_frames)


def extract_frames_from_vid(
    vid_path: Path, disable_progbar: bool = False
) -> np.ndarray:
    """
    Extract the frames from the provided video and return them as a numpy array
    :param vid_path: The video to extract frames from
    :param disable_progbar: Whether to show the progress bar
    :return: Frames of the video
    """
    logging.info("Extracting frames from video.")
    v_cap: cv.VideoCapture = cv.VideoCapture(str(vid_path), cv.CAP_FFMPEG)
    frames: List[np.ndarray] = []

    pbar: tqdm = tqdm(desc="Extracting frames", disable=disable_progbar)
    while v_cap.isOpened():
        read_success, frame = v_cap.read()

        if read_success:
            frames.append(frame)
            pbar.update(1)
        else:
            break

    v_cap.release()
    logging.info("Frames extracted successfully.")
    return np.array(frames)


def calibrate_camera(
    world_coords: np.ndarray, image_coords: np.ndarray, image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the components of the camera matrix.

    :param image_size: The dimensions of the image used to approximate the camera parameters
    :param image_coords: 4 known image coordinates (2D) that correspond the 4 provided world coordinates
    :param world_coords: 4 known world coordinates (3D) that correspond the 4 provided image coordinates
    :return: Returns the rotation matrix, camera intrinsics matrix and translation vector as a three-tuple
    """
    if world_coords.shape[0] != 4 or image_coords.shape[0] != 4:
        raise ValueError(
            "You must provide 4 points in image coordinates and world coordinates each."
        )
    if image_coords.shape[1] != 2:
        raise ValueError("Image coordinates must be 2D, matrix shape should be (4,2).")
    if world_coords.shape[1] != 3:
        raise ValueError("World coordinates must be 3D, matrix shape should be (4,3).")

    # Approximate camera matrix
    camera_intrinsics_mtx: np.ndarray = cv.initCameraMatrix2D(
        [world_coords], [image_coords], image_size
    )

    dist_coeffs = np.zeros((4, 1))  # Minimal/no distortion so can pass as null

    # Estimate camera extrinsic
    _, rotation_vector, t_vec = cv.solvePnP(
        world_coords,
        image_coords,
        camera_intrinsics_mtx,
        dist_coeffs,
        flags=0,
    )

    # Convert rotation from vector notation to matrix using Rodrigues' method
    rot_mtx, _ = cv.Rodrigues(rotation_vector)

    return camera_intrinsics_mtx, rot_mtx, t_vec


def transform_nums_to_range(
    numbers: np.ndarray, old_bounds: List, new_bounds: List
) -> np.ndarray:
    """
    Transforms a given list of numbers from one range into a new range
    :param numbers: Numbers to transform
    :param old_bounds: Current range of numbers
    :param new_bounds: Range to convert numbers to
    :return: Range transformed numbers
    """
    if len(old_bounds) != 2 or len(new_bounds) != 2:
        raise ValueError("Range must describe a lowe and upper bound.")

    old_range = old_bounds[1] - old_bounds[0]
    new_range = new_bounds[1] - new_bounds[0]

    tformed_numbers = []
    for num in numbers:
        tformed_num = (((num - old_bounds[0]) * new_range) / old_range) + new_bounds[0]
        tformed_numbers.append(tformed_num)

    return np.array(tformed_numbers)


def point_bb_collided(point: np.ndarray, bb_name: str) -> bool:
    """
    Compute whether the given point is in collision with the give bounding box
    :param point: Point to check collision for
    :param bb_name: Name of bounding box which will be obtained from predefined list of court bounding boxes and checked
    for collision
    :return: True if the point and bounding box are in collision, false otherwise
    """
    if point.shape[0] != 3:
        raise ValueError("Expecting a 3D point.")

    # For irregular cuboid volumes, use Delaunay triangulation to detect if point is in polyhedron
    bb = FIELD_BOUNDING_BOXES[bb_name]
    if bb_name.startswith(("left", "right")):
        poly = bb["verts"]
        return Delaunay(poly).find_simplex(point) >= 0

    # For cuboid use simple coordinate comparison since BBs are axis aligned
    return (
        (bb["min_x"] <= point[0].item() <= bb["max_x"])
        and (bb["min_y"] <= point[1].item() <= bb["max_y"])
        and (bb["min_z"] <= point[2].item() <= bb["max_z"])
    )


def plot_bb(
    bb_name: str,
    ax: plt.axes,
    *,
    bb_face_annotation: str = None,
    show_vertices: bool = False,
    show_annotation: bool = False,
) -> None:
    """Plot the given bounding box (obtained from predefined list of court BBs) on the given axis - ax"""
    bb = FIELD_BOUNDING_BOXES[bb_name]

    if bb_name.startswith(("left", "right")):
        verts = bb["verts"]
    else:
        verts = [
            [x, y, z]
            for x in [bb["min_x"], bb["max_x"]]
            for y in [bb["min_y"], bb["max_y"]]
            for z in [bb["min_z"], bb["max_z"]]
        ]
    verts = np.array(verts)

    # Isolate vertices of plane which correspond to the inner face of the wall polyhedron via masking
    if bb_name.startswith(("front", "tin")):
        face_verts = verts[~np.any(verts == HALF_COURT_LENGTH + BB_DEPTH, axis=1)]
        # Swap face corners for non-intersecting plane plotting
        face_verts[[0, 1]] = face_verts[[1, 0]]
    elif bb_name.startswith("right"):
        face_verts = verts[~np.any(verts == HALF_COURT_WIDTH + BB_DEPTH, axis=1)]
    elif bb_name.startswith("left"):
        face_verts = verts[~np.any(verts == -HALF_COURT_WIDTH - BB_DEPTH, axis=1)]
    elif bb_name.startswith("back"):
        face_verts = verts[~np.any(verts == -HALF_COURT_LENGTH - BB_DEPTH, axis=1)]
        # Swap face corners for non-intersecting plane plotting
        face_verts[[0, 1]] = face_verts[[1, 0]]
    else:  # Back wall case
        raise ValueError(f"Plotting face for {bb_name} not implemented")

    # Annotate center of bounding boxes inner face
    if show_annotation:
        face_center_vert = np.mean(face_verts, axis=0)
        ax.text(
            face_center_vert[0],
            face_center_vert[2],
            face_center_vert[1],
            bb_face_annotation,
            zdir="y",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

    # Convert vertices to list of lists for Poly3DCollection and swap z and y since no zdir param
    face_verts = [list(zip(face_verts[:, 0], face_verts[:, 2], face_verts[:, 1]))]

    # Plot volume inner face and vertices
    ax.add_collection3d(
        Poly3DCollection(face_verts, color=bb["colour"], alpha=0.3, lw=0.1)
    )
    if show_vertices:
        ax.scatter3D(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            zdir="y",
            color=bb["colour"],
        )


def approximate_homography(video_path: Path) -> np.ndarray:
    """
    Approximate the homography which transforms points from the image plane to a given world plane and vice-versa
    :param video_path: Video to extract frame from to approximate homography
    :return: The homography matrix
    """
    frames = extract_frames_from_vid(video_path, disable_progbar=True)
    first_frame = frames[0].copy()

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

    h, _ = cv.findHomography(
        front_wall_image_coords, FRONT_WALL_WORLD_COORDS, method=cv.RANSAC
    )

    return h


def get_sim_ball_pos(
    sim_id: int, root_dir_path: Path, n_frames_to_avg: int
) -> np.ndarray:
    """Read the ball positions of the given simulation id from file"""
    # Get true ball positions (output at time of simulation data exporting)
    BALL_POS_TRUE = root_dir_path / "ball_pos" / f"sim_{sim_id}.csv"
    ball_pos_WC = pd.DataFrame(pd.read_csv(BALL_POS_TRUE), columns=["x", "y", "z"])

    # Get position of ball in frames rather than simulation steps (different due to blurring by averaging frames)
    ball_pos_frames_WC = ball_pos_WC.iloc[
        n_frames_to_avg::n_frames_to_avg, :
    ].reset_index(drop=True)

    x = ball_pos_frames_WC["x"].to_numpy()
    y = ball_pos_frames_WC["y"].to_numpy()
    z = ball_pos_frames_WC["z"].to_numpy()

    return np.c_[x, y, z]


def get_init_ball_pos(_vid_dir, video_fname: str) -> Tuple[float, float]:
    """Obtain the ball position in the first frame of the video from the user"""
    video_file_path = _vid_dir / video_fname
    frames = extract_frames_from_vid(video_file_path)
    first_frame = frames[0].copy()
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

    return click_store.click_pos()


# ToDo: Classes below could easily be one class
class SinglePosStore:
    """Stores a single 2D point, used for obtaining the ball position in a frame manually"""

    def __init__(self, first_frame: np.ndarray):
        self._frame: np.ndarray = first_frame
        self._click_pos: Tuple[int, int] = None

    def img_clicked(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(self._frame, (x, y), 7, (0, 255, 0))
            self._click_pos = (x, y)

    def click_pos(self) -> Tuple:
        return self._click_pos


class FourCoordsStore:
    """Stores 4, 2D points. Used for obtaining 4 image points to use for homography approximation"""

    def __init__(self, first_frame: np.ndarray):
        self._frame: np.ndarray = first_frame
        self._coords: List = []

    def img_clicked(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(self._frame, (x, y), 7, (255, 0, 0))
            self._coords.append([x, y])

    def click_pos(self) -> np.ndarray:
        return np.array(self._coords, dtype="float32")
