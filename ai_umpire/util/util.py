import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import cv2 as cv
import numpy as np
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
]

from ai_umpire.util import (
    FIELD_BOUNDING_BOXES,
    HALF_COURT_LENGTH,
    BB_DEPTH,
    HALF_COURT_WIDTH,
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
    # The format OpenCV loaded images are in
    if img_dims[0] > img_dims[1]:
        warnings.warn(
            "Warning, expecting image dimensions given as height x width, possible opposite provided."
        )
    img_dims.reverse()
    """Only works for synthetic videos in 720p as the camera matrix has been extracted specifically for that use."""
    pos_x_wc, pos_y_wc, pos_z_wc = pos_wc[0], pos_wc[1], pos_wc[2]

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
    foreground_segmented_frames: List[np.ndarray] = []

    for i in tqdm(
        range(1, frames.shape[0] - 1),
        desc="Differencing frames",
        disable=disable_progbar,
    ):
        # Convert all to greyscale
        preceding: np.ndarray = frames[i - 1]
        current: np.ndarray = frames[i]
        succeeding: np.ndarray = frames[i + 1]

        diff = cv.bitwise_and(current - preceding, succeeding - current)
        # c_p = current - preceding
        # s_c = succeeding - current

        foreground_segmented_frames.append(diff)

        # fig, axes = plt.subplots(1, 3, figsize=(15, 7))
        # axes[0].imshow(c_p)  # , cmap="gray", vmin=0, vmax=255)
        # axes[1].imshow(s_c)  # , cmap="gray", vmin=0, vmax=255)
        # axes[2].imshow(np.mean(diff, -1), cmap="gray", vmin=0, vmax=255)
        # for ax in axes:
        #     ax.axis("off")
        # plt.tight_layout()
        # plt.show()

    return np.array(foreground_segmented_frames)


def extract_frames_from_vid(
    vid_path: Path, disable_progbar: bool = False
) -> np.ndarray:
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


# ToDo: Classes below could easily be one class
class SinglePosStore:
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
    def __init__(self, first_frame: np.ndarray):
        self._frame: np.ndarray = first_frame
        self._coords: List = []

    def img_clicked(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(self._frame, (x, y), 7, (255, 0, 0))
            self._coords.append([x, y])

    def click_pos(self) -> np.ndarray:
        return np.array(self._coords, dtype="float32")
