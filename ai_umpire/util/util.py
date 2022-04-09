import logging
from pathlib import Path
from typing import List, Tuple, Dict

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from numpy.linalg import inv, det
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
]

CAM_EXTRINSICS_HOMOG: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.75, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 3.0, -13.0, 1.0],
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
    pos_x_wc: float, pos_y_wc: float, pos_z_wc: float, img_dims: List[int]
) -> Tuple[int, int]:
    homog_ball_wc: np.ndarray = np.reshape(
        np.array([pos_x_wc, pos_y_wc, pos_z_wc, 1]), (1, 4)
    )
    homog_tformed_ball_wc: np.ndarray = homog_ball_wc @ CAM_EXTRINSICS_HOMOG_INV
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
            cv.THRESH_BINARY + cv.THRESH_OTSU,
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

    eroded_frames: List[np.ndarray] = []
    for i in tqdm(
        range(frames.shape[0]),
        desc=f"Applying morph. op. ({morph_op})",
        disable=disable_progbar,
    ):
        eroded_frame = cv.morphologyEx(
            src=frames[i],
            op=morph_ops[morph_op],
            kernel=cv.getStructuringElement(struc_el, kernel_shape),
            iterations=n_iter,
        )
        eroded_frames.append(eroded_frame)

    return np.array(eroded_frames)


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
        ret, frame = v_cap.read()

        # If frame is read correctly ret is True
        if ret:
            frames.append(frame)
            pbar.update(1)
        else:
            break

    v_cap.release()
    logging.info("Frames extracted successfully.")
    return np.array(frames)
