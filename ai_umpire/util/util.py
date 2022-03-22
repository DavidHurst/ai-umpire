from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2 as cv
import logging

from numpy import pi
from numpy.linalg import inv, det
from tqdm import tqdm
import pychrono as chrono

__all__ = [
    "extract_frames_from_vid",
    "MyReportContactCallback",
    "difference_frames",
    "blur_frames",
    "binarize_frames",
    "apply_morph_op",
    "wc_to_ic",
    "multivariate_norm_pdf",
]

HOMOG_CAM_TFORM_MAT_INV: np.ndarray = np.linalg.inv(
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, -13.0, 1.0],
        ]
    )
)


def multivariate_norm_pdf(x: np.array, mu: np.array, sigma: np.array) -> float:
    if x.shape[0] != mu.shape[0] or sigma.shape != (x.shape[0], x.shape[0]):
        raise ValueError("Matrix/vector dimensions incompatible.")
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
    homog_tformed_ball_wc: np.ndarray = np.dot(homog_ball_wc, HOMOG_CAM_TFORM_MAT_INV)
    tformed_ball_wc: np.ndarray = homog_tformed_ball_wc[:, :-1]
    pos_ic_coefs: np.ndarray = np.array(
        [
            0.5 + (tformed_ball_wc[:, 0] / tformed_ball_wc[:, -1]),
            0.5 - (tformed_ball_wc[:, 1] / tformed_ball_wc[:, -1]),
        ]
    )
    pos_ic: np.ndarray = np.multiply(np.reshape(img_dims, (2, 1)), pos_ic_coefs)

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
    struc_el: np.ndarray = cv.MORPH_ELLIPSE,
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
        preceding: np.ndarray = frames[i - 1]
        current: np.ndarray = frames[i]
        succeeding: np.ndarray = frames[i + 1]

        foreground_segmented_frames.append(
            cv.bitwise_and(current - preceding, succeeding - current)
        )

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


class MyReportContactCallback(chrono.ReportContactCallback):
    def __init__(self):
        super(MyReportContactCallback, self).__init__()
        self._contacts = []

    def OnReportContact(
        self,
        contact_point_A,
        contact_point_B,
        plane_coord,
        distance,
        eff_radius,
        react_forces,
        react_torques,
        contactobjA,
        contactobjB,
    ):
        bodyUpA = chrono.CastContactableToChBody(contactobjA)
        nameA = bodyUpA.GetName()
        bodyUpB = chrono.CastContactableToChBody(contactobjB)
        nameB = bodyUpB.GetName()
        if nameB != "Floor" and nameA != "Floor":
            self._contacts.append("Contact between {nameA} & {nameB}")
        return True  # return False to stop reporting contacts

    def reset(self):
        self._contacts = []
