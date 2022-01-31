from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2 as cv
import logging

from tqdm import tqdm
import pychrono as chrono

__all__ = [
    "extract_frames_from_vid",
    "MyReportContactCallback",
    "difference_frames",
    "blur_frames",
    "binarize_frames",
    "apply_morph_op",
]


def binarize_frames(
    frames: np.ndarray, thresh_low: int, thresh_high: int = 255
) -> np.ndarray:
    binary_frames: List[np.ndarray] = []
    for i in tqdm(range(frames.shape[0]), desc="Binarizing frames"):
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
    frames: np.ndarray, kernel_sz: Tuple[int, int], sigma_x: int
) -> np.ndarray:
    blurred_frames: List[np.ndarray] = []
    for i in tqdm(range(frames.shape[0]), desc="Blurring frames"):
        blurred_frames.append(cv.GaussianBlur(frames[i], kernel_sz, sigma_x))

    return np.array(blurred_frames)


def apply_morph_op(
    frames: np.ndarray,
    morph_op: str,
    n_iter: int,
    kernel_shape: Tuple[int, int],
    struc_el: np.ndarray = cv.MORPH_ELLIPSE,
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
    for i in tqdm(range(frames.shape[0]), desc=f"Applying morph. op. ({morph_op})"):
        eroded_frame = cv.morphologyEx(
            src=frames[i],
            op=morph_ops[morph_op],
            kernel=cv.getStructuringElement(struc_el, kernel_shape),
            iterations=n_iter,
        )
        eroded_frames.append(eroded_frame)

    return np.array(eroded_frames)


def difference_frames(frames: np.ndarray) -> np.ndarray:
    foreground_segmented_frames: List[np.ndarray] = []

    for i in tqdm(range(1, frames.shape[0] - 1), desc="Differencing frames"):
        preceding: np.ndarray = frames[i - 1]
        current: np.ndarray = frames[i]
        succeeding: np.ndarray = frames[i + 1]

        foreground_segmented_frames.append(
            cv.bitwise_and(current - preceding, succeeding - current)
        )

    return np.array(foreground_segmented_frames)


def extract_frames_from_vid(vid_path: Path) -> np.ndarray:
    logging.info("Extracting frames from video.")
    v_cap: cv.VideoCapture = cv.VideoCapture(str(vid_path), cv.CAP_FFMPEG)
    frames: List[np.ndarray] = []

    pbar: tqdm = tqdm(desc="Extracting frames")
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