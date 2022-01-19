__all__ = ["Localiser"]

import logging
from pathlib import Path

import cv2
import numpy as np
from typing import List

from matplotlib import pyplot as plt
from skimage.io import imshow

from tqdm import tqdm


class Localiser:
    def __init__(self, root: Path) -> None:
        self._root: Path = root
        self._frames: np.ndarray

    def extract_frames(self, vid_path: Path) -> np.ndarray:
        logging.info('Extracting frames from video.')
        v_cap: cv2.VideoCapture = cv2.VideoCapture(str(vid_path))
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
        logging.info('Frames extracted successfully.')
        return np.array(frames)

    def segment_foreground(self, frames: np.ndarray) -> np.ndarray:
        foreground_segmented_frames: List[np.ndarray] = []
        for i in tqdm(range(1, frames.shape[0]), desc="Differencing frames"):
            first_grey: np.ndarray = np.mean(frames[i - 1], axis=2)
            second_grey: np.ndarray = np.mean(frames[i], axis=2)

            diff: np.ndarray = first_grey - second_grey

            foreground_segmented_frames.append(diff)

        return np.array(foreground_segmented_frames)

# Segment foreground/moving objects

# Apply elliptical Hough/blob detection
