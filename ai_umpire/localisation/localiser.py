__all__ = ["Localiser"]

import logging
from pathlib import Path

import cv2
import numpy as np
from typing import List

from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.io import imshow

from tqdm import tqdm


class Localiser:
    def __init__(self, root: Path) -> None:
        self._root: Path = root
        self._frames: np.ndarray

    def extract_frames(self, vid_path: Path) -> np.ndarray:
        logging.info("Extracting frames from video.")
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
        logging.info("Frames extracted successfully.")
        return np.array(frames)

    def segment_foreground(self, frames: np.ndarray) -> np.ndarray:
        foreground_segmented_frames: List[np.ndarray] = []
        for i in tqdm(range(1, frames.shape[0]), desc="Segmenting foreground"):
            first_grey: np.ndarray = np.mean(frames[i - 1], axis=2)
            second_grey: np.ndarray = np.mean(frames[i], axis=2)

            foreground_segmented_frames.append(first_grey - second_grey)

        return np.array(foreground_segmented_frames)


    def localise_ball_blob(self, foreground_segmented_frames: np.ndarray) -> List[np.ndarray]:
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1500

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        for i in tqdm(range(foreground_segmented_frames.shape[0]), desc='Detecting blobs in frames'):
            frame: np.ndarray = foreground_segmented_frames[i]

            # Normalise to uint8 range and convert dtype to uint8
            frame_normed_uint8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            keypoints = detector.detect(frame_normed_uint8)



