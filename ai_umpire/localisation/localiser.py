__all__ = ["Localiser"]

import logging
from pathlib import Path

import cv2
import numpy as np
from typing import List

from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.feature import blob_log
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

    def localise_ball_blob_filter(
        self, foreground_segmented_frames: np.ndarray
    ) -> None:
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = False

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 150

        # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 1500

        # Filter by Circularity, a circle has a circularity of 1
        params.filterByCircularity = False
        params.minCircularity = 0
        params.maxCircularity = 1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.9

        # Filter by Inertia, higher value means search for more elongation
        params.filterByInertia = True
        params.minInertiaRatio = 0
        params.maxInertiaRatio = 1

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        for i in tqdm(
            range(foreground_segmented_frames.shape[0]),
            desc="Detecting blobs in frames",
        ):
            frame: np.ndarray = foreground_segmented_frames[i]

            # Normalise to uint8 range and convert dtype to uint8
            frame_normed_uint8 = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Detect blobs
            keypoints = detector.detect(frame_normed_uint8)

            # Draw blobs
            colour_im_path = (
                    Path(
                        "C:\\Users\\david\\Data\\AI Umpire DS\\blurred_frames\\sim_0_blurred"
                    )
                    / f"frame{str(i).zfill(5)}.jpg"
            )
            colour_im = cv2.imread(str(colour_im_path), cv2.IMREAD_GRAYSCALE)
            display_img = cv2.cvtColor(colour_im, cv2.COLOR_GRAY2BGR)
            im_with_keypoints = cv2.drawKeypoints(
                display_img,
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            # Show keypoints
            # cv2.imshow(f"Frame {i} with Keypoints Drawn", im_with_keypoints)
            # cv2.waitKey(0)
            # plt.tight_layout()
            # plt.imsave(f"frame_{str(i).zfill(5)}_simple_blob_filter.jpg", im_with_keypoints)

    def localise_ball_hough_circle(
        self, foreground_segmented_frames: np.ndarray
    ) -> None:
        for i in tqdm(
            range(foreground_segmented_frames.shape[0]),
            desc="Detecting circles in frames",
        ):
            # Normalise to uint8 range and convert dtype to uint8
            frame: np.ndarray = foreground_segmented_frames[i]
            frame_normed_uint8 = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Detect circles
            min_dist_between_circles: float = frame_normed_uint8.shape[0] * 0.1
            max_radius: int = int(frame_normed_uint8.shape[1] * 0.1)
            print(f"Max radius={max_radius}, min dist={min_dist_between_circles}")
            circles_detected = cv2.HoughCircles(
                image=frame_normed_uint8,
                minDist=min_dist_between_circles,
                method=cv2.HOUGH_GRADIENT_ALT,
                dp=1,
                param1=400,
                param2=0.2,
                minRadius=0,
                maxRadius=max_radius,
            )

            # Draw circles
            colour_im_path = (
                Path(
                    "C:\\Users\\david\\Data\\AI Umpire DS\\blurred_frames\\sim_0_blurred"
                )
                / f"frame{str(i).zfill(5)}.jpg"
            )
            colour_im = cv2.imread(str(colour_im_path), cv2.IMREAD_GRAYSCALE)
            display_img = cv2.cvtColor(colour_im, cv2.COLOR_GRAY2BGR)
            if circles_detected is not None:
                for x, y, r in circles_detected[0]:
                    cv2.circle(display_img, (int(x), int(y)), int(r), (0, 0, 255))

            # cv2.imshow(f"Frame {i} with Circles Drawn", display_img)
            # cv2.waitKey(0)
            # plt.tight_layout()
            # plt.imsave(f"frame_{str(i).zfill(5)}_hough_circle.jpg", display_img)

    def localise_ball_blob_log(self, foreground_segmented_frames: np.ndarray) -> None:
        for i in tqdm(
            range(foreground_segmented_frames.shape[0]),
            desc="Detecting blobs in frames (LoG)",
        ):
            # Normalise to uint8 range and convert dtype to uint8
            frame: np.ndarray = foreground_segmented_frames[i]
            frame_normed_uint8 = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Detect blobs- can give kernel std devs as sequence per axis maybe to elongate blobs?
            blobs_log = blob_log(
                frame_normed_uint8,
                min_sigma=1,
                max_sigma=30,
                num_sigma=10,
                threshold=0.05,
            )

            # Draw detected blobs
            colour_im_path = (
                Path(
                    "C:\\Users\\david\\Data\\AI Umpire DS\\blurred_frames\\sim_0_blurred"
                )
                / f"frame{str(i).zfill(5)}.jpg"
            )
            colour_im = cv2.imread(str(colour_im_path), cv2.IMREAD_GRAYSCALE)
            display_img = cv2.cvtColor(colour_im, cv2.COLOR_GRAY2BGR)
            for blob in blobs_log:
                y, x, r = blob
                cv2.circle(display_img, (int(x), int(y)), int(r), (0, 0, 255))

            # cv2.imshow(f"Frame {i} with Circles Drawn", display_img)
            # cv2.waitKey(0)
            # plt.tight_layout()
            # plt.imsave(f"frame_{str(i).zfill(5)}_log.jpg", display_img)
