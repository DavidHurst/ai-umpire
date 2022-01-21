__all__ = ["Localiser"]

import logging
from math import sqrt
from pathlib import Path

import cv2 as cv
import numpy as np
from typing import List

from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.feature import blob_log, blob_dog, blob_doh

from tqdm import tqdm


class Localiser:
    def __init__(self, root: Path) -> None:
        self._root: Path = root
        self._frames: np.ndarray

    def extract_frames(self, vid_path: Path) -> np.ndarray:
        logging.info("Extracting frames from video.")
        v_cap: cv.VideoCapture = cv.VideoCapture(str(vid_path))
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

        flattened_frames = np.vstack([f.reshape(1, f.shape[0] * f.shape[1]) for f in frames])
        print([f.shape for f in frames])
        # print(f'flattend shape={flattened_frames.shape}')
        # median_image = np.median(frames, axis=0).reshape(frames.shape[1:])
        # print(f'median image shape={median_image.shape}')
        # plt.imshow(median_image)
        # plt.show()
        exit()

        for i in tqdm(range(1, frames.shape[0]), desc="Segmenting foreground"):
            first_grey: np.ndarray = np.mean(frames[i - 1], axis=2)
            second_grey: np.ndarray = np.mean(frames[i], axis=2)

            foreground_segmented_frames.append(first_grey - second_grey)



            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(first_grey - second_grey)
            axs[0, 1].imshow(dilated_frame)
            axs[1, 0].imshow()

            axs[0, 0].set_title("Normal Diff")
            axs[0, 1].set_title("Median of Video")
            axs[1, 0].set_title("Diff of Frame from Median Image")

            axs[0, 0].axis("off")
            axs[0, 1].axis("off")
            axs[1, 0].axis("off")
            plt.show()

        return np.array(foreground_segmented_frames)

    def apply_dilation(self, foreground_segmented_frames: np.ndarray) -> np.ndarray:
        kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        frames: List[np.ndarray] = []

        for i in tqdm(
            range(foreground_segmented_frames.shape[0]), desc="Applying dilation"
        ):
            frame: np.ndarray = foreground_segmented_frames[i]

            # Normalise to uint8 range and convert dtype to uint8
            frame_normed_uint8 = cv.normalize(
                frame, None, 0, 255, cv.NORM_MINMAX
            ).astype(np.uint8)

            # Apply opening to frame
            dilated_frame: np.ndarray = cv.morphologyEx(
                src=frame_normed_uint8, op=cv.MORPH_DILATE, kernel=kernel, iterations=1
            )

            # Apply binary thresholding with otsu's method
            _, thresh = cv.threshold(
                dilated_frame, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )

            # Apply blur to reduce noise and threshold
            blur = cv.GaussianBlur(dilated_frame, (3, 3), 0)
            _, thresh_blur = cv.threshold(
                blur, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )

            # fig, axs = plt.subplots(2, 2)
            # axs[0, 0].imshow(frame_normed_uint8, cmap="gray", vmin=0, vmax=255)
            # axs[0, 1].imshow(dilated_frame, cmap="gray", vmin=0, vmax=255)
            # axs[1, 0].imshow(thresh, cmap="gray", vmin=0, vmax=255)
            # axs[1, 1].imshow(thresh_blur, cmap="gray", vmin=0, vmax=255)
            # 
            # axs[0, 0].set_title("Foreground Segmented")
            # axs[0, 1].set_title("Dilated")
            # axs[1, 0].set_title("Dilated + Threshold")
            # axs[1, 1].set_title("Dilated + Threshold + Blur")
            # 
            # axs[0, 0].axis("off")
            # axs[0, 1].axis("off")
            # axs[1, 0].axis("off")
            # axs[1, 1].axis("off")
            # plt.show()

            frames.append(thresh_blur)

        return np.array(frames)

    def localise_ball_blob_filter(
        self, foreground_segmented_frames: np.ndarray
    ) -> None:
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()

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
        detector = cv.SimpleBlobDetector_create(params)

        for i in tqdm(
            range(foreground_segmented_frames.shape[0]),
            desc="Detecting blobs in frames",
        ):
            frame: np.ndarray = foreground_segmented_frames[i]

            # Normalise to uint8 range and convert dtype to uint8
            frame_normed_uint8 = cv.normalize(
                frame, None, 0, 255, cv.NORM_MINMAX
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
            colour_im = cv.imread(str(colour_im_path), cv.IMREAD_GRAYSCALE)
            display_img = cv.cvtColor(colour_im, cv.COLOR_GRAY2BGR)
            im_with_keypoints = cv.drawKeypoints(
                display_img,
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            # Show keypoints
            # cv.imshow(f"Frame {i} with Keypoints Drawn", im_with_keypoints)
            # cv.waitKey(0)
            # plt.tight_layout()
            # plt.imsave(f"frame_{str(i).zfill(5)}_simple_blob_filter.jpg", im_with_keypoints)

    def localise_ball_hough(self, foreground_segmented_frames: np.ndarray):
        pass

    def localise_ball_hough_circle(
        self, foreground_segmented_frames: np.ndarray
    ) -> None:
        for i in tqdm(
            range(foreground_segmented_frames.shape[0]),
            desc="Detecting circles in frames",
        ):
            # Normalise to uint8 range and convert dtype to uint8
            frame: np.ndarray = foreground_segmented_frames[i]
            frame_normed_uint8 = cv.normalize(
                frame, None, 0, 255, cv.NORM_MINMAX
            ).astype(np.uint8)

            # Detect circles
            min_dist_between_circles: float = frame_normed_uint8.shape[0] * 0.1
            max_radius: int = int(frame_normed_uint8.shape[1] * 0.1)
            print(f"Max radius={max_radius}, min dist={min_dist_between_circles}")
            circles_detected = cv.HoughCircles(
                image=frame_normed_uint8,
                minDist=min_dist_between_circles,
                method=cv.HOUGH_GRADIENT_ALT,
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
            colour_im = cv.imread(str(colour_im_path), cv.IMREAD_GRAYSCALE)
            display_img = cv.cvtColor(colour_im, cv.COLOR_GRAY2BGR)
            if circles_detected is not None:
                for x, y, r in circles_detected[0]:
                    cv.circle(display_img, (int(x), int(y)), int(r), (0, 0, 255))

            # cv.imshow(f"Frame {i} with Circles Drawn", display_img)
            # cv.waitKey(0)
            # plt.tight_layout()
            # plt.imsave(f"frame_{str(i).zfill(5)}_hough_circle.jpg", display_img)

    def localise_ball_blob(
        self, foreground_segmented_frames: np.ndarray, method: str
    ) -> None:
        for i in tqdm(
            range(foreground_segmented_frames.shape[0]),
            desc=f"Detecting blobs, method={method}",
        ):
            method_types: List[str] = ["log", "dog", "doh"]
            if method not in method_types:
                e: ValueError = ValueError(
                    f"Invalid method/method not supported, available options: {method_types}"
                )
                logging.exception(e)
                raise e
            logging.info(f"Detecting blobs using {method}.")

            # Normalise to uint8 range and convert dtype to uint8
            frame: np.ndarray = foreground_segmented_frames[i]
            frame_normed_uint8 = cv.normalize(
                frame, None, 0, 255, cv.NORM_MINMAX
            ).astype(np.uint8)

            # Detect blobs- can give kernel std devs as sequence per axis maybe to elongate blobs?
            blobs: np.ndarray = None
            if method == "log":
                blobs: np.ndarray = blob_log(
                    frame_normed_uint8,
                    min_sigma=1,
                    max_sigma=20,
                    num_sigma=4,
                    threshold=0.1,
                )
            if method == "dog":
                blobs = blob_dog(
                    frame_normed_uint8,
                    min_sigma=1,
                    max_sigma=20,
                    sigma_ratio=1.6,
                    threshold=0.1,
                )
            if method == "doh":
                blobs: np.ndarray = blob_doh(
                    frame_normed_uint8,
                    min_sigma=1,
                    max_sigma=20,
                    num_sigma=1,
                    threshold=0.005,
                )

            # Compute radii in the third column
            if method == "log" or "dog":
                blobs[:, 2] = blobs[:, 2] * sqrt(2)

            # Draw detected blobs
            colour_im_path = (
                Path(
                    "C:\\Users\\david\\Data\\AI Umpire DS\\blurred_frames\\sim_0_blurred"
                )
                / f"frame{str(i).zfill(5)}.jpg"
            )
            colour_im = cv.imread(str(colour_im_path), cv.IMREAD_GRAYSCALE)
            display_img = cv.cvtColor(colour_im, cv.COLOR_GRAY2BGR)
            for blob in blobs:
                y, x, r = blob
                cv.circle(display_img, (int(x), int(y)), int(r), (0, 0, 255))

                # cv.imshow(f"Frame #{i} with Blobs Drawn, Method={method}", display_img)
                # cv.waitKey(0)
            # plt.tight_layout()
            # plt.imsave(f"frame_{str(i).zfill(5)}_{method}.jpg", display_img)
