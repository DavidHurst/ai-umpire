__all__ = ["Detector"]

import logging
import math
from math import sqrt
from pathlib import Path

import cv2 as cv
import numpy as np
from typing import List, Tuple

from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from skimage.feature import blob_log, blob_dog, blob_doh

from tqdm import tqdm

from ai_umpire.util import (
    extract_frames_from_vid,
    difference_frames,
    blur_frames,
    binarize_frames,
    apply_morph_op,
)


class Detector:
    def __init__(self, root_dir: Path):
        self._root_dir: Path = root_dir
        self._vid_dir: Path = self._root_dir / "videos"
        self._blurred_dir: Path = self._root_dir / "blurred_frames"
        self._frames_dir: Path = self._root_dir / "sim_frames"
        self._all_detections = []

    def _localise_ball_blob_filter(self, frames: np.ndarray) -> np.ndarray:
        detections: List[List] = []

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
            range(frames.shape[0]),
            desc="Localising ball (blob filter)",
        ):
            # Detect blobs
            keypoints = detector.detect(frames[i])

            # # Return first blob detected as detection (temporary)
            if keypoints:
                for kp in keypoints:
                    detections.append([kp.pt[1], kp.pt[0], kp.size])
                    break
            else:
                # No detections made
                detections.append([10, 10, 5])

        return np.array(detections)

    def _localise_ball_hough_circle(self, frames: np.ndarray) -> np.ndarray:
        detections: List[List] = []
        for i in tqdm(
            range(frames.shape[0]),
            desc="Localising ball (Hough circle)",
        ):
            # Detect circles
            max_radius: int = int(frames[i].shape[1] * 0.4)
            circles_detected = cv.HoughCircles(
                image=frames[i],
                method=cv.HOUGH_GRADIENT,
                dp=1,
                minDist=20.0,
                param1=50,
                param2=30,
                minRadius=0,
                maxRadius=0,
            )

            if circles_detected is not None:
                for x, y, r in circles_detected[0]:
                    print(f"Frame #{i}: (x={x},y={y}), rad={r}")
            else:
                print("No detections")

            # Return first detection (temporary)
            if circles_detected is not None:
                for x, y, r in circles_detected[0]:
                    detections.append([y, x, r])
                    break
            else:
                # No detections made
                detections.append([10, 10, 5])

        return np.array(detections)

    def _localise_ball_blob(self, frames: np.ndarray, method: str) -> np.ndarray:
        detections: List[List] = []
        for i in tqdm(
            range(frames.shape[0]),
            desc=f"Localising ball ({method})",
        ):
            method_types: List[str] = ["log", "dog", "doh"]
            if method not in method_types:
                e: ValueError = ValueError(
                    f"Invalid method/method not supported, available options: {method_types}"
                )
                logging.exception(e)
                raise e
            logging.info(f"Detecting blobs using {method}.")

            frame = frames[i]

            # Detect blobs- can give kernel std devs as sequence per axis maybe to elongate blobs?
            blobs: np.ndarray = None
            if method == "log":
                blobs: np.ndarray = blob_log(
                    frame,
                    min_sigma=10,
                    max_sigma=50,
                    num_sigma=5,
                    threshold=0.0001,
                )
            elif method == "dog":
                blobs = blob_dog(
                    frame,
                    min_sigma=5,
                    max_sigma=10,
                    sigma_ratio=1.6,
                    threshold=0.0001,
                )
            elif method == "doh":
                blobs: np.ndarray = blob_doh(
                    frame,
                    min_sigma=1,
                    max_sigma=20,
                    num_sigma=1,
                    threshold=0.005,
                )

            # Compute radii in the third column
            if method == "log" or "dog":
                blobs[:, 2] = blobs[:, 2] * sqrt(2)
            for blob in blobs:
                y, x, r = blob
                print(f"Frame #{i}: (x={x},y={y}), rad={r}")

            # Return first blob detected as detection (temporary)
            for blob in blobs:
                y, x, r = blob
                detections.append([y, x, r])
                break

        return np.array(detections)

    def get_ball_candidates_contour(
        self,
        sim_id: int,
        morph_op: str,
        morph_op_iters: int,
        morph_op_SE_shape: Tuple[int, int],
        struc_el_shape: np.ndarray,
        blur_kernel_size: Tuple[int, int],
        blur_sigma_x: int,
        binary_thresh_low: int,
        disable_progbar: bool = False,
    ) -> List[List]:
        detections: List[List] = []
        # Extract frames from video
        vid_path: Path = self._vid_dir / f"sim_{sim_id}.mp4"
        video_frames: np.ndarray = extract_frames_from_vid(
            vid_path, disable_progbar=disable_progbar
        )

        # Preprocess frames
        blurred_frames: np.ndarray = blur_frames(
            video_frames,
            blur_kernel_size,
            blur_sigma_x,
            disable_progbar=disable_progbar,
        )
        fg_seg_frames: np.ndarray = difference_frames(
            blurred_frames, disable_progbar=disable_progbar
        )
        binary_frames: np.ndarray = binarize_frames(
            fg_seg_frames, binary_thresh_low, disable_progbar=disable_progbar
        )
        morph_op_frames: np.ndarray = apply_morph_op(
            binary_frames,
            morph_op,
            morph_op_iters,
            morph_op_SE_shape,
            struc_el=struc_el_shape,
            disable_progbar=disable_progbar,
        )

        for i in tqdm(
            range(morph_op_frames.shape[0]),
            desc=f"Localising ball (contour det.)",
            disable=disable_progbar,
        ):
            # fig, axes = plt.subplots(1, 3, figsize=(15, 7))
            contours, hierarchy = cv.findContours(
                morph_op_frames[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            if contours:
                estimated_pos = []
                for c in contours:
                    # Compute contour centroid
                    M = cv.moments(c)
                    m00 = M["m00"] + 1e-5  # Add 1e-5 to avoid div by 0
                    contour_centroid_x = int(M["m10"] / m00)
                    contour_centroid_y = int(M["m01"] / m00)

                    # Compute area of contour
                    contour_area = cv.contourArea(c)

                    estimated_pos.append(
                        (
                            contour_centroid_x,
                            contour_centroid_y,
                            math.sqrt(contour_area),
                        )
                    )

                detections.append(estimated_pos)

                display_im = cv.imread(
                    str(
                        self._blurred_dir
                        / f"sim_{sim_id}_blurred"
                        / f"frame{str(i).zfill(5)}.png"
                    )
                )
                # cv.drawContours(display_im, contours, -1, (0, 0, 255), 2)
                # cv.imshow(f"Frame #{i} - Contours", display_im)
                # cv.imshow(f"Frame #{i} - Differenced", fg_seg_frames[i])
                # cv.imshow(f"Frame #{i} Features", morph_op_frames[i])
                # cv.waitKey(0)
                # fig, ax = plt.subplots(figsize=(10, 10))
                # ax.imshow(fg_seg_frames[i], cmap="gray", vmin=0, vmax=255)
                # ax.imshow(morph_op_frames[i], cmap="gray", vmin=0, vmax=255)
                # ax.imshow(cv.cvtColor(display_im, cv.COLOR_BGR2RGB))
                # for ax in axes:
                # ax.axis("off")
                # plt.tight_layout()
                # # plt.savefig(f"detection{str(i).zfill(2)}.png")
                # plt.show()
            else:
                # No detections, worst values, position miles away from anywhere on the screen
                # and area infinitely large when ball should be small
                print(f"No detections frame #{i}")
                detections.append([(0, 0, float("inf"))])

        # ToDo: convert to numpy array
        self._all_detections = detections
        return detections

    def get_ball_candidates(
        self,
        vid_path: Path,
        morph_op: str,
        detection_method: str,
        morph_op_iters: int,
        morph_op_SE_shape: Tuple[int, int],
        blur_kernel_size: Tuple[int, int],
        blur_sigma_x: int,
        binary_thresh_low: int,
    ) -> np.ndarray:
        # Extract frames from video
        video_frames: np.ndarray = extract_frames_from_vid(vid_path)

        # Preprocess frames
        fg_seg_frames: np.ndarray = difference_frames(video_frames)
        blurred_frames: np.ndarray = blur_frames(
            fg_seg_frames, blur_kernel_size, blur_sigma_x
        )
        binary_frames: np.ndarray = binarize_frames(blurred_frames, binary_thresh_low)
        morph_op_frames: np.ndarray = apply_morph_op(
            binary_frames, morph_op, morph_op_iters, morph_op_SE_shape
        )

        # Detect ball in processed frames
        if detection_method == "log" or detection_method == "dog":
            detections: np.ndarray = self._localise_ball_blob(
                morph_op_frames, detection_method
            )
        elif detection_method == "blob_filter":
            detections: np.ndarray = self._localise_ball_blob_filter(morph_op_frames)
        elif detection_method == "hough_circle":
            detections: np.ndarray = self._localise_ball_hough_circle(morph_op_frames)
        elif detection_method == "contour":
            detections: np.ndarray = self._localise_ball_contour(morph_op_frames)
        else:
            e: ValueError = ValueError("Invalid detection method chosen.")
            logging.exception(e)
            raise e

        # self._display_detections(
        #     [
        #         video_frames,
        #         fg_seg_frames,
        #         blurred_frames,
        #         binary_frames,
        #         morph_op_frames,
        #         detections,
        #     ],
        #     morph_op,
        #     detection_method,
        # )

        # Return frame detections
        return detections

    def _display_detections(
        self, detection_phases_frames: List[np.ndarray], morph_op: str, det_method: str
    ):
        # Draw detections
        for i in range(detection_phases_frames[0].shape[0]):
            display_img: np.ndarray = detection_phases_frames[0][i].copy()
            # print(
            #     f"Frame #{i} = ({detection_phases_frames[-1][i][0]}, {detection_phases_frames[-1][i][1]}), radius={detection_phases_frames[-1][i][2]}"
            # )
            cv.circle(
                display_img,
                (
                    int(detection_phases_frames[-1][i][1]),
                    int(detection_phases_frames[-1][i][0]),
                ),
                int(detection_phases_frames[-1][i][2]),
                (0, 0, 255),
                2,
            )

            fig, axs = plt.subplots(2, 3, figsize=(12, 6))
            axs[0, 0].imshow(
                cv.cvtColor(detection_phases_frames[0][i], cv.COLOR_BGR2RGB)
            )
            axs[0, 1].imshow(
                cv.cvtColor(detection_phases_frames[1][i], cv.COLOR_BGR2RGB)
            )
            axs[0, 2].imshow(
                cv.cvtColor(detection_phases_frames[2][i], cv.COLOR_BGR2RGB)
            )
            axs[1, 0].imshow(
                detection_phases_frames[3][i], cmap="gray", vmin=0, vmax=255
            )
            axs[1, 1].imshow(
                detection_phases_frames[4][i], cmap="gray", vmin=0, vmax=255
            )
            axs[1, 2].imshow(cv.cvtColor(display_img, cv.COLOR_BGR2RGB))

            axs[0, 0].set_title("1. Original")
            axs[0, 1].set_title("2. FG Seg.")
            axs[0, 2].set_title("3. FG Seg.+Blur")
            axs[1, 0].set_title("4. FG Seg.+Blur+Binarize")
            axs[1, 1].set_title(f"5. FG Seg.+Blur+Binarize+Morph. Op.({morph_op})")
            axs[1, 2].set_title(f"6. Detection ({det_method})")

            axs[0, 0].axis("off")
            axs[0, 1].axis("off")
            axs[0, 2].axis("off")
            axs[1, 0].axis("off")
            axs[1, 1].axis("off")
            axs[1, 2].axis("off")
            plt.tight_layout()
            # fig.savefig(f"detection{i}.png")
            plt.show()
