__all__ = ["BallDetector"]

import math
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

from ai_umpire.util import (
    extract_frames_from_vid,
    difference_frames,
    blur_frames,
    binarize_frames,
    apply_morph_op,
)
from ai_umpire.util.util import get_init_ball_pos

plt.rcParams["figure.figsize"] = (8, 4.5)


class BallDetector:
    def __init__(self, root_dir: Path):
        self._root_dir: Path = root_dir
        self._vid_dir: Path = self._root_dir / "videos"
        self._frames_dir: Path = self._root_dir / "frames"
        self._all_detections = []

    def get_ball_detections(
        self,
        vid_fname: str,
        morph_op: str,
        morph_op_iters: int,
        morph_op_se_shape: Tuple[int, int],
        struc_el: np.ndarray,
        blur_kernel_size: Tuple[int, int],
        blur_sigma: int,
        binary_thresh: int,
        *,
        disable_progbar: bool = False,
        visualise=None,
    ) -> List[List]:
        """
        Extracts frames from given video, applies Gaussian blur, differences then binarizes frames finally applying the
        specified morphological operation to each processed frame which then have their contours extracted and returned.
        :param vid_fname: The video to detect ball candidates in
        :param morph_op: The morphological operation to apply
        :param morph_op_iters: The number of iterations of the morphological operator to perform
        :param morph_op_se_shape: The shape of the morphological operator's structuring element
        :param struc_el: The structuring element to use for the morphological operation
        :param blur_kernel_size: The size of the kernel to use for Gaussian blurring
        :param blur_sigma: The effective strength of the Gaussian blurring to apply
        :param binary_thresh: The minimum pixel intensity threshold to use for binarization
        :param disable_progbar: Disables display of the progress bar if set to True
        :param visualise: Shows the detection process operation selected out of
                        ["blurred" "fg_seg", "binary", "morph", "contours", "filtering"]
        :return: All detections in each frame
        """
        if visualise is None:
            visualise = ["none"]
        detections: List[List] = []
        # Extract frames from video
        vid_path: Path = self._vid_dir / vid_fname
        video_frames: np.ndarray = extract_frames_from_vid(
            vid_path, disable_progbar=disable_progbar
        )

        # Blur all frames
        blurred_frames: np.ndarray = blur_frames(
            video_frames, blur_kernel_size, blur_sigma, disable_progbar=disable_progbar,
        )

        # Difference frames to extract foreground
        fg_seg_frames: np.ndarray = difference_frames(
            blurred_frames, disable_progbar=disable_progbar
        )

        # Binarize all frames
        binary_frames: np.ndarray = binarize_frames(
            fg_seg_frames, binary_thresh, disable_progbar=disable_progbar
        )

        # Apply morphological operator to all frames
        morph_op_frames: np.ndarray = apply_morph_op(
            binary_frames,
            morph_op,
            morph_op_iters,
            morph_op_se_shape,
            struc_el=struc_el,
            disable_progbar=disable_progbar,
        )

        # Apply contour detection and store all detections in each frame
        for i in tqdm(
            range(morph_op_frames.shape[0]),
            desc=f"Localising ball (contour det.)",
            disable=disable_progbar,
        ):
            # fig, axes = plt.subplots(1, 3, figsize=(15, 7))
            contours, hierarchy = cv.findContours(
                morph_op_frames[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
            if contours:
                estimated_pos = []
                for c in contours:
                    # Compute contour centroid
                    m = cv.moments(c)
                    m00 = m["m00"] + 1e-5  # Add 1e-5 to avoid div by 0
                    contour_centroid_x = int(m["m10"] / m00)
                    contour_centroid_y = int(m["m01"] / m00)

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

                # Visualise specified operations
                if "none" not in visualise:
                    if "blurred" in visualise:
                        plt.imshow(cv.cvtColor(blurred_frames[i], cv.COLOR_BGR2RGB))
                        plt.axis("off")
                        plt.tight_layout()
                        plt.show()
                    if "fg_seg" in visualise:
                        plt.imshow(cv.cvtColor(fg_seg_frames[i], cv.COLOR_BGR2RGB))
                        plt.axis("off")
                        plt.tight_layout()
                        plt.show()
                    if "binary" in visualise:
                        plt.imshow(binary_frames[i], cmap="gray", vmin=0, vmax=1)
                        plt.axis("off")
                        plt.tight_layout()
                        plt.show()

                    if "morph" in visualise:
                        plt.imshow(morph_op_frames[i], cmap="gray", vmin=0, vmax=255)
                        plt.axis("off")
                        plt.tight_layout()
                        plt.show()

                    if "contours" in visualise:
                        cv.drawContours(video_frames[i], contours, -1, (0, 0, 255), 2)
                        plt.imshow(cv.cvtColor(video_frames[i], cv.COLOR_BGR2RGB))
                        plt.axis("off")
                        plt.tight_layout()
                        plt.show()
            else:
                # No detections, indicator values used for filtering
                detections.append([(-1, -1, -1)])

        self._all_detections = detections
        return detections

    def _filter_ball_detections(
        self,
        frame_detections: List[List],
        init_ball_pos: Tuple[float, float],
        *,
        sim_id: int = None,
        min_ball_travel_dist: float = 5,
        max_ball_travel_dist: float = 130,
        min_det_area: float = 2.0,
        max_det_area: float = 65.0,
        disable_progbar: bool = False,
        visualise=None,
    ) -> List[List]:
        if visualise is None:
            visualise = ["none"]
        filtered_dets = []

        def get_frame_detections_com(frame_idx: int) -> Tuple[float, float]:
            """com = Center of Mass"""
            prev_frame_accepted_dets_x = [x for x, _, _ in filtered_dets[frame_idx]]
            prev_frame_accepted_dets_y = [y for _, y, _ in filtered_dets[frame_idx]]
            x_com = sum(prev_frame_accepted_dets_x) / len(prev_frame_accepted_dets_x)
            y_com = sum(prev_frame_accepted_dets_y) / len(prev_frame_accepted_dets_y)

            return x_com, y_com

        for i in tqdm(
            range(len(frame_detections)),
            desc=f"Filtering ball detections",
            disable=disable_progbar,
        ):
            # Filter detections base on their size, i.e. filter out the player detections and noise
            curr_frame_dets = [
                (x, y, z)
                for x, y, z in frame_detections[i]
                if min_det_area < z < max_det_area
            ]

            # Compare each detection in the current frame to the filtered detections from the previous frame to find
            # the detections within the acceptable range of motion
            if i > 0:
                velocity_constrained_dets = []
                for curr_frame_det in curr_frame_dets:
                    for prev_frame_filtered_det in filtered_dets[i - 1]:

                        curr_x, curr_y = curr_frame_det[0], curr_frame_det[1]
                        prev_x, prev_y = (
                            prev_frame_filtered_det[0],
                            prev_frame_filtered_det[1],
                        )

                        euclid_dist_between_prev_and_curr_dets = math.sqrt(
                            ((curr_x - prev_x) ** 2) + ((curr_y - prev_y) ** 2)
                        )
                        in_acceptable_range_of_motion = (
                            min_ball_travel_dist
                            < euclid_dist_between_prev_and_curr_dets
                            < max_ball_travel_dist
                        )

                        if (
                            in_acceptable_range_of_motion
                            and curr_frame_det not in filtered_dets[i - 1]
                        ):
                            velocity_constrained_dets.append(curr_frame_det)

                # If no detections in the current frame satisfy the velocity constraint, add the dectection closest to
                # the center of mass of the previous frame's detections
                if len(velocity_constrained_dets) == 0:
                    com_x, com_y = get_frame_detections_com(i - 1)

                    dets_dist_to_com = [
                        math.sqrt(((x - com_x) ** 2) + ((y - com_y) ** 2))
                        for x, y, _ in frame_detections[i]
                    ]
                    closest_det = frame_detections[i][
                        dets_dist_to_com.index(min(dets_dist_to_com))
                    ]
                    filtered_dets.append([closest_det])
                else:
                    filtered_dets.append(list(set(velocity_constrained_dets)))
            else:
                # Find the closest detection to user provided initial ball position
                dists = [
                    math.sqrt(
                        ((x - init_ball_pos[0]) ** 2) + ((y - init_ball_pos[1]) ** 2)
                    )
                    for x, y, _ in curr_frame_dets
                ]
                closest_det = curr_frame_dets[dists.index(min(dists))]

                filtered_dets.append([closest_det])

            # Visualise filtered detections
            if (
                "none" not in visualise
                and "filtering" in visualise
                and sim_id is not None
            ):
                # Draw all detections and colour them green if acceptable, red otherwise
                frame_num = f"{i}".zfill(5)
                current_frame_path = (
                    self._root_dir
                    / "frames"
                    / f"sim_{sim_id}"
                    / f"frame{frame_num}.jpg"
                )
                curr_frame = cv.imread(str(current_frame_path), 1)
                for d in filtered_dets[i]:
                    cv.circle(
                        curr_frame, (int(d[0]), int(d[1])), int(d[2]), (0, 255, 0), 2
                    )
                for d in [
                    det for det in frame_detections[i] if det not in filtered_dets[i]
                ]:
                    pt1 = int(d[0] - d[2]), int(d[1] - d[2])
                    pt2 = int(d[0] + d[2]), int(d[1] + d[2])
                    cv.rectangle(curr_frame, pt1, pt2, (0, 0, 255), 2)
                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        label="Accepted Det.",
                        markerfacecolor="none",
                        markersize=15,
                        lw=0,
                        markeredgecolor="g",
                    ),
                    Patch(facecolor="none", edgecolor="r", label="Discarded Det."),
                ]
                plt.imshow(cv.cvtColor(curr_frame, cv.COLOR_BGR2RGB))
                plt.legend(handles=legend_elements)
                plt.axis("off")
                plt.tight_layout()
                plt.show()

        return filtered_dets

    def get_filtered_ball_detections(
        self,
        vid_fname: str,
        morph_op: str,
        morph_op_iters: int,
        morph_op_se_shape: Tuple[int, int],
        struc_el_shape: np.ndarray,
        blur_kernel_size: Tuple[int, int],
        blur_sigma: int,
        binary_thresh: int,
        min_ball_travel_dist: float,
        max_ball_travel_dist: float,
        min_det_area: float,
        max_det_area: float,
        *,
        disable_progbar: bool = False,
        visualise=None,
        sim_id: int,
    ) -> np.ndarray:
        """
        Returns a single detection per frame by filtering all detections in each frame by candidate size and speed
        """

        # Get all ball detection candidates
        if visualise is None:
            visualise = ["none"]
        all_detections = self.get_ball_detections(
            vid_fname=vid_fname,
            morph_op=morph_op,
            morph_op_iters=morph_op_iters,
            morph_op_se_shape=morph_op_se_shape,
            struc_el=struc_el_shape,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            binary_thresh=binary_thresh,
            disable_progbar=disable_progbar,
            visualise=visualise,
        )

        init_ball_pos = get_init_ball_pos(self._vid_dir, vid_fname)

        # Filter detections using the user provided initial ball position
        filtered_dets = self._filter_ball_detections(
            sim_id=sim_id,
            frame_detections=all_detections,
            init_ball_pos=init_ball_pos,
            min_ball_travel_dist=min_ball_travel_dist,
            max_ball_travel_dist=max_ball_travel_dist,
            min_det_area=min_det_area,
            max_det_area=max_det_area,
            disable_progbar=disable_progbar,
            visualise=visualise,
        )

        # Temporary solution until KF is used to filter candidate detections:
        # Arbitrarily select first detection in frame detections if more than one detection present.
        # This is in order to get one detection per frame to form the detections_IC for the KF.
        return np.array([detection[0] for detection in filtered_dets])
