__all__ = ["BallDetector"]

import math
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from ai_umpire.util import (
    extract_frames_from_vid,
    difference_frames,
    blur_frames,
    binarize_frames,
    apply_morph_op,
)


class BallDetector:
    def __init__(self, root_dir: Path):
        self._root_dir: Path = root_dir
        self._vid_dir: Path = self._root_dir / "videos"
        self._frames_dir: Path = self._root_dir / "frames"
        self._all_detections = []

    # def _localise_ball_blob_filter(self, frames: np.ndarray) -> np.ndarray:
    #     detections: List[List] = []
    #
    #     # Setup SimpleBlobDetector parameters.
    #     params = cv.SimpleBlobDetector_Params()
    #
    #     params.filterByColor = False
    #
    #     # Change thresholds
    #     params.minThreshold = 0
    #     params.maxThreshold = 150
    #
    #     # Filter by Area.
    #     # params.filterByArea = True
    #     # params.minArea = 1500
    #
    #     # Filter by Circularity, a circle has a circularity of 1
    #     params.filterByCircularity = False
    #     params.minCircularity = 0
    #     params.maxCircularity = 1
    #
    #     # Filter by Convexity
    #     params.filterByConvexity = True
    #     params.minConvexity = 0.9
    #
    #     # Filter by Inertia, higher value means search for more elongation
    #     params.filterByInertia = True
    #     params.minInertiaRatio = 0
    #     params.maxInertiaRatio = 1
    #
    #     # Create a detector with the parameters
    #     detector = cv.SimpleBlobDetector_create(params)
    #
    #     for i in tqdm(
    #         range(frames.shape[0]),
    #         desc="Localising ball (blob filter)",
    #     ):
    #         # Detect blobs
    #         keypoints = detector.detect(frames[i])
    #
    #         # # Return first blob detected as detection (temporary)
    #         if keypoints:
    #             for kp in keypoints:
    #                 detections.append([kp.pt[1], kp.pt[0], kp.size])
    #                 break
    #         else:
    #             # No detections made
    #             detections.append([10, 10, 5])
    #
    #     return np.array(detections)

    # def _localise_ball_hough_circle(self, frames: np.ndarray) -> np.ndarray:
    #     detections: List[List] = []
    #     for i in tqdm(
    #         range(frames.shape[0]),
    #         desc="Localising ball (Hough circle)",
    #     ):
    #         # Detect circles
    #         max_radius: int = int(frames[i].shape[1] * 0.4)
    #         circles_detected = cv.HoughCircles(
    #             image=frames[i],
    #             method=cv.HOUGH_GRADIENT,
    #             dp=1,
    #             minDist=20.0,
    #             param1=50,
    #             param2=30,
    #             minRadius=0,
    #             maxRadius=0,
    #         )
    #
    #         if circles_detected is not None:
    #             for x, y, r in circles_detected[0]:
    #                 print(f"Frame #{i}: (x={x},y={y}), rad={r}")
    #         else:
    #             print("No detections")
    #
    #         # Return first detection (temporary)
    #         if circles_detected is not None:
    #             for x, y, r in circles_detected[0]:
    #                 detections.append([y, x, r])
    #                 break
    #         else:
    #             # No detections made
    #             detections.append([10, 10, 5])
    #
    #     return np.array(detections)
    #
    # def _localise_ball_blob(self, frames: np.ndarray, method: str) -> np.ndarray:
    #     detections: List[List] = []
    #     for i in tqdm(
    #         range(frames.shape[0]),
    #         desc=f"Localising ball ({method})",
    #     ):
    #         method_types: List[str] = ["log", "dog", "doh"]
    #         if method not in method_types:
    #             e: ValueError = ValueError(
    #                 f"Invalid method/method not supported, available options: {method_types}"
    #             )
    #             logging.exception(e)
    #             raise e
    #         logging.info(f"Detecting blobs using {method}.")
    #
    #         frame = frames[i]
    #
    #         # Detect blobs- can give kernel std devs as sequence per axis maybe to elongate blobs?
    #         blobs: np.ndarray = None
    #         if method == "log":
    #             blobs: np.ndarray = blob_log(
    #                 frame,
    #                 min_sigma=10,
    #                 max_sigma=50,
    #                 num_sigma=5,
    #                 threshold=0.0001,
    #             )
    #         elif method == "dog":
    #             blobs = blob_dog(
    #                 frame,
    #                 min_sigma=5,
    #                 max_sigma=10,
    #                 sigma_ratio=1.6,
    #                 threshold=0.0001,
    #             )
    #         elif method == "doh":
    #             blobs: np.ndarray = blob_doh(
    #                 frame,
    #                 min_sigma=1,
    #                 max_sigma=20,
    #                 num_sigma=1,
    #                 threshold=0.005,
    #             )
    #
    #         # Compute radii in the third column
    #         if method == "log" or "dog":
    #             blobs[:, 2] = blobs[:, 2] * sqrt(2)
    #         for blob in blobs:
    #             y, x, r = blob
    #             print(f"Frame #{i}: (x={x},y={y}), rad={r}")
    #
    #         # Return first blob detected as detection (temporary)
    #         for blob in blobs:
    #             y, x, r = blob
    #             detections.append([y, x, r])
    #             break
    #
    #     return np.array(detections)

    def get_ball_detections(
        self,
        vid_fname: str,
        morph_op: str,
        morph_op_iters: int,
        morph_op_se_shape: Tuple[int, int],
        struc_el_shape: np.ndarray,
        blur_kernel_size: Tuple[int, int],
        blur_sigma: int,
        binary_thresh: int,
        *,
        disable_progbar: bool = False,
        sim_id: int,
    ) -> List[List]:
        detections: List[List] = []
        # Extract frames from video
        vid_path: Path = self._vid_dir / vid_fname
        video_frames: np.ndarray = extract_frames_from_vid(
            vid_path, disable_progbar=disable_progbar
        )

        # Preprocess frames
        blurred_frames: np.ndarray = blur_frames(
            video_frames,
            blur_kernel_size,
            blur_sigma,
            disable_progbar=disable_progbar,
        )
        fg_seg_frames: np.ndarray = difference_frames(
            blurred_frames, disable_progbar=disable_progbar
        )
        binary_frames: np.ndarray = binarize_frames(
            fg_seg_frames, binary_thresh, disable_progbar=disable_progbar
        )
        morph_op_frames: np.ndarray = apply_morph_op(
            binary_frames,
            morph_op,
            morph_op_iters,
            morph_op_se_shape,
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
                morph_op_frames[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
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

                # display_im = cv.imread(
                #     str(
                #         self._frames_dir
                #         / f"sim_{sim_id}"
                #         / f"frame{str(i).zfill(5)}.jpg"
                #     )
                # )
                # cv.drawContours(display_im, contours, -1, (0, 0, 255), 2)
                # # cv.imshow(f"Frame #{i} - Contours", display_im)
                # # cv.imshow(f"Frame #{i} - Differenced", fg_seg_frames[i])
                # # cv.imshow(f"Frame #{i} Features", morph_op_frames[i])
                # # cv.waitKey(0)
                # fig, axes = plt.subplots(1, 4, figsize=(15, 3))
                # axes[0].imshow(cv.cvtColor(fg_seg_frames[i], cv.COLOR_BGR2RGB))  # , cmap="gray", vmin=0, vmax=255)
                # axes[1].imshow(binary_frames[i], cmap="gray", vmin=0, vmax=1)
                # axes[2].imshow(morph_op_frames[i], cmap="gray", vmin=0, vmax=255)
                # axes[3].imshow(
                #     cv.cvtColor(display_im, cv.COLOR_BGR2RGB)
                # )
                # titles = ["Diff", "Binary", "Morph. Close", "Contours"]
                # for k, ax in enumerate(axes):
                #     ax.axis("off")
                #     ax.set_title(titles[k])

                # if i == 3:
                #     cv.imwrite(f'./real_vid_frame_{i}.jpg', video_frames[i])
                #
                # plt.imshow(cv.cvtColor(video_frames[i], cv.COLOR_BGR2RGB))
                # plt.axis("off")
                # plt.tight_layout()
                # # plt.savefig(f"detection{str(i).zfill(2)}.png")
                # plt.show()
            else:
                # No detections, worst values, position miles away from anywhere on the screen
                # and area infinitely large when ball should be small
                print(f"No detections frame #{i}")
                detections.append([(-1, -1, -1)])

        # ToDo: convert to numpy array
        self._all_detections = detections
        return detections

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
        init_ball_pos: Tuple[float, float],
        min_ball_travel_dist: float,
        max_ball_travel_dist: float,
        min_det_area: float,
        max_det_area: float,
        *,
        disable_progbar: bool = False,
        sim_id: int,
    ) -> np.ndarray:

        self.get_ball_detections(
            vid_fname=vid_fname,
            morph_op=morph_op,
            morph_op_iters=morph_op_iters,
            morph_op_se_shape=morph_op_se_shape,
            struc_el_shape=struc_el_shape,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            binary_thresh=binary_thresh,
            disable_progbar=disable_progbar,
            sim_id=sim_id,
        )

        filtered_dets = self._filter_ball_detections(
            sim_id=sim_id,
            frame_detections=self._all_detections,
            init_ball_pos=init_ball_pos,
            min_ball_travel_dist=min_ball_travel_dist,
            max_ball_travel_dist=max_ball_travel_dist,
            min_det_area=min_det_area,
            max_det_area=max_det_area,
            disable_progbar=disable_progbar,
        )

        # Temporary solution until KF is used to filter candidate detections:
        # Arbitrarily select first detection in frame detections if more than one detection present.
        # This is in order to get one detection per frame to form the detections_IC for the KF.
        return np.array([detection[0] for detection in filtered_dets])

    def _filter_ball_detections(
        self,
        frame_detections: List[List],
        init_ball_pos: Tuple[float, float],
        sim_id: int,
        *,
        min_ball_travel_dist: float = 5,
        max_ball_travel_dist: float = 130,
        min_det_area: float = 2.0,
        max_det_area: float = 65.0,
        disable_progbar: bool = False,
    ) -> List[List]:
        filtered_dets = []

        def get_frame_detections_com(frame_num: int) -> Tuple[float, float]:
            """com = Center of Mass"""
            prev_frame_accepted_dets_x = [x for x, _, _ in filtered_dets[frame_num]]
            prev_frame_accepted_dets_y = [y for _, y, _ in filtered_dets[frame_num]]
            com_x = sum(prev_frame_accepted_dets_x) / len(prev_frame_accepted_dets_x)
            com_y = sum(prev_frame_accepted_dets_y) / len(prev_frame_accepted_dets_y)

            return com_x, com_y

        for i in tqdm(
            range(len(frame_detections)),
            desc=f"Filtering ball detections",
            disable=disable_progbar,
        ):
            # print(f"Frame #{i}", "-" * 40)
            # if len(filtered_dets) > 0:
            #     print(f"Accumulated filtered dets:")
            #     for d in filtered_dets:
            #         if len(d) > 1:
            #             print(" " * 8, f"{len(d)} dets below:")
            #             for d_ in d:
            #                 print(" " * 8, d_)
            #         else:
            #             print(" " * 4, d)
            dets = frame_detections[i]

            # Filter detections base on their size, i.e. filter out the player detections and noise
            dets = [(x, y, z) for x, y, z in dets if min_det_area < z < max_det_area]

            # print(
            #     f">> Num dets filtered by size = {len(frame_detections[i]) - len(dets)}"
            # )
            # ROOT_DIR_PATH: Path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
            # frame_num = f"{i}".zfill(5)
            # frame_path = (
            #     ROOT_DIR_PATH / "frames" / f"sim_{sim_id}" / f"frame{frame_num}.jpg"
            # )
            # frame = cv.imread(str(frame_path), 1)

            if i > 0:
                velocity_constrained_dets = []
                for curr_frame_det in dets:
                    for prev_frame_det in filtered_dets[i - 1]:

                        curr_x, curr_y = curr_frame_det[0], curr_frame_det[1]
                        prev_x, prev_y = prev_frame_det[0], prev_frame_det[1]

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
                            # cv.circle(frame, (curr_x, curr_y), 10, (0, 255, 0))
                            velocity_constrained_dets.append(curr_frame_det)
                            # print(f">> Added {curr_frame_det}")
                        # else:
                        # cv.circle(frame, (curr_x, curr_y), 10, (0, 0, 255))
                if len(velocity_constrained_dets) == 0:
                    # If no detections satisfy the velocity constraint, add the detection closest to the center of mass of
                    # the previous frame's acceptable detections
                    # print(
                    #     "[i] No dets added, added det closest to center of mass (com) of prev acceptable dets."
                    # )

                    com_x, com_y = get_frame_detections_com(frame_num=i - 1)

                    dets_dist_to_com = [
                        math.sqrt(((x - com_x) ** 2) + ((y - com_y) ** 2))
                        for x, y, _ in dets
                    ]
                    closest_det = dets[dets_dist_to_com.index(min(dets_dist_to_com))]
                    filtered_dets.append([closest_det])

                    # print(f">> Added {closest_det}")

                    # frame_ = frame.copy()
                    # cv.circle(frame_, (int(com_x), int(com_y)), 5, (0, 0, 255), -1)
                    # cv.circle(frame_, (closest_det[0], closest_det[1]), 5, (0, 255, 0))
                    # cv.imshow(f"Frame #{i} - Center of mass & closest det to com", frame_)
                    # cv.waitKey(0)
                else:
                    filtered_dets.append(list(set(velocity_constrained_dets)))
            else:
                # Find the closest detection to manually initialised initial ball position
                dists = [
                    math.sqrt(
                        ((x - init_ball_pos[0]) ** 2) + ((y - init_ball_pos[1]) ** 2)
                    )
                    for x, y, _ in dets
                ]
                closest_det = dets[dists.index(min(dists))]

                # for det in dets:
                #     if det == closest_det:
                #         # cv.circle(frame, (det[0], det[1]), 10, (0, 255, 0))
                #     else:
                #         # cv.circle(frame, (det[0], det[1]), 10, (0, 0, 255))
                # print(f">> First frame, adding closest det -> {closest_det}")

                filtered_dets.append([closest_det])

            # cv.imshow(f"Frame #{i}", frame)
            # cv.waitKey(0)

        return filtered_dets

    # def get_ball_candidates(
    #     self,
    #     vid_path: Path,
    #     morph_op: str,
    #     detection_method: str,
    #     morph_op_iters: int,
    #     morph_op_SE_shape: Tuple[int, int],
    #     blur_kernel_size: Tuple[int, int],
    #     blur_sigma_x: int,
    #     binary_thresh_low: int,
    # ) -> np.ndarray:
    #     # Extract frames from video
    #     video_frames: np.ndarray = extract_frames_from_vid(vid_path)
    #
    #     # Preprocess frames
    #     fg_seg_frames: np.ndarray = difference_frames(video_frames)
    #     blurred_frames: np.ndarray = blur_frames(
    #         fg_seg_frames, blur_kernel_size, blur_sigma_x
    #     )
    #     binary_frames: np.ndarray = binarize_frames(blurred_frames, binary_thresh_low)
    #     morph_op_frames: np.ndarray = apply_morph_op(
    #         binary_frames, morph_op, morph_op_iters, morph_op_SE_shape
    #     )
    #
    #     # Detect ball in processed frames
    #     if detection_method == "log" or detection_method == "dog":
    #         detections: np.ndarray = self._localise_ball_blob(
    #             morph_op_frames, detection_method
    #         )
    #     elif detection_method == "blob_filter":
    #         detections: np.ndarray = self._localise_ball_blob_filter(morph_op_frames)
    #     elif detection_method == "hough_circle":
    #         detections: np.ndarray = self._localise_ball_hough_circle(morph_op_frames)
    #     elif detection_method == "contour":
    #         detections: np.ndarray = self._localise_ball_contour(morph_op_frames)
    #     else:
    #         e: ValueError = ValueError("Invalid detection method chosen.")
    #         logging.exception(e)
    #         raise e
    #
    #     # self._display_detections(
    #     #     [
    #     #         video_frames,
    #     #         fg_seg_frames,
    #     #         blurred_frames,
    #     #         binary_frames,
    #     #         morph_op_frames,
    #     #         detections,
    #     #     ],
    #     #     morph_op,
    #     #     detection_method,
    #     # )
    #
    #     # Return frame detections
    #     return detections
    #
    # def _display_detections(
    #     self, detection_phases_frames: List[np.ndarray], morph_op: str, det_method: str
    # ):
    #     # Draw detections
    #     for i in range(detection_phases_frames[0].shape[0]):
    #         display_img: np.ndarray = detection_phases_frames[0][i].copy()
    #         # print(
    #         #     f"Frame #{i} = ({detection_phases_frames[-1][i][0]}, {detection_phases_frames[-1][i][1]}), radius={detection_phases_frames[-1][i][2]}"
    #         # )
    #         cv.circle(
    #             display_img,
    #             (
    #                 int(detection_phases_frames[-1][i][1]),
    #                 int(detection_phases_frames[-1][i][0]),
    #             ),
    #             int(detection_phases_frames[-1][i][2]),
    #             (0, 0, 255),
    #             2,
    #         )
    #
    #         fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    #         axs[0, 0].imshow(
    #             cv.cvtColor(detection_phases_frames[0][i], cv.COLOR_BGR2RGB)
    #         )
    #         axs[0, 1].imshow(
    #             cv.cvtColor(detection_phases_frames[1][i], cv.COLOR_BGR2RGB)
    #         )
    #         axs[0, 2].imshow(
    #             cv.cvtColor(detection_phases_frames[2][i], cv.COLOR_BGR2RGB)
    #         )
    #         axs[1, 0].imshow(
    #             detection_phases_frames[3][i], cmap="gray", vmin=0, vmax=255
    #         )
    #         axs[1, 1].imshow(
    #             detection_phases_frames[4][i], cmap="gray", vmin=0, vmax=255
    #         )
    #         axs[1, 2].imshow(cv.cvtColor(display_img, cv.COLOR_BGR2RGB))
    #
    #         axs[0, 0].set_title("1. Original")
    #         axs[0, 1].set_title("2. FG Seg.")
    #         axs[0, 2].set_title("3. FG Seg.+Blur")
    #         axs[1, 0].set_title("4. FG Seg.+Blur+Binarize")
    #         axs[1, 1].set_title(f"5. FG Seg.+Blur+Binarize+Morph. Op.({morph_op})")
    #         axs[1, 2].set_title(f"6. Detection ({det_method})")
    #
    #         axs[0, 0].axis("off")
    #         axs[0, 1].axis("off")
    #         axs[0, 2].axis("off")
    #         axs[1, 0].axis("off")
    #         axs[1, 1].axis("off")
    #         axs[1, 2].axis("off")
    #         plt.tight_layout()
    #         # fig.savefig(f"detection{i}.png")
    #         plt.show()
