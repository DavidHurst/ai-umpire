__all__ = ["BallDetector"]

import math
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
from tqdm import tqdm

from ai_umpire.util import (
    extract_frames_from_vid,
    difference_frames,
    blur_frames,
    binarize_frames,
    apply_morph_op,
)
from ai_umpire.util.util import get_init_ball_pos


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

        init_ball_pos = get_init_ball_pos(self._vid_dir, vid_fname)

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
