from typing import List, Dict, Tuple

import numpy as np

__all__ = ["TrajectoryInterpreter"]

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

from ai_umpire import KalmanFilter
from ai_umpire.util import (
    FIELD_BOUNDING_BOXES,
    gen_grid_of_points,
    plot_bb,
    point_bb_collided,
)

plt.rcParams["figure.figsize"] = (10, 10)


class TrajectoryInterpreter:
    def __init__(
        self,
        *,
        kalman_filter: KalmanFilter,
        n_dim_samples: List = None,
        n_std_devs_to_sample: int = 1,
    ):
        # ToDo: Bring KF init into this constructor, makes more sense
        if n_dim_samples is None:
            n_dim_samples = [5, 5, 5]
        self._kf: KalmanFilter = kalman_filter
        self._trajectory: np.ndarray = self._kf.get_trajectory()
        self._n_measurements = self._trajectory.shape[0]
        self._n_variables = self._kf.get_n_variables()
        if len(n_dim_samples) != self._n_variables:
            raise ValueError(
                "You must provide how many points to sample for each dimension."
            )
        self._dim_samples = n_dim_samples
        self._sample_size_coef = n_std_devs_to_sample

        # Dictionary to store collision probabilities for all bounding boxes after processing each measurement
        # {bb:[p(m_1), p(m_2), ...], ...}
        self._bb_collision_probs: Dict = {
            name: [] for name in FIELD_BOUNDING_BOXES.keys()
        }

    def _visualise_interpretation(
        self,
        mu: np.ndarray,
        *,
        show_sample_points: bool = False,
        sample_points: np.ndarray,
        bbs_to_show: str = "all",
        show_bb_verts: bool = True,
        save: bool = False,
        display: bool = False,
    ) -> None:
        if bbs_to_show not in ["all", "out", "in"]:
            raise ValueError(
                "Options for bbs to show are: ['all', 'out_bbs', 'in_bbs']"
            )
        fig = plt.figure()
        self._ax = Axes3D(fig, elev=20, azim=-140)

        self._ax.grid(False)
        self._ax.set_title(
            f"Probabilistic Interpretation of Measurement #{self._kf.get_t_step()}"
        )
        self._ax.set_xlim3d(-4, 4)
        self._ax.set_zlim3d(0, 7)
        self._ax.set_ylim3d(-6, 6)
        self._ax.set_xlabel("$x$")
        self._ax.set_zlabel("$y$")
        self._ax.set_ylabel("$z$")

        # Plot detections_IC
        self._ax.plot3D(
            self._trajectory[:, 0],
            self._trajectory[:, 1],
            self._trajectory[:, 2],
            "r-",
            label="Measurements",
            alpha=0.5,
            zdir="y",
        )

        # Show KF predicted mean
        self._ax.plot3D(
            mu[0],
            mu[1],
            mu[2],
            "*",
            alpha=0.7,
            label="Mean",
            zdir="y",
            markersize=15,
            zorder=4,
        )

        # Plot sample points
        if show_sample_points:
            self._ax.plot3D(
                sample_points[:, 0],
                sample_points[:, 1],
                sample_points[:, 2],
                "r+",
                label="Sampled Points",
                alpha=0.5,
                zdir="y",
            )

        # Plot bounding boxes corresponding to court walls and out-of-court regions
        for bb_name in FIELD_BOUNDING_BOXES.keys():
            bb_collision_prob = self._bb_collision_probs[bb_name][
                self._kf.get_t_step() - 1
            ]
            if bbs_to_show == "all":
                plot_bb(
                    bb_name=bb_name,
                    ax=self._ax,
                    bb_face_annotation="{:.4f}".format(bb_collision_prob),
                    show_vertices=show_bb_verts,
                    show_annotation=bb_collision_prob > 0.001,
                )
            else:
                if FIELD_BOUNDING_BOXES[bb_name]["in_out"] == bbs_to_show:
                    plot_bb(
                        bb_name=bb_name,
                        ax=self._ax,
                        bb_face_annotation="{:.4f}".format(bb_collision_prob),
                        show_vertices=show_bb_verts,
                        show_annotation=bb_collision_prob > 0.001,
                    )

        self._ax.legend()

        if save:
            plt.savefig(f"measurement_{self._kf.get_t_step() - 1}.png")
        if display:
            plt.show()

    def classify_trajectory(
        self,
        confidence_threshold: float,
        *,
        visualise: bool = False,
        save: bool = False,
        show_sample_points: bool = False,
    ) -> str:
        """
        Returns the in/out classification of the trajectory

        :return: "out" if the trajectory is interpreted as out, "in" otherwise
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be in the range [0, 1].")

        no_probs_recorded = len(list(self._bb_collision_probs.values())[0]) == 0
        if no_probs_recorded:
            p_out, out_bb_name, frame_out = self.interpret_trajectory(
                visualise=visualise, save=save, show_sample_points=show_sample_points
            )
        else:
            p_out, out_bb_name, frame_out = 0.0, "", 0

            # Scan through stored probability detections_IC and keep track of highest prob out, bb name and frame
            for i in tqdm(
                range(self._n_measurements),
                desc="Scanning stored collision probabilities",
            ):
                for bb_name in FIELD_BOUNDING_BOXES.keys():
                    bb_out_prob_frame = self._bb_collision_probs[bb_name][i]
                    if bb_out_prob_frame >= p_out:
                        p_out, out_bb_name, frame_out = bb_out_prob_frame, bb_name, i

        return "out of court" if p_out >= confidence_threshold else "in"

    def interpret_trajectory(
        self,
        *,
        visualise: bool = False,
        save: bool = False,
        show_sample_points: bool = False,
    ) -> Tuple[float, str, int]:
        """
        Returns probability of trajectory being out, which out-area it most likely hit and in which frame.

        :return: Returns the probability the trajectory was out, which out BB it hit to be out and in which frame
        """

        if len(list(self._bb_collision_probs.values())[0]) > 0:
            raise NotImplementedError("Attempted to interpret trajectory twice.")
            # warnings.warn("Warning, trajectory already interpreted, resetting and recalculating probabilities")

        highest_p_out, out_bb_name, out_frame = 0.0, "", 0
        for i in range(self._n_measurements):
            p, bb = self.interpret_next_measurement(
                visualise=visualise, save=save, show_sample_points=show_sample_points
            )
            if p >= highest_p_out and FIELD_BOUNDING_BOXES[bb]["in_out"] == "out":
                highest_p_out, out_bb_name, out_frame = p, bb, i
                print(
                    f"[i] New highest prob, {highest_p_out}, {out_bb_name}, {out_frame}"
                )
            print(
                f"Measurement #{i}: \n    Most likely collision with - {bb} \n    Probability - {p:.4f}"
            )

        return highest_p_out, out_bb_name, out_frame

    def interpret_next_measurement(
        self,
        *,
        visualise: bool = False,
        save: bool = False,
        show_sample_points: bool = False,
    ) -> Tuple[float, str]:
        """Return the probability of a measurement being out of court"""
        mu, cov = self._kf.step()  # KF inference

        std_dev_x = np.sqrt(cov[0, 0])
        std_dev_y = np.sqrt(cov[1, 1])
        std_dev_z = np.sqrt(cov[2, 2])

        # Generate sample area size with dimension sizes scaled by dimension's standard deviation
        sampling_area_size: List = [
            self._sample_size_coef * std_dev_x,
            self._sample_size_coef * std_dev_y,
            self._sample_size_coef * std_dev_z,
        ]

        # Generate grid of sample points
        sample_points = gen_grid_of_points(mu, self._dim_samples, sampling_area_size)

        # Generate sample points' probabilities given KF internal parameters for each bounding box
        sample_points_probs = [
            self._kf.prob_of_point(np.reshape(p, (self._n_variables, 1)))
            for p in sample_points
        ]
        for bb_name in tqdm(
            FIELD_BOUNDING_BOXES.keys(), desc="Calculating collision probabilities"
        ):
            # Calculate prob of collision with bb
            sample_points_weighted_probs = [
                int(point_bb_collided(p, bb_name))
                * self._kf.prob_of_point(np.reshape(p, (self._n_variables, 1)))
                for p in sample_points
            ]

            weighted_summed_p_samples = sum(sample_points_weighted_probs)
            summed_p_samples = sum(sample_points_probs)
            collision_prob = weighted_summed_p_samples / summed_p_samples

            self._bb_collision_probs[bb_name].append(collision_prob)

        if save or visualise:
            self._visualise_interpretation(
                mu,
                sample_points=sample_points,
                display=visualise,
                save=save,
                show_sample_points=show_sample_points,
            )

        return self._most_likely_collision(self._kf.get_t_step() - 1)

    def _most_likely_collision(self, measurement_num: int) -> Tuple[float, str]:
        collision_prob = 0.0
        collision_bb = ""

        for bb_name in FIELD_BOUNDING_BOXES.keys():
            bb_collision_prob = self._bb_collision_probs[bb_name][measurement_num]
            if bb_collision_prob > collision_prob:
                collision_prob = bb_collision_prob
                collision_bb = bb_name

        if collision_bb == "back_wall_out":
            return collision_prob, collision_bb
        else:
            return collision_prob * 2, collision_bb
