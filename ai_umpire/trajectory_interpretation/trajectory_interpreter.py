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
        self.bb_collision_probs: Dict = {
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

        # Plot measurements
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
            bb_collision_prob = self.bb_collision_probs[bb_name][
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

    def interpret_trajectory(
        self,
        *,
        visualise: bool = False,
        save: bool = False,
        show_sample_points: bool = False,
    ) -> Tuple[str, float]:
        """Returns probability of trajectory being out and which out-area it most likely hit"""
        for i in range(self._n_measurements):
            p, bb = self.interpret_next_measurement(
                visualise=visualise, save=save, show_sample_points=show_sample_points
            )
            print(
                f"Measurement #{i}, most likely collision with {bb} with probability {p:.4f}."
            )

        print(self.bb_collision_probs)

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

            self.bb_collision_probs[bb_name].append(collision_prob)

        if save or visualise:
            self._visualise_interpretation(
                mu,
                sample_points=sample_points,
                display=visualise,
                save=save,
                show_sample_points=show_sample_points,
            )

        return self._most_likely_collision()

    def _most_likely_collision(self) -> Tuple[float, str]:
        collision_prob = 0.0
        collision_bb = ""

        for bb_name in FIELD_BOUNDING_BOXES.keys():
            bb_collision_prob = self.bb_collision_probs[bb_name][
                self._kf.get_t_step() - 1
            ]
            if bb_collision_prob > collision_prob:
                collision_prob = bb_collision_prob
                collision_bb = bb_name
        return collision_prob, collision_bb
