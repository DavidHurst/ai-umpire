from itertools import product, permutations
from math import ceil, floor, prod
from typing import Dict, List

import numpy as np

__all__ = ["TrajectoryInterpreter"]

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ai_umpire import KalmanFilter
from ai_umpire.util import (
    COURT_LENGTH,
    COURT_WIDTH,
    WALL_HEIGHT,
    TIN_HEIGHT,
    gen_grid_of_points,
)

COURT_WALL_HEIGHT = WALL_HEIGHT
HALF_COURT_LENGTH = COURT_LENGTH / 2
HALF_COURT_WIDTH = COURT_WIDTH / 2

court_bbs = {
    "front_wall_bb": {
        "min_x": -HALF_COURT_WIDTH,
        "max_x": HALF_COURT_WIDTH,
        "min_y": TIN_HEIGHT,
        "max_y": COURT_WALL_HEIGHT,
        "min_z": HALF_COURT_LENGTH,
        "max_z": HALF_COURT_LENGTH + 1,
    },
    "test_bb": {
        "min_x": -1.0,
        "max_x": 1.25,
        "min_y": -0.25,
        "max_y": 1.6,
        "min_z": HALF_COURT_LENGTH,
        "max_z": HALF_COURT_LENGTH + 1,
    },
}


class TrajectoryInterpreter:
    def __init__(self, kalman_filter: KalmanFilter):
        self._kf: KalmanFilter = kalman_filter
        self.trajectory: np.ndarray = self._kf.get_trajectory()
        self._n_measurements = self.trajectory.shape[0]
        self._n_variables = self._kf.n_variables()

    def in_out_prob(
        self,
        *,
        n_dim_samples: List,
        n_std_devs_to_sample: int = 1,
    ) -> float:
        mu_list = []
        cov_list = []
        p_trajectory_out: float = 0.0

        # fig, ax = plt.subplots(figsize=(7, 5))
        ax = plt.axes(projection='3d')
        ax.plot3D(
            self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2], "k-", label="Measurements", zdir="y", alpha=0.5
        )
        self._plot_bb("front_wall_bb", ax)
        legend_shown = 0
        for i in range(self._n_measurements):
            mu, cov = self._kf.step()
            mu_list.append(mu)
            cov_list.append(cov)

            std_dev_x = np.sqrt(cov[0, 0])
            std_dev_y = np.sqrt(cov[1, 1])
            std_dev_z = np.sqrt(cov[2, 2])
            sampling_area_size: List = [
                std_dev_x * n_std_devs_to_sample,
                std_dev_y * n_std_devs_to_sample,
                std_dev_z * n_std_devs_to_sample,
            ]

            # Generate sample points
            sample_points = gen_grid_of_points(mu, n_dim_samples, sampling_area_size)
            sample_points_probs = [
                self._kf.prob_of_point(np.reshape(p, (self._n_variables, 1))) for p in sample_points
            ]
            sample_points_weighted_probs = [
                int(self._point_bb_collided(p, "test_bb"))
                * self._kf.prob_of_point(np.reshape(p, (self._n_variables, 1)))
                for p in sample_points
            ]

            # Accumulate probabilities
            weighted_summed_p_samples = sum(sample_points_weighted_probs)
            summed_p_samples = sum(sample_points_probs)
            p_measurement_out = weighted_summed_p_samples / summed_p_samples

            p_trajectory_out = max(p_measurement_out, p_trajectory_out)

            print(f"p(InOut(m))={p_trajectory_out}")

            # Plot
            if legend_shown == 0:
                ax.plot3D(mu[0], mu[1], mu[2], "b^-", alpha=0.5, label="Mean", zdir="y")
                legend_shown = 1
            else:
                ax.plot3D(mu[0], mu[1], mu[2], "b^-", alpha=0.5, zdir="y")

            # Plot sample point probabilities
            # for j in range(sample_points.shape[0]):
            #     ax.text(
            #         sample_points[j][0],
            #         sample_points[j][1],
            #         f"{sample_points_weighted_probs[j]:.2f}",
            #     )

            # Plot in out prob of measurement
            # ax.text(
            #     mu[0].item(),
            #     mu[1].item(),
            #     mu[2].item(),
            #     f"{p_measurement_out:.2f}",
            #     zdir="y"
            # )

            # if legend_shown == 0:
            #     ax.plot(
            #         sample_points[:, 0],
            #         sample_points[:, 1],
            #         "r+",
            #         label="Sampled Points",
            #         alpha=0.5,
            #         markersize=10,
            #     )
            #     legend_shown = 1
            # else:



        title = (
            "$\sum_0^m \sum_0^i InOut(p) \\times p(x_i; \mu, \Sigma)$"
            + f"= {p_trajectory_out:.5f}"
            + f"\n$m$ = {self._n_measurements} (# measurements), $i$ = {prod(n_dim_samples)} (# sample points)"
        )
        # title = f"$p(inOut(T)) = {p_trajectory_out:.5f}$"
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
        return p_trajectory_out

    def _point_bb_collided(self, point: np.ndarray, bb_name: str) -> bool:
        """In 2D for now, will change to 3D later."""
        bb = court_bbs[bb_name]
        return (bb["min_x"] <= point[0].item() <= bb["max_x"]) and (
            bb["min_y"] <= point[1].item() <= bb["max_y"]
        )

    def _plot_bb(self, bb_name: str, ax: plt.axis) -> None:
        bb = court_bbs[bb_name]
        xs = []
        ys = []
        zs = []
        for x in [bb["min_x"], bb["max_x"]]:
            for y in [bb["min_y"], bb["max_y"]]:
                for z in [bb["min_z"], bb["max_z"]]:
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

        ax.scatter3D(xs, ys, zs, zdir="y", label=bb_name)
