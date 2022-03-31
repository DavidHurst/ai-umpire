from itertools import product, permutations
from math import ceil, floor, prod
from typing import Dict

import numpy as np

__all__ = ["TrajectoryInterpreter"]

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
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

    def in_out_prob(
        self,
        n_dim_samples: list,
        sampling_area_size: list,
    ) -> float:
        mu_list = []
        cov_list = []
        p_trajectory_out: float = 0.0

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            self.trajectory[:, 0], self.trajectory[:, 1], "ko-", label="Measurements"
        )
        self.plot_bb(court_bbs["test_bb"], ax)
        m = 10
        legend_shown = 0
        for i in range(m):  # self._n_measurements:
            mu, cov = self._kf.step()
            mu_list.append(mu)
            cov_list.append(cov)

            # Generate sample points
            sample_points = gen_grid_of_points(mu, n_dim_samples, sampling_area_size)
            sample_points_probs = [
                self._kf.prob_of_point(np.reshape(p, (2, 1))) for p in sample_points
            ]
            sample_points_weighted_probs = [
                int(self.point_bb_collided(p, "test_bb"))
                * self._kf.prob_of_point(np.reshape(p, (2, 1)))
                for p in sample_points
            ]

            # Accumulate probabilities
            weighted_summed_p_samples = sum(sample_points_weighted_probs)
            summed_p_samples = sum(sample_points_probs)
            p_measurement_out = weighted_summed_p_samples / summed_p_samples

            p_trajectory_out += p_measurement_out / m

            # Plot
            rad_x = np.sqrt(cov[0, 0])  # Standard deviation of x
            rad_y = np.sqrt(cov[1, 1])  # # Standard deviation of y

            if legend_shown == 0:
                ax.plot(mu[0], mu[1], "bo", markersize=10, alpha=0.5, label="Mean")
                legend_shown = 1
            else:
                ax.plot(mu[0], mu[1], "bo", markersize=10, alpha=0.5)

            # Plot sample point probabilities
            for j in range(sample_points.shape[0]):
                ax.text(
                    sample_points[j][0],
                    sample_points[j][1],
                    f"{sample_points_weighted_probs[j]:.2f}",
                )

            if legend_shown == 0:
                ax.plot(
                    sample_points[:, 0],
                    sample_points[:, 1],
                    "r+",
                    label="Sampled Points",
                    alpha=0.5,
                    markersize=10,
                )
                legend_shown = 1
            else:
                ax.plot(
                    sample_points[:, 0],
                    sample_points[:, 1],
                    "r+",
                    alpha=0.5,
                    markersize=10,
                )
            # std_1 = Ellipse(
            #     (mu[0], mu[1]),
            #     width=rad_x,
            #     height=rad_y,
            #     fill=False,
            #     linestyle="-",
            #     edgecolor="blue",
            #     alpha=0.5,
            #     lw=2.0,
            #     label="$\sigma$",
            # )
            # std_2 = Ellipse(
            #     (mu[0], mu[1]),
            #     width=rad_x * 2,
            #     height=rad_y * 2,
            #     fill=False,
            #     linestyle="--",
            #     lw=2.0,
            #     edgecolor="fuchsia",
            #     alpha=0.5,
            #     label="$2\sigma$",
            # )
            # ax.add_patch(std_1)
            # ax.add_patch(std_2)

        # title = (
        #     "$\sum_0^m \sum_0^i InOut(p) \\times p(x_i; \mu, \Sigma)$"
        #     + f"= {p_trajectory_out:.5f}"
        #     + f"\n$m$ = {m} (# measurements), $i$ = {prod(n_dim_samples)} (# sample points)"
        # )
        title = f"$p(inOut(T)) = {p_trajectory_out:.5f}$"
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
        return p_trajectory_out

    # def visualise(self) -> None:
    #     """Visualise estimated trajectory in 3D with confidence around ball position"""
    #     self.trajectory[:, [1, 2]] = self.trajectory[:, [2, 1]]
    #
    #     fig = plt.figure(figsize=(10, 7))
    #     ax = fig.add_subplot(111, projection=Axes3D.name)
    #     ax.set(xlabel="X", ylabel="Z", zlabel="Y")
    #     ax.view_init(15, -155)
    #
    #     ax.set_xlim(-(HALF_COURT_WIDTH + 1), HALF_COURT_WIDTH + 1)
    #     # Swap y and z for visualisation
    #     ax.set_zlim([0, COURT_WALL_HEIGHT + 1])
    #     ax.set_ylim(-(HALF_COURT_LENGTH + 1), HALF_COURT_LENGTH + 1)
    #
    #     # Exaggerate trajectory
    #     self.trajectory[:, 1] = self.trajectory[:, 1] + 1
    #
    #     x = self.trajectory[:, 0]
    #     y = self.trajectory[:, 1]
    #     z = self.trajectory[:, 2]
    #
    #     # Plot ball trajectory
    #     ax.plot3D(x, y, z, "blue", label="Ball Trajectory")
    #
    #     plane_verts_x_y = np.array(
    #         [
    #             (x, y)
    #             for x in [front_wall_bb["max_x"], front_wall_bb["min_x"]]
    #             for y in [front_wall_bb["min_y"], front_wall_bb["max_y"]]
    #         ]
    #     )
    #     bb_plane = np.c_[plane_verts_x_y, np.ones((4,)) * front_wall_bb["min_z"]]
    #
    #     temp = bb_plane[0].copy()
    #     bb_plane[0] = bb_plane[1]
    #     bb_plane[1] = temp
    #     ax.add_collection3d(
    #         Poly3DCollection(
    #             [list(zip(bb_plane[:, 0], bb_plane[:, 2], bb_plane[:, 1]))],
    #             color="orange",
    #             alpha=0.3,
    #             linewidths=(0,),
    #         )
    #     )
    #
    #     # Detect collision(s)
    #     collisions = []
    #     for point in self.trajectory:
    #         collision = (
    #             (front_wall_bb["min_x"] <= point[0] <= front_wall_bb["max_x"])
    #             and (front_wall_bb["min_y"] <= point[2] <= front_wall_bb["max_y"])
    #             and (front_wall_bb["min_z"] <= point[1] <= front_wall_bb["max_z"])
    #         )
    #         if collision:
    #             # print(f"Collision detected")
    #             collisions.append((point[0], point[1], point[2]))
    #     collisions = np.array(collisions)
    #     ax.scatter3D(
    #         collisions[:, 0],
    #         collisions[:, 1],
    #         collisions[:, 2],
    #         label="Collision",
    #         marker="x",
    #         color="red",
    #         s=100,
    #     )
    #
    #     ax.legend()
    #     ax.grid(False)
    #     plt.show()

    def point_bb_collided(self, point: np.ndarray, bounding_box: str) -> bool:
        """In 2D for now, will change to 3D later."""
        bb = court_bbs[bounding_box]
        return (bb["min_x"] <= point[0].item() <= bb["max_x"]) and (
            bb["min_y"] <= point[1].item() <= bb["max_y"]
        )

    def plot_bb(self, bb: Dict, ax: plt.axis) -> None:
        height = bb["max_y"] - bb["min_y"]
        width = bb["max_x"] - bb["min_x"]
        rect = plt.Rectangle(
            (bb["min_x"], bb["min_y"]), width, height, fill=False, label="BB"
        )
        ax.add_patch(rect)
