from typing import List

import numpy as np

__all__ = ["TrajectoryInterpreter"]

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import patch_2d_to_3d, pathpatch_2d_to_3d, Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import transform
from scipy.spatial import Delaunay

from ai_umpire import KalmanFilter
from ai_umpire.util import (
    FIELD_BOUNDING_BOXES as court_BBs,
    gen_grid_of_points,
)
from ai_umpire.util.field_constants import HALF_COURT_WIDTH, BB_DEPTH

plt.rcParams["figure.figsize"] = (12, 10)


class TrajectoryInterpreter:
    def __init__(self, *, kalman_filter: KalmanFilter):
        self._kf: KalmanFilter = kalman_filter
        self._trajectory: np.ndarray = self._kf.get_trajectory()
        self._n_measurements = self._trajectory.shape[0]
        self._n_variables = self._kf.n_variables()
        self._init_vis()

    def _init_vis(self) -> None:
        # title = (
        #         "$\sum_0^m \sum_0^i InOut(p) \\times p(x_i; \mu, \Sigma)$"
        #         + f"= {p_trajectory_out:.5f}"
        #         + f"\n$m$ = {self._n_measurements} (# measurements), $i$ = {prod(n_dim_samples)} (# sample points)"
        # )
        # title = f"$p(inOut(T)) = {p_trajectory_out:.5f}$"

        fig = plt.figure()
        ax = Axes3D(fig, elev=15, azim=-175)

        plt.gca().grid(False)
        plt.gca().set_xlim3d(-4, 4)
        plt.gca().set_zlim3d(0, 7)
        plt.gca().set_ylim3d(-6, 6)
        plt.gca().set_xlabel("$x$")
        plt.gca().set_zlabel("$y$")
        plt.gca().set_ylabel("$z$")
        # plt.gca().set_title(title)

        # Plot measurements
        plt.gca().plot3D(
            self._trajectory[:, 0],
            self._trajectory[:, 1],
            self._trajectory[:, 2],
            "r-",
            label="Measurements",
            alpha=0.5,
            zdir="y",
        )

        # Plot bounding boxes corresponding to court walls and out-of-court regions
        for bb_name in court_BBs.keys():
            if court_BBs[bb_name]["in_out"] == "out":
                self._plot_bb(bb_name)

    def in_out_prob(
        self,
        *,
        n_dim_samples: List,
        n_std_devs_to_sample: int = 1,
    ) -> float:
        mu_list = []
        cov_list = []
        p_trajectory_out: float = 0.0

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
                self._kf.prob_of_point(np.reshape(p, (self._n_variables, 1)))
                for p in sample_points
            ]
            sample_points_weighted_probs = [
                int(self._point_bb_collided(p, "right_wall_in_top"))
                * self._kf.prob_of_point(np.reshape(p, (self._n_variables, 1)))
                for p in sample_points
            ]

            # Accumulate probabilities
            weighted_summed_p_samples = sum(sample_points_weighted_probs)
            summed_p_samples = sum(sample_points_probs)
            p_measurement_out = weighted_summed_p_samples / summed_p_samples

            p_trajectory_out = max(p_measurement_out, p_trajectory_out)

            # Plot
            if legend_shown == 0:
                plt.gca().plot3D(
                    mu[0], mu[1], mu[2], "b^-", alpha=0.5, label="Mean", zdir="y"
                )
                legend_shown = 1
            else:
                plt.gca().plot3D(mu[0], mu[1], mu[2], "b^-", alpha=0.5, zdir="y")

            # Plot sample point probabilities
            # for j in range(sample_points.shape[0]):
            #     plt.gca().text(
            #         sample_points[j][0],
            #         sample_points[j][1],
            #         f"{sample_points_weighted_probs[j]:.2f}",
            #     )

            # Plot in out prob of measurement
            # if p_measurement_out > 0.09:
            #     plt.gca().text(
            #         mu[0].item(),
            #         mu[2].item(),
            #         mu[1].item(),
            #         f"{p_measurement_out:.3f}",
            #         bbox=dict(facecolor='white', alpha=0.8)
            #     )
            #
            # # Plot sample points
            # plt.gca().plot3D(
            #     sample_points[:, 0],
            #     sample_points[:, 1],
            #     sample_points[:, 2],
            #     "r+",
            #     label="Sampled Points",
            #     alpha=0.5,
            #     zdir="y"
            # )

        plt.gca().legend()
        plt.show()
        return p_trajectory_out

    def _point_bb_collided(self, point: np.ndarray, bb_name: str) -> bool:
        if point.shape[0] != 3:
            raise ValueError("Expecting a 3D point.")

        bb = court_BBs[bb_name]
        if bb_name in [
            "right_wall_out_bottom",
            "left_wall_out_bottom",
            "right_wall_in_top",
            "left_wall_in_top",
        ]:
            poly = bb["verts"]
            return Delaunay(poly).find_simplex(point) >= 0

        return (
            (bb["min_x"] <= point[0].item() <= bb["max_x"])
            and (bb["min_y"] <= point[1].item() <= bb["max_y"])
            and (bb["min_z"] <= point[2].item() <= bb["max_z"])
        )

    def _plot_bb(self, bb_name: str) -> None:
        if bb_name in [
            "right_wall_out_bottom",
            "left_wall_out_bottom",
            "right_wall_in_top",
            "left_wall_in_top",
        ]:
            bb = court_BBs[bb_name]
            verts = bb["verts"]
            plt.gca().scatter3D(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                zdir="y",
                color=bb["colour"],
            )
            if bb_name.startswith("right"):
                plane_verts = verts[~np.any(verts == HALF_COURT_WIDTH + BB_DEPTH, axis=1)]
            else:
                plane_verts = verts[~np.any(verts == -HALF_COURT_WIDTH - BB_DEPTH, axis=1)]
            plane_verts = [list(zip(plane_verts[:, 0], plane_verts[:, 2], plane_verts[:, 1]))]
            plt.gca().add_collection3d(Poly3DCollection(plane_verts, color=bb["colour"], alpha=0.5, lw=0.1))
            return

        bb = court_BBs[bb_name]
        verts = []
        for x in [bb["min_x"], bb["max_x"]]:
            for y in [bb["min_y"], bb["max_y"]]:
                for z in [bb["min_z"], bb["max_z"]]:
                    verts.append([x, y, z])

        verts = np.array(verts)

        # ToDo: Refactor below to use Poly3DCollections to plot desired face.
        xs = [bb["min_x"], bb["max_x"]]
        ys = [bb["min_y"], bb["max_y"]]
        z = bb["min_z"]

        if bb_name == "back_wall" or bb_name == "back_wall_out":
            z = bb["max_z"]
        if bb_name == "left_wall_in_bottom" or bb_name == "left_wall_out_top":
            xs = [bb["min_z"], bb["max_z"]]
            X, Y = np.meshgrid(xs, ys)
            plt.gca().plot_surface(
                bb["max_x"], X, Y, alpha=0.5, facecolors=bb["colour"]
                , lw=0.1
            )
        elif bb_name == "right_wall_in_bottom" or bb_name == "right_wall_out_top":
            xs = [bb["min_z"], bb["max_z"]]
            X, Y = np.meshgrid(xs, ys)
            plt.gca().plot_surface(
                bb["min_x"], X, Y, alpha=0.5, facecolors=bb["colour"]
                , lw=0.1
            )
        else:
            X, Y = np.meshgrid(xs, ys)
            plt.gca().plot_surface(X, z, Y, alpha=0.5, facecolors=bb["colour"], lw=0.1)


        plt.gca().scatter3D(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            zdir="y",
            color=bb["colour"],
        )
