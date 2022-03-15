from itertools import combinations, product, permutations, combinations_with_replacement

import numpy as np

__all__ = ["TrajectoryInterpreter"]

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ai_umpire.util import (
    COURT_LENGTH,
    COURT_WIDTH,
    WALL_HEIGHT,
    TIN_HEIGHT,
    WALL_THICKNESS,
)

COURT_WALL_HEIGHT = WALL_HEIGHT
HALF_COURT_LENGTH = COURT_LENGTH / 2
HALF_COURT_WIDTH = COURT_WIDTH / 2

front_wall_bb = {
    "min_x": -HALF_COURT_WIDTH,
    "max_x": HALF_COURT_WIDTH,
    "min_y": TIN_HEIGHT,
    "max_y": COURT_WALL_HEIGHT,
    "min_z": HALF_COURT_LENGTH,
    "max_z": HALF_COURT_LENGTH + 1,
}


class TrajectoryInterpreter:
    def __init__(self, estimated_ball_positions: np.ndarray):
        self.trajectory: np.ndarray = estimated_ball_positions

    def visualise(self) -> None:
        """Visualise estimated trajectory in 3D with confidence around ball position"""
        self.trajectory[:, [1, 2]] = self.trajectory[:, [2, 1]]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection=Axes3D.name)
        ax.set(xlabel="X", ylabel="Z", zlabel="Y")
        ax.view_init(15, -155)

        ax.set_xlim(-(HALF_COURT_WIDTH + 1), HALF_COURT_WIDTH + 1)
        # Swap y and z for visualisation
        ax.set_zlim([0, COURT_WALL_HEIGHT + 1])
        ax.set_ylim(-(HALF_COURT_LENGTH + 1), HALF_COURT_LENGTH + 1)

        # Exaggerate trajectory
        self.trajectory[:, 1] = self.trajectory[:, 1] + 1

        x = self.trajectory[:, 0]
        y = self.trajectory[:, 1]
        z = self.trajectory[:, 2]

        # Plot ball trajectory
        ax.plot3D(x, y, z, "blue", label="Ball Trajectory")

        plane_verts_x_y = np.array(
            [
                (x, y)
                for x in [front_wall_bb["max_x"], front_wall_bb["min_x"]]
                for y in [front_wall_bb["min_y"], front_wall_bb["max_y"]]
            ]
        )
        bb_plane = np.c_[plane_verts_x_y, np.ones((4,)) * front_wall_bb["min_z"]]

        temp = bb_plane[0].copy()
        bb_plane[0] = bb_plane[1]
        bb_plane[1] = temp
        ax.add_collection3d(
            Poly3DCollection(
                [list(zip(bb_plane[:, 0], bb_plane[:, 2], bb_plane[:, 1]))],
                color="orange",
                alpha=0.3,
                linewidths=(0,),
            )
        )

        # Detect collision(s)
        collisions = []
        for point in self.trajectory:
            collision = (
                (front_wall_bb["min_x"] <= point[0] <= front_wall_bb["max_x"])
                and (front_wall_bb["min_y"] <= point[2] <= front_wall_bb["max_y"])
                and (front_wall_bb["min_z"] <= point[1] <= front_wall_bb["max_z"])
            )
            if collision:
                # print(f"Collision detected")
                collisions.append((point[0], point[1], point[2]))
        collisions = np.array(collisions)
        ax.scatter3D(
            collisions[:, 0],
            collisions[:, 1],
            collisions[:, 2],
            label="Collision",
            marker="x",
            color="red",
            s=100,
        )

        ax.legend()
        ax.grid(False)
        plt.show()
