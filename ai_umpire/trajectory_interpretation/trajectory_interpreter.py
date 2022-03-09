from itertools import combinations, product, permutations, combinations_with_replacement

import numpy as np

__all__ = ["TrajectoryInterpreter"]

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

front_wall_min_x = -HALF_COURT_WIDTH
front_wall_max_x = HALF_COURT_WIDTH
front_wall_min_y = TIN_HEIGHT
front_wall_max_y = COURT_WALL_HEIGHT
front_wall_min_z = HALF_COURT_LENGTH
front_wall_max_z = HALF_COURT_LENGTH + 1


class TrajectoryInterpreter:
    def __init__(self):
        pass

    def visualise(self, estimated_ball_positions: np.ndarray) -> None:
        """Visualise estimated trajectory in 3D with confidence around ball position"""
        estimated_ball_positions[:, [1, 2]] = estimated_ball_positions[:, [2, 1]]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=Axes3D.name)
        ax.set(xlabel="X", ylabel="Z", zlabel="Y")
        ax.view_init(25, -70)

        ax.set_xlim(-(HALF_COURT_WIDTH + 1), HALF_COURT_WIDTH + 1)
        # Swap y and z for visualisation
        ax.set_zlim([0, COURT_WALL_HEIGHT + 1])
        ax.set_ylim(-(HALF_COURT_LENGTH + 1), HALF_COURT_LENGTH + 1)

        print(estimated_ball_positions[:5])
        estimated_ball_positions[:, 1] = estimated_ball_positions[:, 1] + 1
        print(estimated_ball_positions[:5])

        x = estimated_ball_positions[:, 0]
        y = estimated_ball_positions[:, 1]
        z = estimated_ball_positions[:, 2]

        # Plot ball trajectory
        ax.plot3D(x, y, z, "blue", label="Ball Trajectory")

        plane_x_y = np.array(
            [
                (x, y)
                for x in [front_wall_min_x, front_wall_max_x]
                for y in [front_wall_min_y, front_wall_max_y]
            ]
        )

        plane_front = np.c_[plane_x_y, np.ones((4,)) * front_wall_min_z]
        plane_rear = np.c_[plane_x_y, np.ones((4,)) * front_wall_max_z]
        bb = np.r_[plane_front, plane_rear]

        ax.scatter3D(
            plane_front[:, 0],
            plane_front[:, 2],
            plane_front[:, 1],
            label="Front Wall BB (Close)",
        )
        ax.scatter3D(
            plane_rear[:, 0],
            plane_rear[:, 2],
            plane_rear[:, 1],
            label="Front Wall BB (Far)",
        )

        # Detect collision
        det_x = []
        det_y = []
        det_z = []
        for point in estimated_ball_positions:
            collision = (
                (front_wall_min_x <= point[0] <= front_wall_max_x)
                and (front_wall_min_y <= point[2] <= front_wall_max_y)
                and (front_wall_min_z <= point[1] <= front_wall_max_z)
            )
            if collision:
                print(f"Collision detected")
                det_x.append(point[0])
                det_y.append(point[1])
                det_z.append(point[2])
        ax.scatter3D(det_x, det_y, det_z, label="Collisions", marker="x", color="red")

        ax.legend()
        plt.show()
