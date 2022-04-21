__all__ = [
    "WALL_THICKNESS",
    "COURT_LENGTH",
    "WALL_HEIGHT",
    "COURT_WIDTH",
    "BACK_WALL_OUT_LINE_HEIGHT",
    "TIN_HEIGHT",
    "LINE_MARKING_WIDTH",
    "PAINT_THICKNESS",
    "FRONT_WALL_OUT_LINE_HEIGHT",
    "PLAYER_HEIGHT",
    "SERVICE_LINE_HEIGHT",
    "FIELD_BOUNDING_BOXES",
    "HALF_COURT_WIDTH",
    "BB_DEPTH",
    "HALF_COURT_LENGTH",
]

# All measurements in meters
from typing import Dict

import numpy as np

WALL_THICKNESS: float = 0.03
COURT_LENGTH: float = 9.75
WALL_HEIGHT: float = 5.64
COURT_WIDTH: float = 6.4
BACK_WALL_OUT_LINE_HEIGHT: float = 2.13
SERVICE_LINE_HEIGHT: float = 1.78
TIN_HEIGHT: float = 0.43
LINE_MARKING_WIDTH: float = 0.05
PAINT_THICKNESS: float = 0.0001
FRONT_WALL_OUT_LINE_HEIGHT: float = 4.57

HALF_COURT_LENGTH: float = COURT_LENGTH / 2
HALF_COURT_WIDTH: float = COURT_WIDTH / 2

PLAYER_HEIGHT: float = 1.6
BB_DEPTH = 0.5
OUT_BB_MAX_Y = FRONT_WALL_OUT_LINE_HEIGHT + 2

# ToDo:
#  1. This dict should probably be a class or enum,  problem prone in its current state.
#  2. Check that out-bbs start from out-line - 0.5line-marking-width upwards
FIELD_BOUNDING_BOXES: Dict = {
    "front_wall": {
        "min_x": -HALF_COURT_WIDTH,
        "max_x": HALF_COURT_WIDTH,
        "min_y": TIN_HEIGHT,
        "max_y": FRONT_WALL_OUT_LINE_HEIGHT,
        "min_z": HALF_COURT_LENGTH,
        "max_z": HALF_COURT_LENGTH + BB_DEPTH,
        "colour": "blue",
        "in_out": "in",
    },
    "left_wall_in": {
        "verts": np.array(
            [
                [-HALF_COURT_WIDTH, BACK_WALL_OUT_LINE_HEIGHT, -HALF_COURT_LENGTH],
                [-HALF_COURT_WIDTH, FRONT_WALL_OUT_LINE_HEIGHT, HALF_COURT_LENGTH],
                [-HALF_COURT_WIDTH, 0, HALF_COURT_LENGTH],
                [
                    -HALF_COURT_WIDTH - BB_DEPTH,
                    BACK_WALL_OUT_LINE_HEIGHT,
                    -HALF_COURT_LENGTH,
                ],
                [
                    -HALF_COURT_WIDTH - BB_DEPTH,
                    0,
                    -HALF_COURT_LENGTH,
                ],
                [-HALF_COURT_WIDTH, 0, -HALF_COURT_LENGTH],
                [
                    -HALF_COURT_WIDTH - BB_DEPTH,
                    0,
                    HALF_COURT_LENGTH,
                ],
                [
                    -HALF_COURT_WIDTH - BB_DEPTH,
                    FRONT_WALL_OUT_LINE_HEIGHT,
                    HALF_COURT_LENGTH,
                ],
            ]
        ),
        "colour": "blue",
        "in_out": "in",
    },
    "right_wall_in": {
        "verts": np.array(
            [
                [HALF_COURT_WIDTH, BACK_WALL_OUT_LINE_HEIGHT, -HALF_COURT_LENGTH],
                [HALF_COURT_WIDTH, FRONT_WALL_OUT_LINE_HEIGHT, HALF_COURT_LENGTH],
                [HALF_COURT_WIDTH, 0, HALF_COURT_LENGTH],
                [
                    HALF_COURT_WIDTH + BB_DEPTH,
                    BACK_WALL_OUT_LINE_HEIGHT,
                    -HALF_COURT_LENGTH,
                ],
                [
                    HALF_COURT_WIDTH + BB_DEPTH,
                    0,
                    -HALF_COURT_LENGTH,
                ],
                [HALF_COURT_WIDTH, 0, -HALF_COURT_LENGTH],
                [
                    HALF_COURT_WIDTH + BB_DEPTH,
                    0,
                    HALF_COURT_LENGTH,
                ],
                [
                    HALF_COURT_WIDTH + BB_DEPTH,
                    FRONT_WALL_OUT_LINE_HEIGHT,
                    HALF_COURT_LENGTH,
                ],
            ]
        ),
        "colour": "blue",
        "in_out": "in",
    },
    "back_wall": {
        "min_x": -HALF_COURT_WIDTH,
        "max_x": HALF_COURT_WIDTH,
        "min_y": 0,
        "max_y": BACK_WALL_OUT_LINE_HEIGHT,
        "min_z": -HALF_COURT_LENGTH - BB_DEPTH,
        "max_z": -HALF_COURT_LENGTH,
        "colour": "blue",
        "in_out": "in",
    },
    # Out of court bounding boxes
    "front_wall_out": {
        "min_x": -HALF_COURT_WIDTH,
        "max_x": HALF_COURT_WIDTH,
        "min_y": FRONT_WALL_OUT_LINE_HEIGHT,
        "max_y": OUT_BB_MAX_Y,
        "min_z": HALF_COURT_LENGTH,
        "max_z": HALF_COURT_LENGTH + BB_DEPTH,
        "colour": "red",
        "in_out": "out",
    },
    "tin": {
        "min_x": -HALF_COURT_WIDTH,
        "max_x": HALF_COURT_WIDTH,
        "min_y": 0,
        "max_y": TIN_HEIGHT,
        "min_z": HALF_COURT_LENGTH,
        "max_z": HALF_COURT_LENGTH + BB_DEPTH,
        "colour": "red",
        "in_out": "out",
    },
    "back_wall_out": {
        "min_x": -HALF_COURT_WIDTH,
        "max_x": HALF_COURT_WIDTH,
        "min_y": BACK_WALL_OUT_LINE_HEIGHT,
        "max_y": OUT_BB_MAX_Y,
        "min_z": -HALF_COURT_LENGTH - BB_DEPTH,
        "max_z": -HALF_COURT_LENGTH,
        "colour": "red",
        "in_out": "out",
    },
    "left_wall_out": {
        "verts": np.array(
            [
                [-HALF_COURT_WIDTH, BACK_WALL_OUT_LINE_HEIGHT, -HALF_COURT_LENGTH],
                [-HALF_COURT_WIDTH, FRONT_WALL_OUT_LINE_HEIGHT, HALF_COURT_LENGTH],
                [-HALF_COURT_WIDTH, OUT_BB_MAX_Y, HALF_COURT_LENGTH],
                [
                    -HALF_COURT_WIDTH - BB_DEPTH,
                    BACK_WALL_OUT_LINE_HEIGHT,
                    -HALF_COURT_LENGTH,
                ],
                [
                    -HALF_COURT_WIDTH - BB_DEPTH,
                    OUT_BB_MAX_Y,
                    -HALF_COURT_LENGTH,
                ],
                [-HALF_COURT_WIDTH, OUT_BB_MAX_Y, -HALF_COURT_LENGTH],
                [
                    -HALF_COURT_WIDTH - BB_DEPTH,
                    OUT_BB_MAX_Y,
                    HALF_COURT_LENGTH,
                ],
                [
                    -HALF_COURT_WIDTH - BB_DEPTH,
                    FRONT_WALL_OUT_LINE_HEIGHT,
                    HALF_COURT_LENGTH,
                ],
            ]
        ),
        "colour": "red",
        "in_out": "out",
    },
    "right_wall_out": {
        "verts": np.array(
            [
                [HALF_COURT_WIDTH, BACK_WALL_OUT_LINE_HEIGHT, -HALF_COURT_LENGTH],
                [HALF_COURT_WIDTH, FRONT_WALL_OUT_LINE_HEIGHT, HALF_COURT_LENGTH],
                [HALF_COURT_WIDTH, OUT_BB_MAX_Y, HALF_COURT_LENGTH],
                [
                    HALF_COURT_WIDTH + BB_DEPTH,
                    BACK_WALL_OUT_LINE_HEIGHT,
                    -HALF_COURT_LENGTH,
                ],
                [
                    HALF_COURT_WIDTH + BB_DEPTH,
                    OUT_BB_MAX_Y,
                    -HALF_COURT_LENGTH,
                ],
                [HALF_COURT_WIDTH, OUT_BB_MAX_Y, -HALF_COURT_LENGTH],
                [
                    HALF_COURT_WIDTH + BB_DEPTH,
                    OUT_BB_MAX_Y,
                    HALF_COURT_LENGTH,
                ],
                [
                    HALF_COURT_WIDTH + BB_DEPTH,
                    FRONT_WALL_OUT_LINE_HEIGHT,
                    HALF_COURT_LENGTH,
                ],
            ]
        ),
        "colour": "red",
        "in_out": "out",
    },
}
