import pychrono as chrono
from ai_umpire.simulation.sim_consts import *
from ai_umpire.simulation.textures import *

__all__ = [
    "FLOOR",
    "TIN",
    "LEFT_WALL",
    "RIGHT_WALL",
    "FRONT_WALL",
    "BACK_WALL",
    "FRONT_WALL_OUT_LINE",
    "SERVICE_LINE",
    "LEFT_WALL_OUT_LINE",
    "RIGHT_WALL_OUT_LINE",
    "FRONT_WALL_DECAL_CENTERED",
    "FRONT_WALL_DECAL_LEFT",
    "FRONT_WALL_DECAL_RIGHT",
    "HALF_COURT_LINE",
    "SHORT_LINE",
    "LSB_VERTICAL",
    "LSB_HORIZONTAL",
    "RSB_VERTICAL",
    "RSB_HORIZONTAL",
]

# Floor material
FLOOR_MAT: chrono.ChMaterialSurfaceNSC = chrono.ChMaterialSurfaceNSC()
FLOOR_MAT.SetSfriction(0.7)

# Wall material
WALL_MAT: chrono.ChMaterialSurfaceNSC = chrono.ChMaterialSurfaceNSC()
WALL_MAT.SetSfriction(0.2)

FLOOR: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    COURT_WIDTH, WALL_THICKNESS, COURT_LENGTH, 1, True, True, FLOOR_MAT
)
FLOOR.SetName("Floor")
FLOOR.SetPos(chrono.ChVectorD(0, 0, 0))
FLOOR.SetBodyFixed(True)


TIN: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    COURT_WIDTH, TIN_HEIGHT, LINE_MARKING_WIDTH, 1, True, True, WALL_MAT
)
TIN.SetName("Tin")
TIN.SetPos(
    chrono.ChVectorD(0, TIN_HEIGHT / 2, (COURT_LENGTH / 2) - LINE_MARKING_WIDTH / 2)
)
TIN.SetBodyFixed(True)


LEFT_WALL: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    WALL_THICKNESS, WALL_HEIGHT, COURT_LENGTH, 1, True, True, WALL_MAT
)
LEFT_WALL.SetName("Left Wall")
LEFT_WALL.SetPos(chrono.ChVectorD(-(COURT_WIDTH / 2), WALL_HEIGHT / 2, 0))
LEFT_WALL.SetBodyFixed(True)


RIGHT_WALL: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    WALL_THICKNESS, WALL_HEIGHT, COURT_LENGTH, 1, True, True, WALL_MAT
)
RIGHT_WALL.SetName("Right Wall")
RIGHT_WALL.SetPos(chrono.ChVectorD(COURT_WIDTH / 2, WALL_HEIGHT / 2, 0))
RIGHT_WALL.SetBodyFixed(True)


FRONT_WALL: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    COURT_WIDTH, WALL_HEIGHT, WALL_THICKNESS, 1, True, True, WALL_MAT
)
FRONT_WALL.SetName("Front Wall")
FRONT_WALL.SetPos(chrono.ChVectorD(0, WALL_HEIGHT / 2, COURT_LENGTH / 2))
FRONT_WALL.SetBodyFixed(True)


BACK_WALL: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    COURT_WIDTH, BACK_WALL_OUT_LINE_HEIGHT, WALL_THICKNESS, 1, True, True, WALL_MAT
)
BACK_WALL.SetName("Back Wall")
BACK_WALL.SetPos(chrono.ChVectorD(0, BACK_WALL_OUT_LINE_HEIGHT / 2, -COURT_LENGTH / 2))
BACK_WALL.SetBodyFixed(True)


FRONT_WALL_OUT_LINE: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    COURT_WIDTH, LINE_MARKING_WIDTH, PAINT_THICKNESS, 1, True, True, WALL_MAT
)
FRONT_WALL_OUT_LINE.SetName("Front Wall Out-Line")
FRONT_WALL_OUT_LINE.SetPos(
    chrono.ChVectorD(
        0,
        FRONT_WALL_OUT_LINE_HEIGHT + (LINE_MARKING_WIDTH / 2),
        (COURT_LENGTH / 2) - (WALL_THICKNESS / 2) - PAINT_THICKNESS,
    )
)
FRONT_WALL_OUT_LINE.SetBodyFixed(True)


LEFT_WALL_OUT_LINE = chrono.ChBodyEasyBox(
    PAINT_THICKNESS, LINE_MARKING_WIDTH, 10.051, 1000, True, True, WALL_MAT
)
LEFT_WALL_OUT_LINE.SetName("Left Wall Out-Line")
LEFT_WALL_OUT_LINE.SetPos(
    chrono.ChVectorD(
        -(COURT_WIDTH / 2) + (WALL_THICKNESS / 2) + PAINT_THICKNESS,
        BACK_WALL_OUT_LINE_HEIGHT + 1.22 + (LINE_MARKING_WIDTH / 2),
        0,
    )
)
LEFT_WALL_OUT_LINE.SetRot(chrono.ChQuaternionD(0, 0, 0.1222931, 0.992494))
LEFT_WALL_OUT_LINE.SetBodyFixed(True)


RIGHT_WALL_OUT_LINE = chrono.ChBodyEasyBox(
    PAINT_THICKNESS, LINE_MARKING_WIDTH, 10.051, 1000, True, True, WALL_MAT
)
RIGHT_WALL_OUT_LINE.SetName("Right Wall Out-Line")
RIGHT_WALL_OUT_LINE.SetPos(
    chrono.ChVectorD(
        COURT_WIDTH / 2 - (WALL_THICKNESS / 2) - PAINT_THICKNESS,
        BACK_WALL_OUT_LINE_HEIGHT + 1.22 + (LINE_MARKING_WIDTH / 2),
        0,
    )
)
RIGHT_WALL_OUT_LINE.SetRot(chrono.ChQuaternionD(0, 0, 0.1222931, 0.992494))
RIGHT_WALL_OUT_LINE.SetBodyFixed(True)


SERVICE_LINE: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    COURT_WIDTH, LINE_MARKING_WIDTH, PAINT_THICKNESS, 1, True, True, WALL_MAT
)
SERVICE_LINE.SetName("Service Line")
SERVICE_LINE.SetPos(
    chrono.ChVectorD(
        0,
        SERVICE_LINE_HEIGHT + (LINE_MARKING_WIDTH / 2),
        (COURT_LENGTH / 2) - (WALL_THICKNESS / 2) - PAINT_THICKNESS,
    )
)
SERVICE_LINE.SetBodyFixed(True)


HALF_COURT_LINE: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    LINE_MARKING_WIDTH, PAINT_THICKNESS, 4.26, 1, True, True, WALL_MAT
)
HALF_COURT_LINE.SetName("Half-Court Line")
HALF_COURT_LINE.SetPos(
    chrono.ChVectorD(
        0,
        (WALL_THICKNESS / 2) + PAINT_THICKNESS,
        -2.745,
    )
)
HALF_COURT_LINE.SetBodyFixed(True)


SHORT_LINE: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    COURT_WIDTH, PAINT_THICKNESS, LINE_MARKING_WIDTH, 1, True, True, WALL_MAT
)
SHORT_LINE.SetName("Short Line")
SHORT_LINE.SetPos(
    chrono.ChVectorD(
        -WALL_THICKNESS,
        (WALL_THICKNESS / 2) + PAINT_THICKNESS,
        -0.615 + LINE_MARKING_WIDTH,
    )
)
SHORT_LINE.SetBodyFixed(True)


# Left Service Box = LSB
LSB_VERTICAL: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    LINE_MARKING_WIDTH,
    PAINT_THICKNESS,
    1.6 + LINE_MARKING_WIDTH,
    1,
    True,
    True,
    WALL_MAT,
)
LSB_VERTICAL.SetName("Left Service-Box Vertical")
LSB_VERTICAL.SetPos(
    chrono.ChVectorD(
        -1.525 - (LINE_MARKING_WIDTH / 2),
        (WALL_THICKNESS / 2) + PAINT_THICKNESS,
        -0.615 - 0.8 + (LINE_MARKING_WIDTH / 2),
    )
)
LSB_VERTICAL.SetBodyFixed(True)


# Left Service Box = LSB
LSB_HORIZONTAL: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    1.6 + LINE_MARKING_WIDTH + WALL_THICKNESS,
    PAINT_THICKNESS,
    LINE_MARKING_WIDTH,
    1,
    True,
    True,
    WALL_MAT,
)
LSB_HORIZONTAL.SetName("Left Service-Box Horizontal")
LSB_HORIZONTAL.SetPos(
    chrono.ChVectorD(
        -1.525 - 0.8 - (LINE_MARKING_WIDTH / 2) - (WALL_THICKNESS / 2),
        (WALL_THICKNESS / 2) + PAINT_THICKNESS,
        -(0.615 + 1.6) - (LINE_MARKING_WIDTH / 2),
    )
)
LSB_HORIZONTAL.SetBodyFixed(True)


# Right Service Box = RSB
RSB_HORIZONTAL: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    1.6 + LINE_MARKING_WIDTH + WALL_THICKNESS,
    PAINT_THICKNESS,
    LINE_MARKING_WIDTH,
    1,
    True,
    True,
    WALL_MAT,
)
RSB_HORIZONTAL.SetName("Right Service-Box Horizontal")
RSB_HORIZONTAL.SetPos(
    chrono.ChVectorD(
        1.525 + 0.8 + (LINE_MARKING_WIDTH / 2) + (WALL_THICKNESS / 2),
        (WALL_THICKNESS / 2) + PAINT_THICKNESS,
        -(0.615 + 1.6) - (LINE_MARKING_WIDTH / 2),
    )
)
RSB_HORIZONTAL.SetBodyFixed(True)


# Right Service Box = LSB
RSB_VERTICAL: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    LINE_MARKING_WIDTH,
    PAINT_THICKNESS,
    1.6 + LINE_MARKING_WIDTH,
    1,
    True,
    True,
    WALL_MAT,
)
RSB_VERTICAL.SetName("Right Service-Box Vertical")
RSB_VERTICAL.SetPos(
    chrono.ChVectorD(
        1.525 + (LINE_MARKING_WIDTH / 2),
        (WALL_THICKNESS / 2) + PAINT_THICKNESS,
        -0.615 - 0.8 + (LINE_MARKING_WIDTH / 2),
    )
)
RSB_VERTICAL.SetBodyFixed(True)


FRONT_WALL_DECAL_CENTERED: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    1.9, 1.9, PAINT_THICKNESS, 1, True, True, WALL_MAT
)
FRONT_WALL_DECAL_CENTERED.SetName("Front Wall Decal (Centered)")
FRONT_WALL_DECAL_CENTERED.SetPos(
    chrono.ChVectorD(
        0,
        FRONT_WALL_OUT_LINE_HEIGHT - 1.395,
        (COURT_LENGTH / 2) - (WALL_THICKNESS / 2) - PAINT_THICKNESS,
    )
)
FRONT_WALL_DECAL_CENTERED.SetBodyFixed(True)


FRONT_WALL_DECAL_LEFT: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    1.9, 1.9, PAINT_THICKNESS, 1, True, True, WALL_MAT
)
FRONT_WALL_DECAL_LEFT.SetName("Front Wall Decal (Left)")
FRONT_WALL_DECAL_LEFT.SetPos(
    chrono.ChVectorD(
        -2.1,
        FRONT_WALL_OUT_LINE_HEIGHT - 1.395,
        (COURT_LENGTH / 2) - (WALL_THICKNESS / 2) - PAINT_THICKNESS,
    )
)
FRONT_WALL_DECAL_LEFT.SetBodyFixed(True)


FRONT_WALL_DECAL_RIGHT: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    1.9, 1.9, PAINT_THICKNESS, 1, True, True, WALL_MAT
)
FRONT_WALL_DECAL_RIGHT.SetName("Front Wall Decal (Right)")
FRONT_WALL_DECAL_RIGHT.SetPos(
    chrono.ChVectorD(
        2.1,
        FRONT_WALL_OUT_LINE_HEIGHT - 1.395,
        (COURT_LENGTH / 2) - (WALL_THICKNESS / 2) - PAINT_THICKNESS,
    )
)
FRONT_WALL_DECAL_RIGHT.SetBodyFixed(True)

# Add textures that POV-Ray can render to objects
FLOOR.AddAsset(WOOD_TEXTURE_POVRAY)
TIN.AddAsset(RED_TEXTURE_POVRAY)
LEFT_WALL.AddAsset(GLASS_TEXTURE_POVRAY)
RIGHT_WALL.AddAsset(GLASS_TEXTURE_POVRAY)
FRONT_WALL.AddAsset(GLASS_TEXTURE_POVRAY)
BACK_WALL.AddAsset(GLASS_TEXTURE_POVRAY)
FRONT_WALL_OUT_LINE.AddAsset(RED_TEXTURE_POVRAY)
LEFT_WALL_OUT_LINE.AddAsset(RED_TEXTURE_POVRAY)
RIGHT_WALL_OUT_LINE.AddAsset(RED_TEXTURE_POVRAY)
SERVICE_LINE.AddAsset(RED_TEXTURE_POVRAY)
HALF_COURT_LINE.AddAsset(RED_TEXTURE_POVRAY)
SHORT_LINE.AddAsset(RED_TEXTURE_POVRAY)
LSB_VERTICAL.AddAsset(RED_TEXTURE_POVRAY)
LSB_HORIZONTAL.AddAsset(RED_TEXTURE_POVRAY)
RSB_VERTICAL.AddAsset(RED_TEXTURE_POVRAY)
RSB_HORIZONTAL.AddAsset(RED_TEXTURE_POVRAY)
FRONT_WALL_DECAL_CENTERED.AddAsset(WALL_DECAL_A_POVRAY)
FRONT_WALL_DECAL_LEFT.AddAsset(WALL_DECAL_B_POVRAY)
FRONT_WALL_DECAL_RIGHT.AddAsset(WALL_DECAL_C_POVRAY)
