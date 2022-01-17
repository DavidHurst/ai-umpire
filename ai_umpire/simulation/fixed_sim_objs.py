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
LEFT_WALL_OUT_LINE.SetName("Left Wall Out Line")
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
RIGHT_WALL_OUT_LINE.SetName("Left Wall Out Line")
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

FRONT_WALL_DECAL_CENTERED: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
    4, 4, PAINT_THICKNESS, 1, True, True, WALL_MAT
)
FRONT_WALL_DECAL_CENTERED.SetName("Front Wall Decal (Centered)")
FRONT_WALL_DECAL_CENTERED.SetPos(
    chrono.ChVectorD(
        0,
        FRONT_WALL_OUT_LINE_HEIGHT - 1.22,
        (COURT_LENGTH / 2) - (WALL_THICKNESS / 2) - PAINT_THICKNESS,
    )
)
FRONT_WALL_DECAL_CENTERED.SetBodyFixed(True)

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
FRONT_WALL_DECAL_CENTERED.AddAsset(WALL_DECAL_A_POVRAY)
