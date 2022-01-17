import pychrono as chrono
import pychrono.irrlicht as chronoirr
from pathlib import Path
import pychrono.postprocess as postprocess

__all__ = ["SimGenerator"]

WALL_THICKNESS = 0.03
COURT_LENGTH = 9.75
WALL_HEIGHT = 5.25
COURT_WIDTH = 6.4
BACK_WALL_OUT_LINE_HEIGHT = 2.13
TIN_HEIGHT = 0.43
LINE_MARKING_WIDTH = 0.05
PAINT_THICKNESS = 0.001
FRONT_WALL_OUT_LINE_HEIGHT = 4.57

# Set location of data file to enable visualisation of the simulation.
CHRONO_DATA = str(
    Path(
        "C:\\Users\\david\\miniconda3\\pkgs\\pychrono-6.0.0-py37_0\\Library\\data\\"
    ).resolve()
)
chrono.SetChronoDataPath(CHRONO_DATA)


class SimGenerator:
    # Floor material
    FLOOR_MAT = chrono.ChMaterialSurfaceNSC()
    FLOOR_MAT.SetSfriction(0.5)

    # Wall material
    WALL_MAT = chrono.ChMaterialSurfaceNSC()
    WALL_MAT.SetSfriction(0.2)

    # Ball material
    BALL_MAT = chrono.ChMaterialSurfaceNSC()

    def __init__(self):
        self.sys = chrono.ChSystemNSC()  # Simulation system
