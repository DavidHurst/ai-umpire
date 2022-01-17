import pychrono as chrono
import pychrono.irrlicht as chronoirr
from pathlib import Path

from ai_umpire.simulation.sim_consts import (
    BACK_WALL_OUT_LINE_HEIGHT,
    COURT_LENGTH,
    WALL_HEIGHT,
    PLAYER_HEIGHT,
)
from ai_umpire.simulation.fixed_sim_objs import *
from ai_umpire.simulation.textures import (
    BALL_TEXTURE_POVRAY,
    RED_TEXTURE_POVRAY,
    BLUE_TEXTURE_POVRAY,
)
import pychrono.postprocess as postprocess

__all__ = ["Simulation"]

# Ball material
BALL_MAT: chrono.ChMaterialSurfaceNSC = chrono.ChMaterialSurfaceNSC()

# Player material
PLAYER_MAT: chrono.ChMaterialSurfaceNSC = chrono.ChMaterialSurfaceNSC()
PLAYER_MAT.SetSfriction(0.1)


class Simulation:
    def __init__(
        self,
        step_sz: float,
        ball_origin: chrono.ChVectorD,
        ball_speed: chrono.ChVectorD,
        ball_acc: chrono.ChVectorD,
        p1_pos_x: float,
        p1_pos_z: float,
        p1_speed: chrono.ChVectorD,
        p2_pos_x: float,
        p2_pos_z: float,
        p2_speed: chrono.ChVectorD,
    ) -> None:
        self._sys: chrono.ChSystemNSC = chrono.ChSystemNSC()
        self._time_step: float = step_sz

        self._sys.SetStep(self._time_step)

        # Initialise ball body
        ball: chrono.ChBodyEasySphere = chrono.ChBodyEasySphere(
            0.04, 0.5, True, True, BALL_MAT
        )
        ball.SetPos(ball_origin)
        ball.SetName("Ball")
        ball.SetPos_dt(ball_speed)
        ball.SetPos_dtdt(ball_acc)
        ball.AddAsset(BALL_TEXTURE_POVRAY)

        # Initialise player 1 body
        player1 = chrono.ChBodyEasyBox(
            0.6, PLAYER_HEIGHT, 0.6, 50, True, True, PLAYER_MAT
        )
        player1.SetName("Player 1")
        p1_pos: chrono.ChVectorD = chrono.ChVectorD(
            p1_pos_x, (PLAYER_HEIGHT / 2) + 0.01, p1_pos_z
        )
        player1.SetPos(p1_pos)
        player1.SetPos_dt(p1_speed)
        player1.AddAsset(RED_TEXTURE_POVRAY)

        # Initialise player 2 body
        player2 = chrono.ChBodyEasyBox(
            0.6, PLAYER_HEIGHT, 0.6, 50, True, True, PLAYER_MAT
        )
        player2.SetName("Player 2")
        p2_pos: chrono.ChVectorD = chrono.ChVectorD(
            p2_pos_x, (PLAYER_HEIGHT / 2) + 0.01, p2_pos_z
        )
        player2.SetPos(p2_pos)
        player2.SetPos_dt(p2_speed)
        player2.AddAsset(BLUE_TEXTURE_POVRAY)

        # Add bodies to physics system
        self._sys.Add(ball)
        self._sys.Add(player1)
        self._sys.Add(player2)
        self._add_fixed_objects()

    def _add_fixed_objects(self):
        self._sys.Add(FLOOR)
        self._sys.Add(LEFT_WALL)
        self._sys.Add(RIGHT_WALL)
        self._sys.Add(FRONT_WALL)
        self._sys.Add(BACK_WALL)
        self._sys.Add(TIN)
        self._sys.Add(FRONT_WALL_OUT_LINE)
        self._sys.Add(LEFT_WALL_OUT_LINE)
        self._sys.Add(RIGHT_WALL_OUT_LINE)
        self._sys.Add(SERVICE_LINE)
        self._sys.Add(FRONT_WALL_DECAL_CENTERED)

    def run_sim(self, duration: float, visualise: bool = False) -> None:
        # Set up object that exports the simulation data to a format that POV-Ray can render
        pov_exporter: postprocess.ChPovRay = postprocess.ChPovRay(self._sys)
        pov_exporter.SetTemplateFile(".\\assets\\_template_POV.pov")
        pov_exporter.SetBasePath(
            str(Path("C:\\Users\\david\\Downloads\\generated_povray").resolve())
        )
        pov_exporter.SetCamera(
            chrono.ChVectorD(0, BACK_WALL_OUT_LINE_HEIGHT + 1.5, -COURT_LENGTH * 1.2),
            chrono.ChVectorD(0, WALL_HEIGHT / 2, COURT_LENGTH / 2),
            0,
        )
        pov_exporter.SetLight(
            chrono.ChVectorD(0, WALL_HEIGHT * 1.1, 0), chrono.ChColor(1, 1, 1, 1), True
        )
        pov_exporter.SetBackground(chrono.ChColor(0.2, 0.2, 0.2, 1))
        pov_exporter.SetPictureSize(854, 480)  # 1280, 720
        pov_exporter.SetAntialiasing(True, 6, 0.3)
        pov_exporter.AddAll()
        pov_exporter.ExportScript()
        """
            Commands to append to .ini file:
                Output_File_Type=J
                Quality=8
                Continue_Trace=on       # Continue rendering from last frame rendered if render was stopped.
                Work_Threads=2048
        """

        # Run simulation one time step at a time
        while self._sys.GetChTime() < duration:
            pov_exporter.ExportData()
            self._sys.DoStepDynamics(self._time_step)

        if visualise:
            # Visualise system with Irrlicht app
            vis_app = chronoirr.ChIrrApp(
                self._sys, "Ball Visualisation", chronoirr.dimension2du(1200, 800)
            )
            vis_app.AddTypicalCamera(
                chronoirr.vector3df(0, BACK_WALL_OUT_LINE_HEIGHT + 1.5, -COURT_LENGTH)
            )
            vis_app.AddTypicalLights()
            vis_app.AddShadowAll()
            vis_app.AssetBindAll()
            vis_app.AssetUpdateAll()
            vis_app.SetTimestep(self._time_step)
            vis_app.SetTryRealtime(True)
            vis_app.SetShowInfos(True)
            vis_app.SetPaused(True)
            while vis_app.GetDevice().run():
                vis_app.DoStep()
                vis_app.BeginScene()
                vis_app.DrawAll()
                vis_app.DoStep()
                vis_app.EndScene()

    def get_sim_time(self) -> float:
        return self._sys.GetChTime()

    def get_step_sz(self) -> float:
        return self._time_step
