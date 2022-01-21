import logging
from pathlib import Path

import pychrono as chrono
import pychrono.irrlicht as chronoirr
import pychrono.postprocess as postprocess
from tqdm import tqdm

from ai_umpire.simulation.fixed_sim_objs import *
from ai_umpire.simulation.sim_consts import (
    BACK_WALL_OUT_LINE_HEIGHT,
    COURT_LENGTH,
    PLAYER_HEIGHT,
)
from ai_umpire.simulation.textures import (
    BALL_TEXTURE_POVRAY,
    PURPLE_TEXTURE_POVRAY,
    ORANGE_TEXTURE_POVRAY,
)

__all__ = ["Simulation"]

# Ball material
BALL_MAT: chrono.ChMaterialSurfaceNSC = chrono.ChMaterialSurfaceNSC()

# Player material
PLAYER_MAT: chrono.ChMaterialSurfaceNSC = chrono.ChMaterialSurfaceNSC()
PLAYER_MAT.SetSfriction(0.1)


class Simulation:
    def __init__(
        self,
        sim_id: int,
        root: Path,
        out_file: Path,
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
        self._id: int = sim_id
        self._povray_out_file: str = f"sim_{sim_id}_povray"
        self._root: Path = root
        self.out_file: Path = out_file
        self._sys: chrono.ChSystemNSC = chrono.ChSystemNSC()
        self._time_step: float = step_sz

        self._sys.SetStep(self._time_step)

        # Initialise ball body
        self._ball: chrono.ChBodyEasySphere = chrono.ChBodyEasySphere(
            0.04, 0.5, True, True, BALL_MAT
        )
        self._ball.SetPos(ball_origin)
        self._ball.SetName("Ball")
        self._ball.SetPos_dt(ball_speed)
        self._ball.SetPos_dtdt(ball_acc)
        self._ball.AddAsset(BALL_TEXTURE_POVRAY)

        # Initialise player 1 body
        self._player1: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
            0.5, PLAYER_HEIGHT, 0.4, 50, True, True, PLAYER_MAT
        )
        self._player1.SetName("Player 1")
        p1_pos: chrono.ChVectorD = chrono.ChVectorD(
            p1_pos_x, (PLAYER_HEIGHT / 2) + 0.01, p1_pos_z
        )
        self._player1.SetPos(p1_pos)
        self._player1.SetPos_dt(p1_speed)
        self._player1.AddAsset(ORANGE_TEXTURE_POVRAY)

        # Initialise player 1's racket
        # self._p1_racket: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
        #     0.5, 0.15, 0.15, 50, True, True, PLAYER_MAT
        # )
        # p1_racket_pos: chrono.ChVectorD = chrono.ChVectorD(
        #     p1_pos_x + 0.5, 1.2, p1_pos_z
        # )
        # self._p1_racket.SetPos(p1_racket_pos)
        # self._p1_racket.SetRot(chrono.ChQuaternionD(-0.3801884, 0, 0, 0.9249091))
        # self._p1_racket.SetRot_dt(chrono.ChQuaternionD(0, 0, 0, -1))
        # self._p1_racket.SetRot_dtdt(chrono.ChQuaternionD(0, 0, 0, -1))
        # self._p1_racket.AddAsset(ORANGE_TEXTURE_POVRAY)

        # Link player 1 and player 1's racket
        # self._p1_link: chrono.ChLinkRevolute = chrono.ChLinkRevolute()
        # self._p1_link.Initialize(
        #     self._player1,
        #     self._p1_racket,
        #     True,
        #     chrono.ChFrameD(),
        #     chrono.ChFrameD(),
        # )

        # Initialise player 2 body
        self._player2: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
            0.4, PLAYER_HEIGHT - 0.3, 0.4, 50, True, True, PLAYER_MAT
        )
        self._player2.SetName("Player 2")
        p2_pos: chrono.ChVectorD = chrono.ChVectorD(
            p2_pos_x, (PLAYER_HEIGHT / 2) + 0.01, p2_pos_z
        )
        self._player2.SetPos(p2_pos)
        self._player2.SetPos_dt(p2_speed)
        self._player2.AddAsset(PURPLE_TEXTURE_POVRAY)

        # Add bodies to physics system
        self._sys.Add(self._ball)
        self._sys.Add(self._player1)
        # self._sys.Add(self._p1_racket)
        # self._sys.Add(self._p1_link)
        self._sys.Add(self._player2)
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
        self._sys.Add(HALF_COURT_LINE)
        self._sys.Add(SHORT_LINE)
        self._sys.Add(LSB_VERTICAL)
        self._sys.Add(LSB_HORIZONTAL)
        self._sys.Add(RSB_VERTICAL)
        self._sys.Add(RSB_HORIZONTAL)
        self._sys.Add(FRONT_WALL_DECAL_CENTERED)
        # self._sys.Add(FRONT_WALL_DECAL_LEFT)
        # self._sys.Add(FRONT_WALL_DECAL_RIGHT)

    def run_sim(
        self, duration: float, export: bool = True, visualise: bool = False
    ) -> None:
        if not export and not visualise:
            e: ValueError = ValueError(
                "Simulation did not run, visualise and export disabled."
            )
            logging.exception(e)
            raise e
        logging.info(f"Export enabled: {export}.")
        logging.info(f"Visualise enabled: {visualise}.")

        if export:
            logging.info("Simulating, rendering and exporting.")
            # Set up object that exports the simulation data to a format that POV-Ray can render
            pov_exporter: postprocess.ChPovRay = postprocess.ChPovRay(self._sys)
            pov_exporter.SetTemplateFile(".\\assets\\_template_POV.pov")
            pov_exporter.SetBasePath(
                str((self._root / self._povray_out_file).resolve())
            )
            pov_exporter.SetCamera(
                chrono.ChVectorD(0, 3, -11),
                chrono.ChVectorD(0, 3, 11),
                0,
            )
            pov_exporter.SetLight(
                chrono.ChVectorD(0, 7, 0), chrono.ChColor(1.2, 1.2, 1.2, 1), True
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

            # Run simulation one time step at a time exporting data for rendering at each time step
            pbar: tqdm = tqdm(
                total=duration, desc="Running simulation (showing time step)"
            )
            while self._sys.GetChTime() < duration:
                pov_exporter.ExportData()
                self._sys.DoStepDynamics(self._time_step)
                pbar.update(self._time_step)

        if visualise:
            logging.info("Visualising simulation.")
            # Visualise system with Irrlicht app
            vis_app = chronoirr.ChIrrApp(
                self._sys, "Ball Visualisation", chronoirr.dimension2du(1200, 800)
            )
            vis_app.AddTypicalCamera(
                chronoirr.vector3df(0, BACK_WALL_OUT_LINE_HEIGHT + 1.5, -COURT_LENGTH)
            )
            vis_app.AddLight(chronoirr.vector3df(-5, 90, 5), 100)
            vis_app.AddLight(chronoirr.vector3df(5, 90, 5), 100)
            vis_app.AddLight(chronoirr.vector3df(0, 90, 0), 100)
            vis_app.AddLight(chronoirr.vector3df(-5, 90, -5), 100)
            vis_app.AddLight(chronoirr.vector3df(5, 90, -5), 100)
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
